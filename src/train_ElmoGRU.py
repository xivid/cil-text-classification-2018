import os
import time
import datetime
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from gensim.models import KeyedVectors
from utils.feature_extraction import line_list
from utils.io import file_type
import queue
import copy

pos_src = '../data/twitter-datasets/train_pos_full.txt'
neg_src = '../data/twitter-datasets/train_neg_full.txt'
test_src = '../data/twitter-datasets/test_data_stripped.txt'
out_dir = '../output/models/ElmoGRU/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
log_src = out_dir + "log_%s.txt" % (datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

def printl(x):
    print(x)
    with open(log_src, "a+") as f:
        f.write(x + "\n")
        
printl("Loading ELMO module...")
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

printl("Loading word2vec embeddings...")
X, Y, testX, max_tok_count = line_list(pos_src, neg_src, test_src)
Y = np.array([[1, 0] if i == 1 else [0, 1] for i in Y], dtype=np.float32)

def text2array(X):
    #ret = np.zeros((len(X), max_tok, 1024), dtype=np.float32)
    seq_len = np.zeros(len(X), dtype=np.int32)
    ret = np.array(X)
    for i, line in enumerate(X):
        seq_len[i] = len(line.split())
    return ret, seq_len

def text2vecs(X, max_tok, embedding):
    max_tok = 0
    for line in X:
        max_tok = max(len(line.split()), max_tok)

    ret = np.zeros((len(X), max_tok, embedding.vector_size), dtype=np.float32)
    #seq_len = np.zeros(len(X), dtype=np.int32)
    for i, line in enumerate(X):
        tokens = line.split()
        #seq_len[i] = len(tokens)
        for j, token in enumerate(tokens):
            if token in embedding.vocab:
                ret[i][j] = embedding[token]
    return ret#, seq_len

def batch_gen(X, Y, batch_size, n_epochs, embedding, max_tok):
    n_sample = len(X)
    batches_per_epoch = int(np.ceil(n_sample/batch_size))

    for epoch in range(n_epochs):
        shuffled_idx = np.random.permutation(np.arange(n_sample))
        shuffled_X = [X[i] for i in shuffled_idx]
        shuffled_Y = Y[shuffled_idx]
        
        start_idx = 0
        for batch in range(batches_per_epoch):
            end_idx = min(start_idx + batch_size, n_sample)
            vecX, seq_len = text2array(shuffled_X[start_idx:end_idx])
            Y_batch = shuffled_Y[start_idx:end_idx]
            
            yield vecX, Y_batch, seq_len
            start_idx += batch_size

class ElmoGRUModel():
    def __init__(self, max_tok, num_cell=512):
        self.X = tf.placeholder(tf.string, [None], name="X")
        self.seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
        self.Y = tf.placeholder(tf.float32, [None, 2], name="Y")
        self.learning_rate = tf.placeholder(tf.float32, None, name="lr")

        elmo_space = elmo(self.X, signature='default', as_dict=True)        
        elmo_word_vecs = tf.concat([elmo_space['word_emb'], elmo_space['elmo']], 2)
        
        gru_cells = tf.nn.rnn_cell.GRUCell(num_cell, activation=tf.nn.relu)
        gru_output, gru_state = tf.nn.dynamic_rnn(
                cell=gru_cells,
                inputs=elmo_word_vecs,
                dtype=tf.float32,
                sequence_length=self.seq_len,
                swap_memory=True)

        final_output = tf.layers.dropout(gru_state, rate=0.25)

        # Outputs
        with tf.name_scope("output"):
            hidden_dense = tf.layers.dense(final_output, units=64, activation=tf.nn.relu)
            hidden_dense = tf.layers.dropout(hidden_dense, rate=0.4)
            self.score = tf.layers.dense(final_output, units=2, activation=tf.nn.relu)
            self.predictions = tf.nn.softmax(self.score, name='predictions')
            self.class_prediction = tf.argmax(self.predictions, 1)
            
        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.score, labels=self.Y)
            self.loss = tf.reduce_mean(self.losses, name='loss')
        with tf.name_scope("accuracy"):
            self.correct_pred = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"), name="accuracy")


# Hyperperameters for training
val_samples = 10000
val_split = 50
n_epochs = 20
batch_size = 32
learning_rate = 1e-4
eval_every_step = 1000
output_every_step = 50
checkpoint_every_step = 2000

# Model Parameters
num_cell = 512   # <- cell size of biLSTM

# Split into training and validation
printl("Splitting dataset into training and validation...")
shuffled_idx = np.random.permutation(np.arange(len(X)))
shuffled_X = [X[i] for i in shuffled_idx]
shuffled_Y = Y[shuffled_idx]

val_X = [shuffled_X[i:min(i+val_split, val_samples)] for i in range(0, val_samples, val_split)]
val_Y = np.split(shuffled_Y[:val_samples], val_samples/val_split)
val_Y_concatenated = shuffled_Y[:val_samples]
train_X = shuffled_X[val_samples:]
train_Y = shuffled_Y[val_samples:]
testX = [testX[i:i + val_split] for i in range(0, len(testX), val_split)]

# RNN
printl("Building model...")
session_conf = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
sess = tf.Session(config=session_conf)

model = ElmoGRUModel(max_tok_count, num_cell)

global_step = tf.Variable(1, name="global_step", trainable=False)
#learning_rate = tf.train.exponential_decay(1e-5,
#                                           global_step=global_step,
#                                           decay_steps=2000,
#                                           decay_rate=0.97,
#                                           staircase=False)
optimizer = tf.train.AdamOptimizer(model.learning_rate)

# Gradient clipping
#gvs = optimizer.compute_gradients(model.loss)
#capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
#train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
train_op = optimizer.minimize(model.loss, global_step=global_step)
# 'compute_gradients' returns a list of (gradient, variable) pairs.

# Summary
out_dir_full = os.path.abspath(os.path.join(out_dir, 'runs'))
if not os.path.exists(out_dir_full):
    os.makedirs(out_dir_full)
# Summaries for loss and accuracy
loss_summary = tf.summary.scalar("loss", model.loss)
acc_summary = tf.summary.scalar("accuracy", model.accuracy)

# Train Summaries
train_summary_op = tf.summary.merge([loss_summary, acc_summary])
train_summary_dir = os.path.join(out_dir_full, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

# Validation summaries
val_avg_acc = tf.placeholder(tf.float32, name='val_avg_acc')
val_std_acc = tf.placeholder(tf.float32, name='val_std_acc')
val_avg_loss = tf.placeholder(tf.float32, name='val_avg_loss')
val_summary_op = tf.summary.merge([tf.summary.scalar('val_accuracy', val_avg_acc),
                                   tf.summary.scalar('val_std_accuracy', val_std_acc),
                                   tf.summary.scalar('val_loss', val_avg_loss)])
val_summary_dir = os.path.join(out_dir_full, "summaries", "dev")
val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
checkpoint_dir = os.path.abspath(os.path.join(out_dir_full, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables())

# Initialize all variables
printl("Initializing model variables...")
sess.run(tf.global_variables_initializer())

# change the learning rate according to the current step
def adapt_learning_rate(current_step):
    learning_rate = 1e-4
    if current_step >= 20000:
        learning_rate = 1e-5
    elif current_step >= 100000:
        s = np.random.binomial(1, 0.7)
        if s == 1:
            learning_rate = 1e-6
        else:
            learning_rate = 1e-5
    return learning_rate

def train_step(x_batch, y_batch, seq_len, current_step):
    """A single training step"""
    lr = adapt_learning_rate(current_step)
    feed_dict = {
      model.X: x_batch,
      model.Y: y_batch,
      model.seq_len: seq_len,
      model.learning_rate: lr
    }
    # add grad_summaries_merged to keep track of gradient values (time intense!)
    _, step, summaries, train_loss, train_accuracy = sess.run(
        [train_op, global_step, train_summary_op, model.loss, model.accuracy],
        feed_dict)
    train_summary_writer.add_summary(summaries, step)
    return train_loss, train_accuracy

def val_step(x_batch, y_batch, current_step):
    """Performs a model evaluation batch step on the dev set."""
    x_array, seq_len = text2array(x_batch)
    lr = adapt_learning_rate(current_step)
    feed_dict = {
      model.X: x_array,
      model.Y: y_batch,
      model.seq_len: seq_len,
      model.learning_rate: lr
    }
    _, val_loss, val_accuracy, val_pred = sess.run(
        [global_step, model.loss, model.accuracy, model.class_prediction], feed_dict)
    return val_loss, val_accuracy, val_pred

def evaluate_model(current_step):
    """Evaluates model on a dev set."""
    losses = np.zeros(len(val_X))
    accuracies = np.zeros(len(val_X))
    preds = []
    for i in range(len(val_X)):
        loss, accuracy, pred = val_step(val_X[i], val_Y[i], current_step)
        losses[i] = loss
        accuracies[i] = accuracy
        preds.append(pred)
    preds = np.concatenate(preds)

    average_loss = np.nanmean(losses)
    average_accuracy = np.nanmean(accuracies)
    std_accuracy = np.nanstd(accuracies)
    val_summary = sess.run(val_summary_op, feed_dict={val_avg_acc: average_accuracy,
                        val_std_acc: std_accuracy, val_avg_loss: average_loss})
    val_summary_writer.add_summary(val_summary, current_step)
    time_str = datetime.datetime.now().isoformat()
    printl("{}: Evaluation report at step {}:".format(time_str, current_step))
    printl("\tloss {:g}\n\tacc {:g} (stddev {:g})\n"
          "\t(Tested on the full test set)\n"
         .format(average_loss, average_accuracy, std_accuracy))
    return average_accuracy, preds

def predict():
    """Predict on the set set."""
    preds = []
    for testBatch in testX:
        ret, seq_len = text2array(testBatch)
        feed_dict = {
            model.X: ret,
            model.seq_len: seq_len
        }
        predictions = sess.run(model.class_prediction, feed_dict)
        preds.append(predictions)
    return np.concatenate(preds)

printl("Generating batches...")
batches = batch_gen(train_X, train_Y, batch_size, n_epochs, elmo, max_tok_count)

voting_candidates = []

def is_better_candidate(accuracy):
    if len(voting_candidates) < 9:
        return True
    else:
        worst = voting_candidates[0]  # (accuracy, predictions_val, predictions_test)
        if worst[0] < accuracy:
            return True
    return False

def update_voting(accuracy, predictions_val, predictions_test):
    if len(voting_candidates) < 9:
        voting_candidates.append((accuracy, predictions_val, predictions_test))
    elif accuracy > voting_candidates[0][0]:
        # replace worst
        voting_candidates[0] = (accuracy, predictions_val, predictions_test)
    # put new worst to idx 0
    min_accu = 1
    min_idx = 0
    for idx, val in enumerate(voting_candidates):
        if val[0] < min_accu:
            min_accu = val[0]
            min_idx = idx
    if min_idx != 0:
        voting_candidates[0], voting_candidates[min_idx] = voting_candidates[min_idx], voting_candidates[0]


def get_voting_individual_accuracies():
    return [x[0] for x in voting_candidates]

def get_voting_validations():
    return [x[1] for x in voting_candidates]

def get_voting_tests():
    return [x[2] for x in voting_candidates]

def evaluate_voting():
    printl("Evaluating the voting of 9 models with individual accuracies " + str(get_voting_individual_accuracies()))
    vals = get_voting_validations()
    voted_predictions = [0] * val_samples
    for predictions in vals:
        sample_no = 0
        for p in predictions:
            voted_predictions[sample_no] += (1 if p == 0 else -1)
            sample_no += 1
    correct_count = 0
    for i in range(val_samples):
        voted_predictions[i] = 1 if voted_predictions[i] > 0 else -1
        if voted_predictions[i] == val_Y_concatenated[i][0] or voted_predictions[i] == -val_Y_concatenated[i][1]:
            correct_count += 1
    voted_accuracy = correct_count / val_samples
    return voted_accuracy

def predict_voting(submission_file):
    tests = get_voting_tests()
    voted_predictions = [0] * val_samples
    for predictions in tests:
        sample_no = 0
        for p in predictions:
            voted_predictions[sample_no] += (1 if p == 0 else -1)
            sample_no += 1
    with open(submission_file, "w+") as f:
        f.write("Id,Prediction\n")
        sample_no = 1
        for p in voted_predictions:
            f.write("{},{}\n".format(sample_no, (1 if p > 0 else -1)))
            sample_no += 1
    printl("saved to %s" % submission_file)

# Training
printl("Training started")
current_step = 0
try:
    lcum = 0
    acum = 0
    for batchX, batchY, seq_len in batches:
        l, a = train_step(batchX, batchY, seq_len, current_step)
        lcum += l
        acum += a
        current_step = tf.train.global_step(sess, global_step)

        if current_step % output_every_step == 0:
            time_str = datetime.datetime.now().isoformat()
            printl("{}: step {}, loss {:g}, acc {:g}".format(time_str, current_step, lcum / output_every_step, acum / output_every_step))
            lcum = 0
            acum = 0
        if current_step % eval_every_step == 0:
            printl("\nEvaluating...")
            eval_start_ms = int(time.time() * 1000)
            current_accuracy, predictions_val = evaluate_model(current_step)
            eval_time_ms = int(time.time() * 1000) - eval_start_ms
            printl("Evaluation performed in {0}ms.".format(eval_time_ms))
            if is_better_candidate(current_accuracy):
                predictions_test = predict()
                update_voting(current_accuracy, predictions_val, predictions_test)
                # voting list updated, evaluate model with voting, then predict with voting
                ensemble_accuracy = evaluate_voting()
                printl("Voting list updated, evaluation accuracy with ensemble: {}".format(ensemble_accuracy))
                submission_file = out_dir + "kaggle_%s_accu%f.csv" % (datetime.datetime.now().strftime("%Y%m%d%H%M%S"), ensemble_accuracy)
                printl("Generating submission file: %s" % submission_file)
                predict_voting(submission_file)
        if current_step % checkpoint_every_step == 0:
            printl("Save model parameters...")
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            printl("Saved model checkpoint to {}\n".format(path))

    if current_step == 0:
        printl("No steps performed.")
    else:
        printl("\n\nFinished all batches. Performing final evaluations.")

    printl("Performing final evaluation...")
    evaluate_model(current_step)

    printl("Performing final checkpoint...")
    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
    printl("Saved model checkpoint to {}\n".format(path))

    # Evaluate test data
    printl("Final evaluating on test set")
    submission_file = out_dir + "kaggle_final_%s.csv" % (datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    printl("Generating submission file: %s" % submission_file)
    predict_voting(submission_file)

except KeyboardInterrupt:
    if current_step is None:
        printl("No checkpointing to do.")
    else:
        printl("Training interrupted. Performing final checkpoint.")
        printl("Press C-c again to forcefully interrupt this.")
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        printl("Saved model checkpoint to {}\n".format(path))
    submission_file = out_dir + "kaggle_final_%s.csv" % (datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    printl("Generating submission file: %s" % submission_file)
    predict_voting(submission_file)


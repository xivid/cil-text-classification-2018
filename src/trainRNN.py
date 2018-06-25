import os
import time
import datetime
import tensorflow as tf
import numpy as np
from gensim.models import KeyedVectors
from utils.feature_extraction import token_array
pos_src = '../data/train_pos.txt'
neg_src = '../data/train_pos.txt'
test_src = '../data/test_data_stripped.txt'
out_dir = '../output'
embedding_src = 'datasources/word2vec_embedding.txt'

embedding = KeyedVectors.load_word2vec_format(embedding_src)
X, Y, testX, max_tok_count = token_array(pos_src, neg_src, test_src)
Y = np.array([[1, 0] if i == 1 else [0, 1] for i in Y], dtype=np.float32)

def text2vecs(X, max_tok, embedding):
    ret = np.zeros((X.shape[0], max_tok, embedding.vector_size), dtype=np.float32)
    seq_len = np.zeros(X.shape[0], dtype=np.int32)
    for i, line in enumerate(X):
        seq_len[i] = len(line)
        for j, token in enumerate(line):
            if token in embedding.wv.vocab:
                ret[i][j] = embedding.wv[token]
    return ret, seq_len

def batch_gen(X, Y, batch_size, n_epochs, embedding, max_tok):
    n_sample = X.shape[0]
    batches_per_epoch = int(np.ceil(n_sample/batch_size))

    for epoch in range(n_epochs):
        shuffled_idx = np.random.permutation(np.arange(n_sample))
        shuffled_X = X[shuffled_idx]
        shuffled_Y = Y[shuffled_idx]
        
        start_idx = 0
        for batch in range(batches_per_epoch):
            end_idx = min(start_idx + batch_size, n_sample)
            vecX, seq_len = text2vecs(shuffled_X[start_idx:end_idx], max_tok, embedding)
            Y_batch = shuffled_Y[start_idx:end_idx]
            
            yield vecX, Y_batch, seq_len
            start_idx += batch_size

class LSTMModel():
    def __init__(self, embedding_dim, max_tok):
        self.X = tf.placeholder(tf.float32, [None, max_tok, embedding_dim], name="input_X")
        self.seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
        self.Y = tf.placeholder(tf.float32, [None, 2], name="input_Y")
        lstm_cells = tf.nn.rnn_cell.LSTMCell(128)
        _, lstm_final_state = tf.nn.dynamic_rnn(
                cell=lstm_cells,
                inputs=self.X,
                dtype=tf.float32,
                #sequence_length=self.seq_len,
                swap_memory=True)
        #print(lstm_final_state)
        final_output = tf.layers.dropout(lstm_final_state.h, rate=0.5)
        #final_output = lstm_final_state.h

        # Outputs
        with tf.name_scope("output"):
            self.score = tf.layers.dense(final_output, units=2)
            self.predictions = tf.nn.softmax(self.score, name='predictions')
            self.class_prediction = tf.argmax(self.predictions, 1)
            
        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.score, labels=self.Y)
            self.loss = tf.reduce_mean(self.losses, name='loss')
        with tf.name_scope("accuracy"):
            self.correct_pred = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"), name="accuracy")

# Hyperperameters
val_samples = 10000
val_split = 50
n_epochs = 100
batch_size = 64
learning_rate = 1e-4
eval_every_step = 1000
output_every_step = 50
checkpoint_every_step = 1000

# Split into training and validation
shuffled_idx = np.random.permutation(np.arange(X.shape[0]))
shuffled_X = X[shuffled_idx]
shuffled_Y = Y[shuffled_idx]

val_X = np.split(shuffled_X[:val_samples], val_split)
val_Y = np.split(shuffled_Y[:val_samples], val_split)
train_X = shuffled_X[val_samples:]
train_Y = shuffled_Y[val_samples:]

# RNN
session_conf = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
sess = tf.Session(config=session_conf)
embedding_dim = embedding.vector_size

model = LSTMModel(embedding_dim, max_tok_count)

global_step = tf.Variable(1, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate)
gvs = optimizer.compute_gradients(model.loss)
capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
#train_op = optimizer.minimize(model.loss, global_step=global_step)
# 'compute_gradients' returns a list of (gradient, variable) pairs.

# Summary
out_dir_full = os.path.abspath(os.path.join(out_dir, 'runs'))
if not os.path.exists(out_dir_full):
    os.mkdir(out_dir_full)
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
sess.run(tf.global_variables_initializer())

def train_step(x_batch, y_batch, seq_len):
    """A single training step"""
    feed_dict = {
      model.X: x_batch,
      model.Y: y_batch,
      model.seq_len: seq_len
    }
    # add grad_summaries_merged to keep track of gradient values (time intense!)
    _, step, summaries, train_loss, train_accuracy = sess.run(
        [train_op, global_step, train_summary_op, model.loss, model.accuracy],
        feed_dict)
    train_summary_writer.add_summary(summaries, step)
    return train_loss, train_accuracy

def val_step(x_batch, y_batch):
    """Performs a model evaluation batch step on the dev set."""
    x_vec, seq_len = text2vecs(x_batch, max_tok_count, embedding)
    feed_dict = {
      model.X: x_vec,
      model.Y: y_batch,
      model.seq_len: seq_len
    }
    _, val_loss, val_accuracy = sess.run(
        [global_step, model.loss, model.accuracy], feed_dict)
    return val_loss, val_accuracy

def evaluate_model(current_step):
    """Evaluates model on a dev set."""
    losses = np.empty(val_split)
    accuracies = np.empty(val_split)

    for i in range(val_split):
        loss, accuracy = val_step(val_X[i], val_Y[i])
        losses[i] = loss
        accuracies[i] = accuracy

    average_loss = np.nanmean(losses)
    average_accuracy = np.nanmean(accuracies)
    std_accuracy = np.nanstd(accuracies)
    val_summary = sess.run(val_summary_op, feed_dict={val_avg_acc: average_accuracy,
                        val_std_acc: std_accuracy, val_avg_loss: average_loss})
    val_summary_writer.add_summary(val_summary, current_step)
    time_str = datetime.datetime.now().isoformat()
    print("{}: Evaluation report at step {}:".format(time_str, current_step))
    print("\tloss {:g}\n\tacc {:g} (stddev {:g})\n"
          "\t(Tested on the full test set)\n"
         .format(average_loss, average_accuracy, std_accuracy))

batches = batch_gen(train_X, train_Y, batch_size, n_epochs, embedding, max_tok_count)

# Training
current_step = None
try:
    lcum = 0
    acum = 0
    for batchX, batchY, seq_len in batches:
        l, a = train_step(batchX, batchY, seq_len)
        lcum += l
        acum += a
        current_step = tf.train.global_step(sess, global_step)

        if current_step % output_every_step == 0:
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, current_step, lcum / output_every_step, acum / output_every_step))
            lcum = 0
            acum = 0
        if current_step % eval_every_step == 0:
            print("\nEvaluating...")
            eval_start_ms = int(time.time() * 1000)
            evaluate_model(current_step)
            eval_time_ms = int(time.time() * 1000) - eval_start_ms
            print("Evaluation performed in {0}ms.".format(eval_time_ms))

        if current_step % checkpoint_every_step == 0:
            print("Save model parameters...")
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))

    if current_step is None:
        print("No steps performed.")
    else:
        print("\n\nFinished all batches. Performing final evaluations.")

    print("Performing final evaluation...")
    evaluate_model(current_step)

    print("Performing final checkpoint...")
    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
    print("Saved model checkpoint to {}\n".format(path))

    # Evaluate test data
    with open("submission.csv", "w+") as f:
        testX = np.split(testX, val_split)
        sample_no = 1
        for testBatch in testX:
            ret, seq_len = text2vecs(testBatch, max_tok_count, embedding)
            feed_dict = {
                    model.X: ret,
                    model.seq_len: seq_len
            }
            predictions = sess.run([model.class_prediction], feed_dict)
            for p in predictions:
                f.write("{},{}\n".format(sample_no, (1 if p == 0 else -1)))
                sample_no += 1

except KeyboardInterrupt:
    if current_step is None:
        print("No checkpointing to do.")
    else:
        print("Training interrupted. Performing final checkpoint.")
        print("Press C-c again to forcefully interrupt this.")
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))


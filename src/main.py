# Read the embedding vectors from file and train the classifier

# This file is still under construction.. TODO 

import numpy as np

pos_default = "../data/twitter-datasets/train_pos.txt"
neg_default = "../data/twitter-datasets/train_neg.txt"
test_default = "../data/twitter-datasets/test_data.txt"
preprocessed_data = "./preprocessed.npz"

data = np.load(preprocessed_data)

# 3. split dataset to training set and validation set
X_train, X_val, y_train, y_val = train_test_split(data['X'], data['y'], test_size=0.33, random_state=42)

# Train the model (need to be modified) TODO
print("Training " + sys.argv[1] + "model on positive dataset " + pos_default +
      " and negative dataset " + neg_default)
model = models.getModel(sys.argv[1])()
model.train(X_train, y_train)

# 4. compute and output the validation error [TODO]


# 5. Preprocess test data into feature vectors [TODO]


# Evaluate for Kaggle submission (need to be modified) TODO
print("Evaluating for kaggle submission on test set " + test_default)
      utils.ensure_dir("../output/models/%s" % sys.argv[1])
model.evaluate_for_kaggle(test_default,
                          "../output/models/%s/kaggle_%s.csv" % (sys.argv[1], datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
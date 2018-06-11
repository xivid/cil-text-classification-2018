class BaseModel(object):
    def train(self, pos_src, neg_src):
        raise NotImplementedError("The train() method need to be implemented be subclasses!")

    def predict(self, test_src):
        raise NotImplementedError("The predict() method need to be implemented be subclasses!")

    def evaluate_for_kaggle(self, test_src, output_src):
        predictions = self.predict(test_src)
        with open(output_src, "w") as f:
            f.write("Id,Prediction\n")
            for idx, val in enumerate(predictions):
                f.write("%d,%d\n" % (idx + 1, val))
        print("Submission file saved: " + output_src)
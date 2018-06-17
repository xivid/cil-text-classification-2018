# Models

Model definitions. Every model should inherit the `BaseModel` class. Refer to the structure of `BaseModel`.

Existing models (don't refer to any of them because they don't adhere to our datasource-model structure yet):
- Naive Bayes: Multinomial Naive Bayes model, the feature vector of a sample is simply the number of occurrences of each word in the whole corpus.
- SVM: Doesn't work yet. Training is very slow due to the poor complexity and parallelism of SVM.    
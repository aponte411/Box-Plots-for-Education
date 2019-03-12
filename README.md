# Box-Plots-for-Education
Greetings! This repository contains my work for the Box-Plots for Education Data Science Competition. Below is an executive summary of the project.


# Executive Summary

The goal of this project is to correctly predict the probability that a certain label is attached to a budget line item. The task is a multiclass multilabel classification problem. The project was inspired by DrivenData’s Box-Plots for Education Data Science Competition. The competition goal is to help schools automate their budgeting system. The dataset is produced by a third party company (ERS) and is composed of a training set with labels 400,027x25, a test set without labels 50,064x16, and a sample of a submission file 50,064x104. For submitting files to DrivenData.org for leaderboard scoring, a submission file must be 50,064x104, where 50, 064 is the number of rows in the test dataset and 104 is the number of columns. The evaluation metric being used is multi-multiclass log loss.

The projected started with EDA that lead to key findings about the data. Some of which were: 
- The dataset is large enough that we can use a simple holdout method for validation. This will allow me to see how well the model generalizes on new data. I decided not to go with a more computationally expensive method such as K-fold cross validation because it took far too long. 
- The majority of the features are free form text, therefore the bulk of this task is a natural language processing task. This means that most of the feature engineering will be converting text to numbers using feature extraction algorithms. 
There is a large amount of labels for each class. Since this task is a multiclass multilabel problem, we must carefully resample our data during validation so that each set is representative and not imbalanced. 

Starting with the first finding. I had to deal with the large size of this dataset by using a 10% (40,000 rows) sample to quickly experiment and analyze the data. The sample needed to be representative of the original dataset. I chose to split the training data into two equal parts (a 50% split), but needed to ensure that the classes in each group (training set and validation set) had target labels with distributions similar to the original training set given. Using a simple stratification strategy didn't work due to the many labels - stratification still resulted in imbalanced sets - so I had to create helper functions to split the data set correctly. Once the dataset was appropriately split I could then conduct preprocessing and feature engineering. 

# Feature Engineering

Preprocessing consisted of aggregating all of the features into a single vector. The text then needed to be tokenized on punctuation, missing values were imputed with a space, words were lowercased, and then vectorized using TFIDF (TfidfVectorizer) and the hashing trick (HashingVectorizer). This resulted in a large sparse matrix with over a million features.

I then explored dimensionality reduction techniques using scikit learn functions such as SelectKBest in conjunction with a chi-squared test. The chi-squared test will calculate F-scores and rank the features according to their importance. The optimal cutoff point was found through hyperparameter tuning on the validation set. Ultimately, the dimensionality reduction methods did not result in lower log loss scores, so I decided to use all of the extracted features.

# Modeling

The next step was to develop a modeling approach that could handle sparse features without taking too long to train. Through hyperparameter tuning on the validation set, I found that linear models had the lowest log loss; specifically, an ensemble of logistic regression models optimized via SGD (stochastic gradient descent) with differing levels of regularization. Various other models were tested such as tree-based models and neural networks, but I found that linear models produced the lowest log loss.

I then decided to label encode the target labels via scikit learn’s LabelEncoder. The result was a matrix with 9 target columns. The procedure was as follows:
- Iterate through the 9 target columns and build 3 classifiers for every class (column). 
- Instantiate 3 models with varying levels of regularization: L1, L2, a low alpha.
- Horizontally stack the predictions after every iteration.
- Average the predictions when it is finished.

I chose to label encode the target labels column wise so that I could fit a single classifier on each class. But instead of fitting one classifier for each class, I fit 3 separate classifiers with differing levels of regularization per class. One classifier had L1 regularization for feature selection (making some of the coefficients zero); L2 regularization to make some of the coefficients small; and a low penalty term (alpha) to further regularize and prevent overfitting. After making predictions, the results were averaged to produce the final probabilities. Using both holdout validation and the leaderboard scores, I found that this combination of regularization alongside optimization via SGD resulted in probabilities that produced the lowest log loss results. I relied upon holdout validation and the leaderboard scores because cross validation became too difficult to implement without crashing my computer. 

# Conclusion & Recommendations

After testing many ideas I found that simple models worked best. I tested various complex algorithms that required a lot of computational power (such as a neural network) but found that they produced suboptimal results compared to the simpler methods such as the logistic regression and SGD. The difficulty was the size of this dataset and the time it took train. Aside from the goal of winning the competition, the business goal of automating and improving the way schools manage their budgets can be met with the following recommendations:
- The practical goal is to have a production ready model. Using complex ensembles or complex algorithms may produce marginal gains for winning the competition but for putting a model into production, it's practically useless. We need a model that can train relatively quickly given the high volume and velocity of the data. Therefore, I recommend that the production model be a simple linear model optimized via SGD or mini-batch SGD since calculating the gradients in a big data setting can take a very long time, so it helps to not use the entire training data to update the weights but one observation at a time. 
- I would explore more creative ways to generate features from the free-form text. Some ideas are: finding word similarities, interactions between words, text statistics, and more combinations of feature extraction methods. I found promising results with sparse interactions included in the model but chose not to include them because the models took far too long to train. But future analyses should explore making more information rich features, alongside optimizing the model.

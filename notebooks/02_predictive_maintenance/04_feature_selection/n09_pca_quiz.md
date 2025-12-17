# Principal Component Analysis (PCA) Tutorial - Evaluation Questionnaire

## Student Information
- **Name**: ________________________________
- **Date**: ________________________________
- **Course**: Applications of AI for Industrial Control
- **Tutorial**: Principal Component Analysis for Dimensionality Reduction

---

## Section 1: Theoretical Understanding (30 points)

### 1.1 Basic Concepts (15 points)

**Question 1** (5 points): What is the main purpose of Principal Component Analysis (PCA)?
- a) To increase the number of features in a dataset
- b) To reduce the dimensionality of data while preserving most of the variance
- c) To classify data into different categories
- d) To clean missing values from the dataset

**Question 2** (5 points): Before applying PCA, what preprocessing step is typically recommended?
- a) Remove all categorical variables
- b) Convert all features to integers
- c) Standardize/normalize the features (mean=0, std=1)
- d) Sort the data by target variable

**Question 3** (5 points): What do Principal Components represent?
- a) The original features of the dataset
- b) New orthogonal axes that capture maximum variance in the data
- c) The target variables we want to predict
- d) The outliers in the dataset

### 1.2 Variance and Information (15 points)

**Question 4** (5 points): If PCA explains 85% of the variance with 2 components out of 10 original features, what does this mean?
- a) We lost 85% of the information
- b) We kept 85% of the original information using only 2 features instead of 10
- c) The remaining 8 features are more important
- d) The PCA failed because we need 100% variance

**Question 5** (5 points): Which of the following is NOT a common application of PCA?
- a) Data visualization (reducing to 2D or 3D)
- b) Data compression
- c) Noise reduction
- d) Adding new features to increase model complexity

**Question 6** (5 points): In machine learning pipelines, when is PCA typically applied?
- a) After training the model
- b) As a preprocessing step before training
- c) Only during model evaluation
- d) After making predictions

---

## Section 2: Machine Learning Applications (25 points)

### 2.1 Practical Implementation (25 points)

**Question 7** (8 points): How can PCA help with the "curse of dimensionality" in machine learning?
- a) It adds more features to make the problem easier
- b) It reduces the number of dimensions, making algorithms more efficient and less prone to overfitting
- c) It increases the training time to get better results
- d) It has no effect on the curse of dimensionality

**Question 8** (8 points): When using PCA as preprocessing for a classification algorithm, what should you be careful about?
- a) Apply PCA to the entire dataset including test data before splitting
- b) Only apply PCA to categorical features
- c) Fit PCA only on training data, then transform both training and test data
- d) Apply different PCA transformations to training and test data

**Question 9** (9 points): What is a potential disadvantage of using PCA in machine learning?
- a) It always improves model performance
- b) It makes the model faster to train
- c) It reduces interpretability since principal components are combinations of original features
- d) It eliminates all noise from the data

---

## Section 3: Industrial Applications (25 points)

### 3.1 Scenario Analysis (25 points)

**Question 10** (15 points): In which scenario would PCA be most beneficial for machine learning?
- a) Dataset with 5 features and 1000 samples
- b) Dataset with 1000 features and 100 samples (high-dimensional, small sample size)
- c) Dataset with only categorical variables
- d) Dataset with perfect linear relationships between all features

**Question 11** (10 points): Describe a specific industrial control scenario where PCA would be beneficial. Include: (1) the type of high-dimensional data you would have, (2) why dimensionality reduction would be needed, and (3) how the reduced dimensions would improve the control system.

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________

---

## Section 4: Critical Thinking (20 points)

### 4.1 Implementation Strategy (20 points)

**Question 12** (20 points): You are implementing a predictive maintenance system for a manufacturing plant with 50 different sensor measurements from each machine. The plant operators want to understand which sensors are most important and visualize the overall machine health status. 

Explain how you would use PCA in this scenario:
1. How would you prepare the data for PCA?
2. How would you determine the optimal number of principal components?
3. How would you interpret the results for the operators?
4. What are the potential limitations of using PCA in this application?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________

---

## Scoring Summary

| Section | Points Earned | Total Points |
|---------|---------------|--------------|
| 1. Theoretical Understanding | _____ / 30 | 30 |
| 2. Machine Learning Applications | _____ / 25 | 25 |
| 3. Industrial Applications | _____ / 25 | 25 |
| 4. Critical Thinking | _____ / 20 | 20 |
| **Total** | **_____ / 100** | **100** |

**Grade**: ____________

**Comments**:
_______________________________________________________________________________
_______________________________________________________________________________

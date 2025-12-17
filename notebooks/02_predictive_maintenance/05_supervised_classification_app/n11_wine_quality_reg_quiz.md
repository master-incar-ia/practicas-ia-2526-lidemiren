# Wine Quality Regression Tutorial - Evaluation Questionnaire

## Student Information
- **Name**: ________________________________
- **Date**: ________________________________
- **Course**: Applications of AI for Industrial Control
- **Tutorial**: Wine Quality Regression for Continuous Quality Assessment

---

## Section 1: Theoretical Understanding (40 points)

### 1.1 Regression vs Classification (20 points)

**Question 1** (5 points): Why is regression better than multiclass classification for wine quality prediction in industrial quality control?
- a) Regression models are faster to train
- b) Regression provides continuous, precise quality scores rather than discrete categories, enabling more nuanced quality assessment
- c) Classification models require more data
- d) Regression uses less computational resources

**Question 2** (5 points): What is the primary difference in output between classification and regression for wine quality?
- a) Classification: 3.2, 5.7, 6.8; Regression: 3, 5, 6, 7
- b) Classification: 3, 4, 5, 6, 7, 8, 9; Regression: 3.2, 5.7, 6.8, 7.3
- c) Both produce the same output format
- d) Classification uses probabilities, regression uses integers

**Question 3** (5 points): Which loss function is most appropriate for the wine quality regression task?
- a) Cross-Entropy Loss
- b) Mean Squared Error (MSE)
- c) Binary Cross-Entropy
- d) Hinge Loss

**Question 4** (5 points): Which activation function is used in the output layer of the regression neural network?
- a) ReLU
- b) Sigmoid
- c) Softmax
- d) Linear (no activation)

### 1.2 Model Architecture and Training (20 points)

**Question 5** (5 points): What is the purpose of scaling both features (X) and target (y) in the regression model?
- a) To make training faster only
- b) To facilitate neural network convergence and improve gradient flow for both inputs and outputs
- c) To reduce memory usage
- d) To prevent overfitting only

**Question 6** (5 points): The WineQualityRegressor includes dropout layers for:
- a) Faster training
- b) Uncertainty quantification and overfitting prevention
- c) Better gradient flow
- d) Memory optimization

**Question 7** (5 points): What is the purpose of the `predict_with_uncertainty()` method in the regression model?
- a) To make predictions faster
- b) To provide confidence intervals and uncertainty estimates for predictions
- c) To improve model accuracy
- d) To reduce computational cost

**Question 8** (5 points): What does the difference between MSE and MAE as regression metrics represent?
- a) MSE and MAE are identical
- b) MSE penalizes large errors more heavily (squared), MAE treats all errors equally (absolute)
- c) MSE is for classification, MAE is for regression
- d) MAE is always larger than MSE

---

## Section 2: Model Evaluation (30 points)

### 2.1 Regression Metrics (15 points)

**Question 9** (5 points): Which metric is most appropriate for evaluating the overall performance of a regression model?
- a) Accuracy
- b) F1-Score
- c) R² (coefficient of determination)
- d) Precision

**Question 10** (5 points): What does an R² score of 0.85 indicate?
- a) 85% of data points are correctly classified
- b) 85% of the variance in quality scores is explained by the model
- c) The model has 15% error rate
- d) 85% accuracy in predictions

**Question 11** (5 points): The residual analysis in regression helps to:
- a) Speed up training
- b) Identify prediction patterns and model bias
- c) Reduce model complexity
- d) Increase accuracy

### 2.2 Industrial Applications (15 points)

**Question 12** (15 points): In the context of continuous quality scoring, what advantage does regression provide for industrial applications? Provide specific examples of how precise quality measurements benefit manufacturing processes.

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________

---

## Section 3: Practical Implementation (20 points)

### 3.1 Uncertainty Quantification (10 points)

**Question 13** (10 points): Describe the uncertainty quantification approach used in the wine quality regressor. How does the model provide confidence intervals for its predictions and why is this important for industrial applications?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________

### 3.2 Loss Function Selection (10 points)

**Question 14** (10 points): Compare the advantages and disadvantages of using MSE vs MAE as loss functions for wine quality regression. When would you choose one over the other in an industrial setting?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________

---

## Section 4: Critical Thinking (10 points)

### 4.1 Industrial Implementation Strategy (10 points)

**Question 15** (10 points): You're deploying this regression model in a winery that produces 10,000 bottles per day. The quality team wants to use your model to optimize blending decisions (mixing different batches to achieve target quality scores). Design a strategy that uses the regression model's continuous predictions and uncertainty estimates to make optimal blending recommendations.

_Your answer:_
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
| 1. Theoretical Understanding | _____ / 40 | 40 |
| 2. Model Evaluation | _____ / 30 | 30 |
| 3. Practical Implementation | _____ / 20 | 20 |
| 4. Critical Thinking | _____ / 10 | 10 |
| **Total** | **_____ / 100** | **100** |

**Grade**: ____________

**Comments**:
_______________________________________________________________________________
_______________________________________________________________________________

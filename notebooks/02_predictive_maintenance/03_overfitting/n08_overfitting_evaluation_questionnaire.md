# Overfitting and Underfitting Tutorial - Evaluation Questionnaire

## Student Information
- **Name**: ________________________________
- **Date**: ________________________________
- **Course**: Applications of AI for Industrial Control
- **Tutorial**: Overfitting and Underfitting in Neural Networks

---

## Section 1: Fundamentals (25 points)

### 1.1 Basic Concepts (25 points)

**Question 1** (5 points): What is the primary characteristic of an overfitted model?
- a) High training error and high validation error
- b) Low training error and high validation error
- c) High training error and low validation error
- d) Low training error and low validation error

**Question 2** (5 points): Which of the following is NOT a common regularization technique?
- a) Dropout
- b) L2 regularization
- c) Early stopping
- d) Increasing model complexity

**Question 3** (5 points): Adding more training data can help reduce overfitting.
- **True/False**: ___________

_Explanation:_
_______________________________________________________________________________

**Question 4** (5 points): What does the bias-variance trade-off describe?
- a) The relationship between model accuracy and training time
- b) The balance between model simplicity and flexibility
- c) The trade-off between training and validation loss
- d) The relationship between dataset size and model performance

**Question 5** (5 points): Fill in the blanks: A model with high bias and low variance is likely to exhibit __________, while a model with low bias and high variance is likely to exhibit __________.

_Your answer:_
_______________________________________________________________________________

---

## Section 2: Noise and Dataset Size Effects (15 points)

### 2.1 Model Architecture and Data Complexity (15 points)

**Question 6** (8 points): When training a neural network on noisy data, which model architecture is most likely to overfit?
- a) Simple model (5 neurons)
- b) Medium model (32-16 neurons)
- c) Complex model (128-64-32 neurons)
- d) All models overfit equally

**Question 7** (7 points): With a very small dataset (e.g., 20 samples), which strategy is most effective?
- a) Use a complex model with heavy regularization
- b) Use a simple model with light regularization
- c) Increase the number of training epochs
- d) Remove regularization entirely

---

## Section 3: Regularization Techniques (18 points)

### 3.1 Regularization Methods (18 points)

**Question 8** (8 points): Compare the effectiveness of dropout vs L2 regularization. When would you choose one over the other?

_Your answer:_
- Dropout: _______________________________________________________________________________
- L2 regularization: ______________________________________________________________________
- When to use each: ______________________________________________________________________

**Question 9** (10 points): Early stopping can be considered a form of regularization.
- **True/False**: ___________

_Explanation:_
_______________________________________________________________________________
_______________________________________________________________________________

---

## Section 4: Industrial Applications (15 points)

### 4.1 Practical Implementation (15 points)

**Question 10** (8 points): In an industrial monitoring system using neural networks, list three strategies you would implement to prevent overfitting when sensor data is noisy and limited.

_Strategies:_
1. _______________________________________________________________________________
2. _______________________________________________________________________________
3. _______________________________________________________________________________

**Question 11** (7 points): For a real-time industrial control system, which factor is most critical when selecting a model?
- a) Highest possible accuracy on training data
- b) Balance between accuracy and generalization
- c) Fastest training time
- d) Most complex architecture

---

## Section 5: Critical Analysis (20 points)

### 5.1 Calculation and Analysis (10 points)

**Question 12** (10 points): Given the following training results for a neural network:

```
Training MSE: 0.05
Validation MSE: 0.25
Test MSE: 0.30
```

Calculate the overfitting ratio (Validation MSE / Training MSE) and interpret what this indicates about the model.

**Overfitting Ratio:** ___________

_Interpretation:_
_______________________________________________________________________________
_______________________________________________________________________________

### 5.2 Problem Solving (10 points)

**Question 13** (10 points): You are developing a neural network for predictive maintenance in a manufacturing plant. The training dataset is small (100 samples) and noisy due to sensor limitations.

Describe your complete strategy for:
1. Model architecture selection
2. Regularization approach
3. Training procedure
4. Validation strategy

_Your answer:_
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
| 1. Fundamentals | _____ / 25 | 25 |
| 2. Noise and Dataset Size Effects | _____ / 15 | 15 |
| 3. Regularization Techniques | _____ / 18 | 18 |
| 4. Industrial Applications | _____ / 15 | 15 |
| 5. Critical Analysis | _____ / 20 | 20 |
| **Total** | **_____ / 93** | **93** |

**Grade**: ____________

**Comments**:
_______________________________________________________________________________
_______________________________________________________________________________



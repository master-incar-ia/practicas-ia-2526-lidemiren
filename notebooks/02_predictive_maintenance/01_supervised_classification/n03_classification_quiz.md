# Classification Tutorial - Evaluation Questionnaire

## Student Information
- **Name**: ________________________________
- **Date**: ________________________________
- **Course**: Applications of AI for Industrial Control
- **Tutorial**: Classification with Iris Dataset using PyTorch

---

## Section 1: Theoretical Understanding (30 points)

### 1.1 Basic Concepts (10 points)

**Question 1** (2 points): What is the main difference between classification and regression tasks?
- a) Classification predicts continuous values, regression predicts categories
- b) Classification predicts categories, regression predicts continuous values
- c) Classification uses neural networks, regression uses linear models
- d) There is no difference between them

**Question 2** (2 points): In the Iris dataset tutorial, what are the parameters w and b in a neural network layer `y = wx + b`?
- w: ________________________________
- b: ________________________________

**Question 3** (3 points): List three evaluation metrics used in the classification tutorial and briefly explain when each is most important:
1. ________________________________
2. ________________________________
3. ________________________________

**Question 4** (3 points): What is the purpose of feature scaling (StandardScaler) in our classification pipeline? Explain why this preprocessing step is crucial for neural networks.

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

### 1.2 Training Process (10 points)

**Question 5** (4 points): Describe the four main steps of the PyTorch training loop for classification:
1. ________________________________
2. ________________________________
3. ________________________________
4. ________________________________

**Question 6** (3 points): What is CrossEntropyLoss and why is it preferred over MSE for multi-class classification problems?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

**Question 7** (3 points): Explain the difference between training accuracy and test accuracy. Why might test accuracy be lower than training accuracy?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

### 1.3 Model Architecture (10 points)

**Question 8** (4 points): Compare the three model architectures explored in the tutorial:

| Architecture | Number of Parameters | Best Use Case | Activation Functions |
|--------------|---------------------|---------------|---------------------|
| Linear-only  |                     |               |                     |
| Hidden layers|                     |               |                     |
| Deep network |                     |               |                     |

**Question 9** (3 points): What is overfitting in classification? How can you detect it using the confusion matrix and accuracy metrics shown in the tutorial?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

**Question 10** (3 points): If your model shows 95% training accuracy but only 75% test accuracy, what does this indicate about your model?

_Your answer:_
_______________________________________________________________________________

---

## Section 2: Practical Application (25 points)

### 2.1 Dataset Analysis (10 points)

**Question 11** (5 points): About the Iris dataset used in the tutorial:
- a) How many classes? ________________________________
- b) How many features? ________________________________
- c) Which species is most easily separable? ________________________________
- d) Which features are most discriminative? ________________________________
- e) Why is this dataset good for learning classification? ________________________________

**Question 12** (5 points): Explain how the decision boundary visualization helps understand the classification process. What do the colored regions represent?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

### 2.2 Evaluation Metrics (15 points)

**Question 13** (5 points): Calculate accuracy from this confusion matrix:

```
           Predicted
Actual   Setosa  Versicolor  Virginica
Setosa      15       0          0
Versicolor   0      12          3  
Virginica    0       2         13
```

**Accuracy = ____________**

**Show calculation:** ________________________________________________

**Question 14** (10 points): A quality control classifier for industrial parts has:
- Correctly identified defective parts: 45
- False alarms (good parts classified as defective): 8
- Missed defects (defective parts classified as good): 5
- Correctly identified good parts: 142

**Calculate:**
- Precision = ____________
- Recall = ____________
- F1-Score = ____________

**Which metric is most critical for safety and why?**
_______________________________________________________________________________

---

## Section 3: Problem Solving (25 points)

### 3.1 Scenario Analysis (15 points)

**Question 15** (7 points): You're training a 3-class classifier and observe:
- Training accuracy increases to 98% over 200 epochs
- Test accuracy peaks at 85% after 50 epochs, then decreases
- Confusion matrix shows perfect classification on training data

Analyze this scenario:
1. Is the model learning properly? ________________________________
2. What problem is occurring? ________________________________
3. At what epoch should you stop training? ________________________________
4. What techniques could prevent this issue? ________________________________

**Question 16** (8 points): A student tests different neural network architectures on the Iris dataset:
- Linear model (no hidden layers): Test accuracy = 95%, 13 parameters
- 1 hidden layer (10 neurons): Test accuracy = 97%, 63 parameters  
- 3 hidden layers (50 neurons each): Test accuracy = 93%, 3013 parameters

Which model would you recommend and why? Consider performance, complexity, and generalization.

_Your recommendation and justification:_
_______________________________________________________________________________
_______________________________________________________________________________

### 3.2 Code Understanding (10 points)

**Question 17** (5 points): In the PyTorch classification implementation, what does this code sequence accomplish?
```python
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
```

_Your explanation:_
_______________________________________________________________________________
_______________________________________________________________________________

**Question 18** (5 points): Why do we use `torch.max(outputs, 1)` in classification? What do the outputs represent before and after this operation?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

---

## Section 4: Critical Thinking (20 points)

### 4.1 Real-World Applications (10 points)

**Question 19** (5 points): Give three examples of industrial classification scenarios where this tutorial's concepts apply, and explain the specific classification task.

1. ________________________________
   Classification task: ________________________________

2. ________________________________
   Classification task: ________________________________

3. ________________________________
   Classification task: ________________________________

**Question 20** (5 points): What limitations of the Iris dataset might not represent real industrial classification challenges? How would you adapt the approach for industrial applications?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

### 4.2 Reflection and Learning (10 points)

**Question 21** (5 points): What was the most valuable insight you gained from the interactive visualizations (decision boundaries, confusion matrices, training curves)? How did they enhance your understanding?

_Your reflection:_
_______________________________________________________________________________
_______________________________________________________________________________

**Question 22** (5 points): If you were to teach classification to another student, what would be the three most important concepts to emphasize and why?

1. ________________________________
   Why: ________________________________

2. ________________________________
   Why: ________________________________

3. ________________________________
   Why: ________________________________

---

## Scoring Summary

| Section | Points Earned | Total Points |
|---------|---------------|--------------|
| 1. Theoretical Understanding | _____ / 30 | 30 |
| 2. Practical Application | _____ / 25 | 25 |
| 3. Problem Solving | _____ / 25 | 25 |
| 4. Critical Thinking | _____ / 20 | 20 |
| **Total** | **_____ / 100** | **100** |

**Grade**: ____________

**Grading Scale:**
- A (90-100): Excellent understanding of classification concepts
- B (80-89): Good grasp of theory and practice
- C (70-79): Satisfactory understanding with some gaps
- D (60-69): Basic understanding, needs improvement
- F (<60): Insufficient understanding, requires additional study

**Comments**:
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________

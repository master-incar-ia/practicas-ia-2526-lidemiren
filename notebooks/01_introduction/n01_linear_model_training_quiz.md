# Linear Model Training Tutorial - Evaluation Questionnaire

## Student Information
- **Name**: ________________________________
- **Date**: ________________________________
- **Course**: Applications of AI for Industrial Control
- **Tutorial**: Linear Model Training with PyTorch

---

## Section 1: Theoretical Understanding (30 points)

### 1.1 Basic Concepts (10 points)

**Question 1** (2 points): What is the mathematical relationship that our synthetic dataset follows?
- a) y = x + 2
- b) y = 2x + 1
- c) y = x² + 1
- d) y = 2x² + 1

**Question 2** (2 points): In the linear model `y = wx + b`, what do the parameters `w` and `b` represent?
- w: ________________________________
- b: ________________________________

**Question 3** (3 points): List three types of loss functions explored in the tutorial and briefly explain when each might be preferred:
1. ________________________________
2. ________________________________
3. ________________________________

**Question 4** (3 points): What is the purpose of adding Gaussian noise to our synthetic dataset? Explain why this makes the problem more realistic.

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

### 1.2 Training Process (10 points)

**Question 5** (4 points): Describe the four main steps of the training loop in order:
1. ________________________________
2. ________________________________
3. ________________________________
4. ________________________________

**Question 6** (3 points): What is backpropagation and why is it essential for training neural networks?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

**Question 7** (3 points): Explain the difference between training loss and test loss. Why might test loss be higher than training loss?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

### 1.3 Model Architecture (10 points)

**Question 8** (4 points): Compare the three model architectures explored in the tutorial:

| Architecture | Number of Parameters | Best Use Case | Potential Issues |
|--------------|---------------------|---------------|------------------|
| Linear       |                     |               |                  |
| Polynomial   |                     |               |                  |
| Multi-layer  |                     |               |                  |

**Question 9** (3 points): What is overfitting? How can you detect it using the metrics shown in the tutorial?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

**Question 10** (3 points): If the overfitting ratio (test loss / train loss) is 2.5, what does this indicate about your model?

_Your answer:_
_______________________________________________________________________________

---

## Section 2: Practical Application (25 points)

### 2.1 Parameter Analysis (10 points)

**Question 11** (5 points): In the interactive parameter exploration, if you set weight=1.0 and bias=2.0:
- a) What would be the model equation? ________________________________
- b) How would this compare to the target function y = 2x + 1? ________________________________
- c) Would you expect high or low loss? Why? ________________________________

**Question 12** (5 points): Explain how the 3D loss landscape visualization helps understand the optimization process. What does the "valley" or minimum point represent?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

### 2.2 Hyperparameter Tuning (15 points)

**Question 13** (5 points): What happens to training when the learning rate is:
- Too high: ________________________________
- Too low: ________________________________
- Just right: ________________________________

**Question 14** (10 points): If your model shows the following metrics after training, what would you conclude?
- Training R²: 0.95
- Test R²: 0.65
- Overfitting ratio: 2.8

_Your diagnosis and recommendations:_
_______________________________________________________________________________
_______________________________________________________________________________

---

## Section 3: Problem Solving (25 points)

### 3.1 Scenario Analysis (15 points)

**Question 15** (7 points): You're training a linear model and observe that:
- Training loss decreases steadily for 50 epochs then plateaus
- Test loss decreases for 30 epochs then starts increasing
- Final parameters: weight=1.95, bias=1.02

Analyze this scenario:
1. Is the model learning properly? ________________________________
2. Are there any concerns? ________________________________
3. What actions would you take? ________________________________
4. Are the final parameters reasonable? ________________________________

**Question 16** (8 points): A student tries different architectures on the same dataset:
- Linear model: Test R² = 0.88, 2 parameters
- Polynomial (degree 3): Test R² = 0.75, 4 parameters  
- Neural Network (50 hidden units): Test R² = 0.72, 151 parameters

Which model would you recommend and why? Consider performance, complexity, and interpretability.

_Your recommendation and justification:_
_______________________________________________________________________________
_______________________________________________________________________________

### 3.2 Code Understanding (10 points)

**Question 17** (5 points): In the PyTorch linear model implementation, what does this code accomplish?
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

_Your explanation:_
_______________________________________________________________________________
_______________________________________________________________________________

**Question 18** (5 points): Why do we use `model.eval()` and `torch.no_grad()` during evaluation? What would happen if we didn't use them?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

---

## Section 4: Critical Thinking (20 points)

### 4.1 Real-World Applications (10 points)

**Question 19** (5 points): Give three examples of real-world industrial control scenarios where linear regression might be applicable, and explain why.

1. ________________________________
   Explanation: ________________________________

2. ________________________________
   Explanation: ________________________________

3. ________________________________
   Explanation: ________________________________

**Question 20** (5 points): What limitations of linear models might make them insufficient for complex industrial control tasks? Suggest alternative approaches.

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

### 4.2 Reflection and Learning (10 points)

**Question 21** (5 points): What was the most valuable insight you gained from the interactive visualizations? How did they enhance your understanding compared to traditional static plots?

_Your reflection:_
_______________________________________________________________________________
_______________________________________________________________________________

**Question 22** (5 points): If you were to teach this material to another student, what would be the three most important concepts to emphasize and why?

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

**Comments**:
_______________________________________________________________________________
_______________________________________________________________________________


# Complex Function Fitting - Evaluation Questionnaire

## Student Information
- **Name**: ______________________________
- **Date**: ______________________________
- **Course**: Applications of AI for Industrial Control

---

## Section 1: Theoretical Understanding (20 points)

### Question 1 (5 points)
What is the mathematical relationship of the target function used in this tutorial?
- a) y = 3*sin(x) + 2*cos(x) - 0.5
- b) y = 3*sin(x) + 2*cos(10x) - 0.5
- c) y = sin(3x) + cos(2x) - 0.5
- d) y = 3x + 2*cos(10x) - 0.5

### Question 2 (5 points)
Why is a simple linear model insufficient for fitting the target function? 
1. ______________________________
2. ______________________________
3. ______________________________

### Question 3 (5 points)
Complete the ComplexNet architecture table:

| Component | Purpose | Configuration |
|-----------|---------|---------------|
| Input Layer | | |
| Hidden Layers | | |
| Activation Functions | | |
| Output Layer | | |

### Question 4 (5 points)
How does increasing hidden layers affect model performance? What are the trade-offs?

*Answer:*
__________________________________________________________________________
__________________________________________________________________________

---

## Section 2: Practical Application (25 points)

### Question 5 (8 points)
How does noise level affect training? What would happen if noise increased from 0.5 to 2.0?

*Answer:*
__________________________________________________________________________
__________________________________________________________________________

### Question 6 (7 points)
Why might the model perform differently on training range (-4, 4) versus extrapolation beyond this range?

*Answer:*
__________________________________________________________________________
__________________________________________________________________________

### Question 7 (10 points)
Analyze hyperparameter effects:

| Hyperparameter | Too Small | Optimal | Too Large |
|----------------|-----------|---------|-----------|
| Hidden Size | | | |
| Learning Rate | | | |


### Question 8 (10 points)
You observe: Training loss = 0.001, Test loss increases after epoch 500, Training R² = 0.98, Test R² = 0.75

**Diagnosis:** __________________________________________________________________________

**Solutions:** __________________________________________________________________________

---

## Section 3: Problem Solving (25 points)

### Question 9 (12 points)
**Scenario:** You need to fit a function with higher frequency components: y = sin(x) + 0.5sin(20x) + 0.2sin(100x)

How would you modify the neural network architecture?

**Network depth:** __________________________________________________________________________

**Network width:** __________________________________________________________________________

**Training strategy:** __________________________________________________________________________

**Justification:** __________________________________________________________________________

### Question 10 (8 points)
A neural network produces unrealistic oscillations between data points despite low training/test losses.

**Possible causes:** __________________________________________________________________________

**Solutions:** __________________________________________________________________________

### Question 11 (5 points)
Suggest three optimization strategies for faster training without changing model architecture:

1. __________________________________________________________________________
2. __________________________________________________________________________
3. __________________________________________________________________________

---

## Section 4: Industrial Applications (20 points)

### Question 12 (10 points)
Provide three real-world industrial scenarios where complex non-linear function fitting is essential:

1. **Scenario:** __________________________________________________________________________
   **Why needed:** __________________________________________________________________________

2. **Scenario:** __________________________________________________________________________
   **Why needed:** __________________________________________________________________________

3. **Scenario:** __________________________________________________________________________
   **Why needed:** __________________________________________________________________________



## Grading Scale

| Section | Points | Grade |
|---------|--------|-------|
| Section 1: Theoretical Understanding | 20 | ___/20 |
| Section 2: Practical Application | 25 | ___/25 |
| Section 3: Problem Solving | 25 | ___/25 |
| Section 4: Industrial Applications | 20 | ___/20 |
| **Total** | **90** | **___/90** |

**Grade Scale:**
- A: 81-90 points (Excellent)
- B: 72-80 points (Good)
- C: 63-71 points (Satisfactory)
- D: 54-62 points (Needs improvement)
- F: Below 54 points (Insufficient)

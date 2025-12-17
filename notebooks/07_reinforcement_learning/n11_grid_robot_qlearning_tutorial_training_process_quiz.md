# Q-Learning Grid Robot Navigation Tutorial - Evaluation Questionnaire

## Student Information
- **Name**: ________________________________
- **Date**: ________________________________
- **Course**: Applications of AI for Industrial Control
- **Tutorial**: Q-Learning for 2D Grid Robot Navigation

---

## Section 1: Basic Concepts (25 points)

### 1.1 Environment Understanding (15 points)

**Question 1** (3 points): What is the main goal of the robot in the grid world environment?
- a) Reach the goal position while avoiding traps
- b) Visit every cell in the grid exactly once
- c) Reach the goal position in the minimum number of steps
- d) Stay in the start position and avoid moving

**Question 2** (6 points): In the 5x5 grid world setup, identify the correct configuration:
- Start position: ________________________________
- Goal position: ________________________________
- Goal reward: ________________________________
- Trap positions: ________________________________
- Step cost: ________________________________

**Question 3** (3 points): What does the Q-value Q(s,a) represent in Q-learning?
- a) The immediate reward for taking action a in state s
- b) The expected cumulative reward for taking action a in state s and following the optimal policy thereafter
- c) The probability of taking action a in state s
- d) The number of times action a has been taken in state s

**Question 4** (3 points): What are the four possible actions available to the robot in the grid world?
1. ________________________________
2. ________________________________
3. ________________________________
4. ________________________________

### 1.2 Basic Q-Learning Mechanics (10 points)

**Question 5** (5 points): What happens when the robot tries to move outside the grid boundaries?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

---

## Section 2: Q-Learning Algorithm (30 points)

### 2.1 Mathematical Foundations (15 points)

**Question 6** (10 points): Complete the Q-learning update rule equation:

Q(s, a) ← Q(s, a) + α [________________________________]

Where:
- s: ________________________________
- a: ________________________________
- α: ________________________________
- γ: ________________________________
- r: ________________________________

**Question 7** (5 points): Explain the role of each hyperparameter in Q-learning:

**Learning rate (α = 0.1):**
_______________________________________________________________________________

**Discount factor (γ = 0.9):**
_______________________________________________________________________________

**Initial epsilon (ε = 1.0):**
_______________________________________________________________________________

### 2.2 Training Process (15 points)

**Question 8** (10 points): Describe the four main steps of the Q-learning training loop in order:
1. ________________________________
2. ________________________________
3. ________________________________
4. ________________________________

**Question 9** (5 points): What is the epsilon-greedy policy and why is it important?

_Definition:_
_______________________________________________________________________________

_Importance:_
_______________________________________________________________________________

---

## Section 3: Exploration vs Exploitation (20 points)

### 3.1 Strategy Understanding (15 points)

**Question 10** (5 points): If epsilon = 0.2, what percentage of the time will the agent:
- Explore (random action): ________________________________%
- Exploit (best known action): ________________________________%

**Question 11** (5 points): Why does epsilon typically decay during training?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

**Question 12** (5 points): What would happen if we set:
- **Learning rate α = 0** : ________________________________
- **Discount factor γ = 0** : ________________________________

### 3.2 Performance Analysis (5 points)

**Question 13** (5 points): Why might the success rate be low in early episodes but improve over time?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

---

## Section 4: Training Process and Results (25 points)

### 4.1 Performance Monitoring (15 points)

**Question 14** (10 points): What information is tracked during training to monitor the agent's performance?

List at least 4 metrics:
1. ________________________________
2. ________________________________
3. ________________________________
4. ________________________________

**Question 15** (5 points): What does it mean when the Q-values "converge" during training?

_Your answer:_
_______________________________________________________________________________
_______________________________________________________________________________

### 4.2 Critical Analysis (10 points)

**Question 16** (10 points): You observe during training that:
- Episode rewards increase steadily for 200 episodes then plateau
- Success rate reaches 85% by episode 150
- Q-values for optimal state-action pairs stabilize around episode 180
- Epsilon has decayed from 1.0 to 0.1

Analyze this scenario and explain what this indicates about the learning process:

_Your analysis:_
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________

---

## Scoring Summary

| Section | Points Earned | Total Points |
|---------|---------------|--------------|
| 1. Basic Concepts | _____ / 25 | 25 |
| 2. Q-Learning Algorithm | _____ / 30 | 30 |
| 3. Exploration vs Exploitation | _____ / 20 | 20 |
| 4. Training Process and Results | _____ / 25 | 25 |
| **Total** | **_____ / 100** | **100** |

**Grade**: ____________

**Comments**:
_______________________________________________________________________________
_______________________________________________________________________________

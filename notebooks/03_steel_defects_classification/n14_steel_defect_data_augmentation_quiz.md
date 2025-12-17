# Steel Defect Classification - Comprehensive Evaluation Questionnaire

## Student Information
- **Name**: ________________________________
- **Date**: ________________________________
- **Course**: Applications of AI for Industrial Control
- **Tutorial**: Steel Defect Classification with Data Augmentation

## 📋 Instructions

This quiz covers concepts from all three steel defect classification notebooks:
- **Notebook 12**: Binary Classification (Defective vs Non-defective)
- **Notebook 13**: Multiclass Classification (Multiple defect types)  
- **Notebook 14**: Data Augmentation for Performance Improvement

**Total Questions**: 20  
**Time**: Approximately 30-45 minutes  
**Scoring**: Each question is worth 5 points (100 points total)

---

## Section 1: Binary Classification Fundamentals (25 points)

### 1.1 Classification Basics (15 points)

**Question 1** (5 points): In the binary classification notebook, which CIFAR-10 classes were used to simulate steel defect detection?

A) Classes 0 (airplane) as defective and 1 (automobile) as non-defective  
B) Classes 5 (dog) as defective and 9 (truck) as non-defective  
C) All 10 classes were used equally  
D) Classes 2 (bird) and 8 (ship) only  

**Question 2** (5 points): What loss function was used in the binary classification model and why?

A) CrossEntropyLoss because it works best for all classification problems  
B) BCELoss (Binary Cross Entropy) because it's specifically designed for binary classification  
C) MSELoss because it measures the distance between predictions and labels  
D) HingeLoss because it's used in SVMs  

**Question 3** (5 points): What is the key difference in the output layer between binary and multiclass classification models?

A) Binary uses 1 output neuron with sigmoid, multiclass uses multiple neurons with softmax  
B) Binary uses multiple outputs, multiclass uses one output  
C) Both use the same architecture  
D) Binary uses ReLU activation, multiclass uses sigmoid  

### 1.2 Data Preprocessing (10 points)

**Question 4** (5 points): In the binary classification notebook, how was the CIFAR-10 data converted for the binary task?

A) All images were converted to grayscale  
B) Only specific classes were selected and labels were converted to 0 (non-defective) and 1 (defective)  
C) Images were resized to 64x64 pixels  
D) Data was normalized to [-1, 1] range only  

**Question 5** (5 points): Generally, how does multiclass classification performance compare to binary classification?

A) Multiclass always performs better due to more detailed categories  
B) Binary typically achieves higher accuracy as it's a simpler decision boundary  
C) Performance is always identical  
D) Multiclass is faster to train

---

## Section 2: Multiclass Classification (25 points)

### 2.1 Architecture and Loss Functions (15 points)

**Question 6** (5 points): How many different steel defect types were simulated in the multiclass classification notebook?

A) 2 types (defective/non-defective)  
B) 5 types of defects  
C) 10 types including various defects and good steel  
D) 8 types excluding good steel  

**Question 7** (5 points): What loss function is used for multiclass classification and what does it include internally?

A) BCELoss with manual softmax implementation  
B) CrossEntropyLoss which includes softmax internally  
C) MSELoss for measuring classification errors  
D) Multiple BCELoss functions, one for each class  

**Question 8** (5 points): In multiclass classification, what does the confusion matrix diagonal represent?

A) Misclassified samples  
B) Correctly classified samples for each class  
C) Total number of samples per class  
D) The model's confidence scores  

### 2.2 Performance Analysis (10 points)

**Question 9** (10 points): Compare binary and multiclass classification performance for the steel defect task. Explain which typically achieves higher accuracy and why:

_Your analysis:_
_______________________________________________________________________________
_______________________________________________________________________________

---

## Section 3: Data Augmentation Fundamentals (25 points)

### 3.1 Augmentation Concepts (15 points)

**Question 10** (5 points): What is the primary purpose of data augmentation in industrial vision systems?

A) To reduce the computational cost of training neural networks  
B) To artificially expand the dataset size and improve model generalization  
C) To compress images for faster processing  
D) To convert RGB images to grayscale  

**Question 11** (5 points): In the augmentation comparison study, what metric was used to measure overfitting?

A) The validation accuracy alone  
B) The training time duration  
C) The gap between training and validation accuracy  
D) The total number of parameters in the model  

**Question 12** (5 points): Which augmentation techniques would be MOST appropriate for steel defect classification?

A) Extreme rotation (±90 degrees) and color inversion  
B) Random rotation (±15°), brightness/contrast adjustment, and horizontal flip  
C) Converting all images to black and white  
D) Random noise addition and image blurring only  

### 3.2 Training Performance (10 points)

**Question 13** (5 points): According to the notebook results, what is typically the training time overhead when using data augmentation?

A) No additional time  
B) 10-50% increase in training time  
C) 200% increase in training time  
D) Decreased training time  

**Question 14** (5 points): According to the notebooks, which metric showed the clearest improvement when using data augmentation?

A) Training accuracy only  
B) Validation accuracy and reduced overfitting gap  
C) Training loss only  
D) Model inference speed

## ⚙️ PART 4: IMPLEMENTATION AND PRACTICAL APPLICATIONS

### Question 13: PyTorch Implementation Best Practice

**In PyTorch, what is the correct way to apply augmentation only during training and not during validation/testing?**

A) Apply the same transforms to all datasets  
B) Use different transform pipelines: augmented transforms for training, basic transforms for validation/testing  
C) Apply augmentation after training is complete  
D) Use augmentation only on the test set  

<details>
<summary>Show Answer</summary>

**✅ Correct Answer: B**

**Explanation**: Different transform pipelines should be used - augmented transforms for training data to provide variety and robustness, and basic transforms (normalization only) for validation/testing to ensure consistent evaluation.

</details>

---

### Question 14: Industrial Real-World Simulation

**For a steel manufacturing line with high-speed cameras, which augmentation technique would simulate real-world conditions?**

A) Random color inversion  
B) Motion blur and slight perspective transforms  
C) Extreme scaling (10x zoom)  
D) Image rotation by 180 degrees  

<details>
<summary>Show Answer</summary>

**✅ Correct Answer: B**

**Explanation**: Motion blur simulates the effect of fast-moving steel sheets, and slight perspective transforms account for camera angle variations. These are realistic conditions in high-speed manufacturing environments.

</details>

---

### Question 15: Model Architecture Considerations

**Why was dropout used in addition to data augmentation in the notebooks?**

A) To replace data augmentation entirely  
B) To provide additional regularization and prevent overfitting  
C) To speed up training  
D) To reduce the number of parameters  

<details>
<summary>Show Answer</summary>

**✅ Correct Answer: B**

**Explanation**: Dropout provides additional regularization by randomly setting some neurons to zero during training, which complements data augmentation in preventing overfitting and improving generalization.

</details>

---

## Section 4: Implementation and Practical Applications (25 points)

### 4.1 PyTorch Implementation (15 points)

**Question 15** (5 points): In PyTorch, what is the correct way to apply augmentation only during training and not during validation/testing?

A) Apply the same transforms to all datasets  
B) Use different transform pipelines: augmented transforms for training, basic transforms for validation/testing  
C) Apply augmentation after training is complete  
D) Use augmentation only on the test set  

**Question 16** (5 points): When implementing data augmentation for actual steel defect detection, which consideration is MOST critical?

A) Using as many augmentation techniques as possible  
B) Ensuring augmentations don't alter critical defect characteristics  
C) Applying the same augmentation to all industrial domains  
D) Using only geometric transformations  

**Question 17** (5 points): Why was dropout used in addition to data augmentation in the notebooks?

A) To replace data augmentation entirely  
B) To provide additional regularization and prevent overfitting  
C) To speed up training  
D) To reduce the number of parameters  

### 4.2 Real-World Applications (10 points)

**Question 18** (5 points): In the multiclass notebook, what challenge would real steel defect datasets likely face?

A) All defect types occur with equal frequency  
B) Class imbalance - some defects are much rarer than others  
C) Too many samples for each defect type  
D) Perfect data quality with no labeling errors  

**Question 19** (5 points): Based on all three notebooks, what would be the recommended progression for implementing a real steel defect classification system?

A) Start with multiclass, then add augmentation, finally simplify to binary  
B) Start with binary classification, extend to multiclass, then optimize with augmentation  
C) Use only augmentation without considering classification type  
D) Implement all approaches simultaneously without testing  

---

## Scoring Summary

| Section | Points Earned | Total Points |
|---------|---------------|--------------|
| 1. Binary Classification Fundamentals | _____ / 25 | 25 |
| 2. Multiclass Classification | _____ / 25 | 25 |
| 3. Data Augmentation Fundamentals | _____ / 25 | 25 |
| 4. Implementation and Practical Applications | _____ / 25 | 25 |
| **Total** | **_____ / 100** | **100** |

**Grade**: ____________

**Comments**:
_______________________________________________________________________________
_______________________________________________________________________________

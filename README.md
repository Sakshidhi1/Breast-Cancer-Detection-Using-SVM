# 🔍Breast-Cancer-Detection-Using-SVM

This project demonstrates how to use Support Vector Machines (SVM) for binary classification on the Breast Cancer Wisconsin dataset. The model distinguishes between malignant and benign tumors using scikit-learn’s SVM classifier with linear and RBF kernels.

## 📂 Dataset

- *Source*: [Kaggle - Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- *Features*: 30 numeric features related to tumor characteristics.
- *Target*: Diagnosis (M = malignant, B = benign), encoded as 0 and 1.

## 🧪 Steps Followed

### 1. Load and Prepare the Dataset
- Used the Breast Cancer Wisconsin dataset (available via scikit-learn or Kaggle)
- Converted string labels to numerical: Malignant = 0, Benign = 1
- Scaled features using StandardScaler
- Split data into training and test sets

### 2. Train SVM Models
- Trained two classifiers:
  - *Linear Kernel*
  - *RBF Kernel*
- Evaluated accuracy on test data

### 3. Visualize Decision Boundary
- Used PCA to reduce features to 2D for visualization
- Visualized the SVM decision regions using Matplotlib

### 4. Tune Hyperparameters
- Used GridSearchCV to find optimal values for:
  - C: Regularization parameter
  - gamma: Kernel coefficient (for RBF)
- Performed 5-fold cross-validation

### 5. Final Evaluation
- Evaluated the best model using cross-validation
- Calculated mean accuracy

 ## 📈 Results

- *Linear Kernel Accuracy:* 0.9766  
- *RBF Kernel Accuracy:* 0.9708  
- *Best Parameters:* {'C': 10, 'gamma': 0.01}  
- *Best CV Score:* 0.9673  
- *Cross-validation Scores:* [0.9737, 0.9737, 0.9825, 0.9737, 0.9912]  
- *Mean Accuracy:* 0.9789

 ## 📊 Visualization & Interpretation

After reducing the 30-dimensional dataset using PCA to two components, the decision boundary of the SVM classifier was visualized in 2D.

![decision_boundary](https://github.com/user-attachments/assets/da671f77-aff3-4906-bac7-4cfe16705299)

### 🔍 Interpretation:

- *Dots* represent tumor samples:
  - 🔴 Red: Malignant
  - 🔵 Blue: Benign
- *Background colors* show how the SVM classifies new data:
  - Red region: Classified as malignant
  - Blue region: Classified as benign
- The *curved decision boundary* is learned by the SVM with RBF kernel, effectively capturing the nonlinear separation between classes.
- Overlapping points near the boundary indicate samples that are harder to classify.
- The plot demonstrates that the model is performing well and can generalize to unseen data after dimensionality reduction.

  ## 🧠 Technologies Used

- Python
- Scikit-learn
- Matplotlib
- NumPy
- Jupyter Notebook


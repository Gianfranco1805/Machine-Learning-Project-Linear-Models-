# Project 1: Linear Models - Student Performance Predictor

## 🎯 Purpose of the Project
This project explores the fundamental mathematics and practical application of linear models in Machine Learning. The primary goal is to build an end-to-end machine learning pipeline—from data preprocessing to model evaluation—using a student performance dataset. 

To bridge the gap between theory and application, core algorithms (Linear Regression and Logistic Regression) were implemented **entirely from scratch using NumPy** to calculate gradients and optimize loss functions manually. These custom implementations were then mathematically verified against industry-standard models built with **PyTorch**.

## 🛠️ Tools & Technologies Used
* **Python**: Core programming language.
* **NumPy**: Used for matrix operations, vectorized mathematics, and manually programming Gradient Descent step-by-step.
* **PyTorch**: Used for building verification models (`nn.Linear`), utilizing robust loss functions (`MSELoss`, `BCEWithLogitsLoss`), and automated optimization (`SGD`, `Adam`).
* **Pandas**: Used for data loading, cleaning, manipulation, and extracting features.
* **Matplotlib & Seaborn**: Created rich visualizations for exploratory data analysis (EDA), loss convergence curves, polynomial fits, and 2D decision boundaries.
* **Scikit-Learn**: Utilized specifically for the `PolynomialFeatures` transformation.

## 📊 Key Results & Insights

### 1. Data Preprocessing & Feature Engineering
* **Correlation Analysis:** Identified `hours_studied` (0.81) and `prev_exam_score` (0.80) as the strongest predictors of final exam performance. Dropped unhelpful noise variables like `lucky_number` and `student_id`.
* **Outlier Mitigation:** Diagnosed an extreme outlier (`hours_studied` z-score > 5) that was causing violent mathematical instability (Runge's Phenomenon) in higher-degree models. 
* **Normalization:** Applied strict Z-score normalization fitted *only* on the training data to prevent data leakage, ensuring stable and rapid gradient descent without overflow errors.

### 2. Linear Regression (Predicting Exam Scores)
* **Convergence:** The manual Gradient Descent implementation successfully minimized Mean Squared Error (MSE), with the loss curve smoothly converging.
* **Verification:** The manual parameters (weights and bias) and final MSE almost perfectly matched the parameters learned by the PyTorch model, validating the scratch-built calculus.

### 3. Polynomial Regression (The Bias-Variance Tradeoff)
* **The "Sweet Spot" (Degrees 2 & 3):** Achieved the lowest Test MSE, proving that a slightly curved model generalized best to unseen data by capturing the true trend without memorizing noise.
* **Extreme Overfitting (Degree 10):** Demonstrated textbook overfitting. While the model forced itself to memorize the training data, its Test MSE exploded to over 500,000. Visualizations proved the prediction curve was swinging wildly to connect every random noise point, completely failing on new data.

### 4. Logistic Regression (Predicting Pass/Fail)
* **Classification Success:** Successfully predicted binary outcomes using the Sigmoid activation function and Cross-Entropy loss, achieving **~86% Test Accuracy**.
* **Decision Boundary:** Mapped a 2D contour boundary demonstrating how the model separates passing and failing students based on combinations of study hours and previous exam scores.
* **Threshold Experiments:** Explored how adjusting the standard 0.5 decision threshold to 0.3 (optimistic/lenient) or 0.7 (strict) impacts the accuracy and changes the balance of false positives vs. false negatives.

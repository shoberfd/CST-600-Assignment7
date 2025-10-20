# CST-600-Assignment7

# Ensemble Learning for Breast Cancer Diagnosis

This project explores the application of ensemble learning techniques—specifically Bagging and Boosting—to improve the performance and stability of a breast cancer diagnosis model. It compares a single Decision Tree baseline against a Bagging Classifier and a Gradient Boosting Classifier.

## Healthcare Scenario
As a data scientist at MedTech Innovations, my task was to address the issue of high variance and instability in existing diagnostic models. By applying ensemble methods, the goal is to produce more reliable and accurate predictions that oncologists can trust.

---
## Dataset
* **Source**: `sklearn.datasets.load_breast_cancer()`
* **Description**: This is a classic, clean, built-in scikit-learn dataset. It contains 569 samples (split into 455 for training and 114 for testing) with 30 numeric features describing characteristics of cell nuclei from a breast mass.
* **Target**: Binary (0 = Malignant, 1 = Benign). The classes are well-balanced.

---
## Environment Setup
Follow these steps to set up the local environment on a Windows machine.

1.  **Clone the Repository**
2.  **Create and Activate the Virtual Environment**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
---
## How to Run
After setting up the environment:

1.  Navigate to the project's **root directory** in your terminal.
2.  Run the main script **as a module**:
    ```bash
    python -m src.main
    ```
    This command will execute the entire pipeline, printing the training stability and final test set evaluation for the Decision Tree, Bagging, and Gradient Boosting models sequentially.

---
## Summary of Findings
* **Baseline Model**: A single `DecisionTreeClassifier` achieved a test set accuracy of **91.2%**. Its cross-validation scores showed a standard deviation of ~0.019, indicating some instability.
* **Bagging**: The `BaggingClassifier` improved upon the baseline, reaching a test set accuracy of **93.9%**. This method successfully reduced variance and improved overall performance.
* **Boosting**: The `GradientBoostingClassifier` achieved the highest performance with a test set accuracy of **95.6%**. It was also the most stable model, showing the lowest standard deviation (~0.016) in cross-validation.
* **Conclusion**: Both Bagging and Boosting successfully demonstrated their value over a single model. For this task, **Gradient Boosting** provided the best combination of high accuracy and reliability, making it the most suitable candidate for a clinical diagnostic tool.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_on_test_set(name, model, X_test, y_test):
    """
    Evaluates a final trained model on the hold-out test set.
    """
    print(f"\n--- Final Evaluation for {name} on Test Set ---")
    y_pred = model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Malignant (0)', 'Benign (1)']))
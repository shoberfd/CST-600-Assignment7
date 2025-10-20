import os
from sklearn.model_selection import train_test_split

# Use relative imports for package structure
from .data_loader import load_and_preprocess_data
from .model_trainer import get_models, train_and_evaluate_model
from .evaluation import evaluate_on_test_set

def main_pipeline():
    """
    Runs the full model comparison pipeline.
    """
    # 1. Load Data and Preprocessor
    X, y, preprocessor = load_and_preprocess_data()

    # 2. Split Data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nData split into {len(X_train)} training and {len(X_test)} testing samples.")

    # 3. Get model definitions
    models = get_models()

    # 4. Loop through models, train, and evaluate
    for name, model in models.items():
        # Train the model and get stability scores
        trained_model = train_and_evaluate_model(name, model, preprocessor, X_train, y_train)
        
        # Evaluate on the hold-out test set
        evaluate_on_test_set(name, trained_model, X_test, y_test)
        print("-" * 60)
    
    print("\nEnsemble Learning Pipeline finished successfully! 🩺")

if __name__ == "__main__":
    main_pipeline()
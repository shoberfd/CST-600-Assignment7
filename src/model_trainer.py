import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

def train_and_evaluate_model(name, model, preprocessor, X_train, y_train):
    """
    Trains a given model, performs cross-validation, and returns the trained model.
    """
    print(f"\n--- Training and Validating {name} ---")
    
    # Create the full pipeline by combining the preprocessor and the model
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # --- Stability Check with Cross-Validation ---
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Use F1-score for validation as it's a good balanced metric
    cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=kfold, scoring='f1_macro')
    
    print(f"Stability (5-Fold CV F1-Score): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    
    # --- Train the final model on the entire training set ---
    full_pipeline.fit(X_train, y_train)
    
    print(f"Final {name} model trained.")
    return full_pipeline

def get_models():
    """
    Returns a dictionary of models to be trained and evaluated.
    """
    # Base model: A single decision tree with some depth to allow for high variance
    base_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    
    # Bagging model: An ensemble of 50 decision trees
    bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=10, random_state=42),
    n_estimators=50,
    random_state=42,
    oob_score=True,
    n_jobs=-1
)
    
    # Boosting model: Gradient Boosting
    boosting_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3, # Shallow trees to act as weak learners
        random_state=42
    )
    
    models = {
        "Decision Tree (Base)": base_tree,
        "Bagging": bagging_model,
        "Gradient Boosting": boosting_model
    }
    
    return models
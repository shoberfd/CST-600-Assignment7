import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_and_preprocess_data():
    """
    Loads the scikit-learn breast cancer dataset and creates a preprocessing pipeline.
    """
    print("Loading and preparing data...")
    # Load the dataset object
    data = load_breast_cancer()
    
    # Create DataFrame for features and Series for the target
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    
    # Create a simple preprocessing pipeline that only scales the data.
    # While not strictly necessary for trees, it's good practice and ensures
    # any future non-tree models would work correctly.
    preprocessor = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    print(f"Data prepared with {len(X)} samples.")
    return X, y, preprocessor
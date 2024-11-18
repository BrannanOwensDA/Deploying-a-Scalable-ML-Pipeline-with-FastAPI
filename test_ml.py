# TODO: add necessary import
import pytest
from sklearn.ensemble import RandomForestClassifier # type: ignore
from ml.model import train_model, compute_model_metrics, inference
import numpy as np



# TODO: implement the first test. Change the function name and input as needed
def test_train_model():
    """
    Test if train_model returns a RandomForestClassifier instance.
    """
    # Your code here
    # Dummy Data
    X_train = np.random.rand(10, 5)  
    y_train = np.random.randint(0, 2, 10) 
    
    # Use the dummy data with the model trainer
    model = train_model(X_train, y_train)
    
    # Assert the product is a RandomForestClassifier
    assert isinstance(model, RandomForestClassifier), "This is not a RandomForestClassifier"


# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Test if compute_model_metrics returns correct precision, recall, and F1-score.
    """
    # Your code here
    # Dummy Data
    y_true = np.array([1, 0, 1, 0, 1, 1])  # True labels
    y_pred = np.array([1, 1, 1, 0, 0, 1])  # Predicted labels
    
    # Compute the metrics using the function
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    # Expected values 
    expected_precision = 0.75  
    expected_recall = 0.75     
    expected_f1 = 0.75         
    
    # Step 4: Assertions to verify correct values
    assert precision == expected_precision, f"Expected precision {expected_precision}, got {precision}"
    assert recall == expected_recall, f"Expected recall {expected_recall}, got {recall}"
    assert fbeta == expected_f1, f"Expected F1-score {expected_f1}, got {fbeta}"


# TODO: implement the third test. Change the function name and input as needed
def test_inference():
    """
    Test if inference returns predictions of the correct shape and type.
    """
    # Your code here
    # Dummy Data
    X_train = np.random.rand(10, 5) 
    y_train = np.random.randint(0, 2, 10)  
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Mock test data for inference
    X_test = np.random.rand(5, 5)  # 5 samples, 5 features
    
    # Run inference
    preds = inference(model, X_test)
    
    # Assertions to validate the output
    assert preds.shape == (5,), f"Expected predictions of shape (5,), got {preds.shape}"
    assert np.issubdtype(preds.dtype, np.integer), "Expected integer predictions"

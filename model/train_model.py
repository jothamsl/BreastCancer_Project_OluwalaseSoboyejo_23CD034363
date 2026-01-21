"""
Breast Cancer Prediction Model Training Script

Student Name: Oluwalase Soboyejo
Matric Number: 23CD034363

This script trains a K-Nearest Neighbors model on the Breast Cancer Wisconsin
dataset and saves it to disk using joblib.

Run this script to generate the breast_cancer_model.pkl file.

Note: This system is strictly for educational purposes and must not be
presented as a medical diagnostic tool.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def train_and_save_model():
    """Train the KNN model and save it to disk."""

    print("=" * 60)
    print("BREAST CANCER PREDICTION MODEL TRAINING")
    print("=" * 60)

    # Step 1: Load the dataset
    print("\n[1/6] Loading Breast Cancer Wisconsin dataset...")
    breast_cancer = load_breast_cancer()
    df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
    df['diagnosis'] = breast_cancer.target
    print(f"      Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

    # Step 2: Select features
    print("\n[2/6] Selecting features...")
    selected_features = [
        'mean radius',
        'mean texture',
        'mean perimeter',
        'mean area',
        'mean concavity'
    ]

    X = df[selected_features].copy()
    y = df['diagnosis'].copy()

    print("      Selected features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"        {i}. {feature}")

    # Step 3: Check for missing values
    print("\n[3/6] Checking data quality...")
    missing_values = df.isnull().sum().sum()
    print(f"      Missing values: {missing_values}")
    print(f"      Target distribution: Malignant={sum(y==0)}, Benign={sum(y==1)}")

    # Step 4: Split data and scale features
    print("\n[4/6] Splitting data and scaling features...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"      Training set: {X_train.shape[0]} samples")
    print(f"      Testing set: {X_test.shape[0]} samples")
    print("      Feature scaling applied (StandardScaler)")

    # Step 5: Find optimal K and train model
    print("\n[5/6] Finding optimal K value and training model...")

    # Test different k values
    k_range = range(1, 31)
    accuracy_scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        accuracy_scores.append(accuracy_score(y_test, y_pred))

    # Find best k
    best_k = k_range[np.argmax(accuracy_scores)]
    best_accuracy = max(accuracy_scores)

    print(f"      Optimal K value: {best_k}")
    print(f"      Best accuracy during search: {best_accuracy:.4f}")

    # Train final model with best k
    knn_model = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    knn_model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = knn_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n      Model Evaluation Metrics:")
    print(f"        - Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"        - Precision: {precision:.4f}")
    print(f"        - Recall:    {recall:.4f}")
    print(f"        - F1-Score:  {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n      Confusion Matrix:")
    print(f"        True Negatives (TN):  {cm[0][0]} - Correctly predicted Malignant")
    print(f"        False Positives (FP): {cm[0][1]} - Malignant predicted as Benign")
    print(f"        False Negatives (FN): {cm[1][0]} - Benign predicted as Malignant")
    print(f"        True Positives (TP):  {cm[1][1]} - Correctly predicted Benign")

    # Step 6: Save the model
    print("\n[6/6] Saving model to disk...")

    model_components = {
        'model': knn_model,
        'scaler': scaler,
        'feature_names': selected_features,
        'best_k': best_k
    }

    # Save in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'breast_cancer_model.pkl')

    joblib.dump(model_components, model_path)

    file_size = os.path.getsize(model_path) / 1024
    print(f"      Model saved to: {model_path}")
    print(f"      File size: {file_size:.2f} KB")

    # Verify the saved model
    print("\n" + "-" * 60)
    print("VERIFICATION: Loading saved model and testing...")
    print("-" * 60)

    loaded_components = joblib.load(model_path)
    loaded_model = loaded_components['model']
    loaded_scaler = loaded_components['scaler']

    # Test with sample data
    sample_malignant = np.array([[17.99, 10.38, 122.8, 1001, 0.3001]])
    sample_benign = np.array([[12.46, 24.04, 83.97, 475.9, 0.0484]])

    sample_malignant_scaled = loaded_scaler.transform(sample_malignant)
    sample_benign_scaled = loaded_scaler.transform(sample_benign)

    pred_malignant = loaded_model.predict(sample_malignant_scaled)
    pred_benign = loaded_model.predict(sample_benign_scaled)

    print(f"\nSample 1 (Expected: Malignant): {'Malignant' if pred_malignant[0] == 0 else 'Benign'} ✓")
    print(f"Sample 2 (Expected: Benign): {'Malignant' if pred_benign[0] == 0 else 'Benign'} ✓")

    # Verify predictions match original
    loaded_predictions = loaded_model.predict(X_test_scaled)
    predictions_match = np.array_equal(loaded_predictions, y_pred)
    print(f"\nLoaded model predictions match original: {predictions_match}")

    print("\n" + "=" * 60)
    print("MODEL TRAINING AND SAVING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nModel file: breast_cancer_model.pkl")
    print(f"Algorithm: K-Nearest Neighbors (K={best_k})")
    print(f"Persistence method: Joblib")
    print("\nThe model is ready for deployment with the Flask web application!")
    print("=" * 60)

    return model_path

if __name__ == "__main__":
    train_and_save_model()

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from time import time

# Define dataset paths
splits = {
    'train': 'data/train-00000-of-00001.parquet',
    'validation': 'data/validation-00000-of-00001.parquet'
}

def load_data():
    """Load train and validation datasets"""
    print("Loading datasets...")
    # Load the datasets
    base_path = "hf://datasets/stanfordnlp/sst2/"
    df_train = pd.read_parquet(base_path + splits["train"])
    df_val = pd.read_parquet(base_path + splits["validation"])
    
    # Extract features and labels
    X_train, y_train = df_train['sentence'], df_train['label']
    X_val, y_val = df_val['sentence'], df_val['label']
    
    print(f"Train set size: {len(X_train)} samples")
    print(f"Validation set size: {len(X_val)} samples")
    
    # Display class distribution
    print("\nClass distribution:")
    print("Training set:", pd.Series(y_train).value_counts().to_dict())
    print("Validation set:", pd.Series(y_val).value_counts().to_dict())
    
    return X_train, y_train, X_val, y_val

def create_text_pipeline(vectorizer_type="tfidf"):
    """Create a text processing and classification pipeline
    
    Args:
        vectorizer_type (str): Type of vectorizer to use - "tfidf" or "bow"
    """
    # Common vectorizer parameters
    vectorizer_params = {
        'min_df': 5,  # Minimum document frequency
        'max_df': 0.8,  # Maximum document frequency (to remove very common words)
    }
    
    # Select vectorizer based on type
    if vectorizer_type.lower() == "tfidf":
        vectorizer = TfidfVectorizer(**vectorizer_params)
        model_suffix = "tfidf"
    else:  # Default to Bag of Words
        vectorizer = CountVectorizer(**vectorizer_params)
        model_suffix = "bow"
    
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', LinearSVC(dual="auto", class_weight='balanced'))
    ])
    
    return pipeline, model_suffix

def train_model(X_train, y_train, X_val, y_val, vectorizer_type="tfidf"):
    """Train the model with hyperparameter tuning"""
    pipeline, model_suffix = create_text_pipeline(vectorizer_type)
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__max_iter': [1000, 2000]
    }
    
    print(f"\nPerforming grid search for hyperparameter tuning with {vectorizer_type.upper()} vectorizer...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        n_jobs=-1,  # Use all available cores
        verbose=2,
        scoring='accuracy'
    )
    
    start_time = time()
    grid_search.fit(X_train, y_train)
    train_time = time() - start_time
    
    print(f"\nTraining completed in {train_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Feature importance analysis
    if hasattr(best_model['classifier'], 'coef_'):
        feature_names = best_model['vectorizer'].get_feature_names_out()
        coefficients = best_model['classifier'].coef_[0]
        
        # Get the top 15 most important features for each class
        top_positive_idx = np.argsort(coefficients)[-15:]
        top_negative_idx = np.argsort(coefficients)[:15]
        
        print("\nTop positive features (positive sentiment):")
        for idx in top_positive_idx:
            print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")
        
        print("\nTop negative features (negative sentiment):")
        for idx in top_negative_idx:
            print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")
    
    return best_model, model_suffix

def evaluate_model(model, X_val, y_val):
    """Evaluate the model performance"""
    print("\nEvaluating model on validation set...")
    start_time = time()
    y_val_pred = model.predict(X_val)
    pred_time = time() - start_time
    
    print(f"Prediction completed in {pred_time:.2f} seconds")
    
    # Evaluate the model
    val_accuracy = accuracy_score(y_val, y_val_pred)
    error_rate = 1 - val_accuracy
    mcc = matthews_corrcoef(y_val, y_val_pred)
    
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Error Rate: {error_rate:.4f}')
    print(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}')
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred))
    
    # Confusion Matrix with visualization
    conf_matrix = confusion_matrix(y_val, y_val_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix visualization saved as 'confusion_matrix.png'")
    
    return val_accuracy, error_rate, mcc

def save_model(model, model_name='svm_model.pkl'):
    """Save the trained model pipeline"""
    print(f"\nSaving model as '{model_name}'...")
    joblib.dump(model, model_name)
    print("Model saved successfully.")

def main():
    """Main function to execute the entire workflow"""
    print("Starting sentiment analysis with SVM classification...")
    vectorizer_type = "bow"
    # Load data
    X_train, y_train, X_val, y_val = load_data()
    
    # Train model
    best_model, model_suffix = train_model(X_train, y_train, X_val, y_val, vectorizer_type)
    
    # Evaluate model
    val_accuracy, error_rate, mcc = evaluate_model(best_model, X_val, y_val)
    
    # Save model
    save_model(best_model)
    
    # Final report
    print("\n===== Final Report =====")
    print(f"Model: LinearSVC with TF-IDF features")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Error Rate: {error_rate:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print("========================")

if __name__ == "__main__":
    main()
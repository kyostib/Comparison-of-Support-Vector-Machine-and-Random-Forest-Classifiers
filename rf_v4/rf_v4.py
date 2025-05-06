import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier # Changed import
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, matthews_corrcoef, make_scorer)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from collections import Counter
import re
from time import time
import json

# Define dataset paths
splits = {
    'train': 'data/train-00000-of-00001.parquet',
    'validation': 'data/validation-00000-of-00001.parquet'
}

def tokenize(text):
    """Basic tokenizer that lowercases and extracts words"""
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


# Custom stop-word list from the most frequent words. 
# This is done because sklearns 'english' stop-words list is quite bad.
def build_custom_stopwords(corpus, top_n=50):
    word_counts = Counter()
    for doc in corpus:
        tokens = tokenize(doc)
        word_counts.update(tokens)
    return set([word for word, _ in word_counts.most_common(top_n)])


# Loading data and splitting it into features and labels, and training and validation
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
# Create a pipeline function - now generally applicable
def create_pipeline(vectorizer_instance, classifier_instance):
    """Creates a text processing and classification pipeline"""
    return Pipeline([
        ('vectorizer', vectorizer_instance),
        ('classifier', classifier_instance)
    ])

def perform_grid_search(X_train, y_train, all_custom_stopwords_options=None):
    """
    Performs GridSearchCV to find the best hyperparameters for RandomForest.
    
    Args:
        X_train: Training features (text data)
        y_train: Training labels
        all_custom_stopwords_options: A list of pre-calculated custom stopword lists.
                                     Each list in this list will be an option for GridSearchCV.

    Returns:
        tuple: (best_estimator, best_params, cv_results)
    """
    print("\nPerforming GridSearchCV for RandomForestClassifier...")
    
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()), # Placeholder, will be set by param_grid
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
    ])

    # Prepare the stop_words options for the grid
    stop_words_grid_options = [None, 'english']
    if all_custom_stopwords_options:
        stop_words_grid_options.extend(all_custom_stopwords_options)
        print(f"Included {len(all_custom_stopwords_options)} custom stopword lists in grid search.")


    param_grid = [
        # Configuration for TfidfVectorizer
        {
            'vectorizer': [TfidfVectorizer(tokenizer=tokenize)],
            'vectorizer__min_df': [5], # Example values, adjust as needed
            'vectorizer__max_df': [0.5], # Example values
            'vectorizer__stop_words': stop_words_grid_options, # Iterate over different stopword sets
            'classifier': [RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)],
            'classifier__n_estimators': [100, 150], # Reduced for quicker test, expand as needed
            'classifier__max_depth': [None, 20],    # Reduced for quicker test
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 3]
        },
        # Configuration for CountVectorizer
        {
            'vectorizer': [CountVectorizer(tokenizer=tokenize)],
            'vectorizer__stop_words': stop_words_grid_options, # Iterate over different stopword sets
            'classifier': [RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)],
            'classifier__n_estimators': [100, 150], # Reduced
            'classifier__max_depth': [None, 20],    # Reduced
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 3]
        }
    ]
    
    mcc_scorer = make_scorer(matthews_corrcoef)

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, # cv=3 for faster run, recommend 5 for real use
                               scoring={'accuracy': 'accuracy', 'mcc': mcc_scorer}, 
                               refit='accuracy',
                               n_jobs=-1, verbose=2, return_train_score=True) 
    
    start_time = time()
    print("Fitting GridSearchCV...")
    grid_search.fit(X_train, y_train)
    fit_time = time() - start_time
    print(f"GridSearchCV fitting completed in {fit_time:.2f} seconds")

    print(f"\nBest parameters found by GridSearchCV (based on accuracy):")
    # Custom print for best_params to handle potentially long stopword lists
    best_params_to_print = {}
    for k, v in grid_search.best_params_.items():
        if k == 'vectorizer__stop_words' and isinstance(v, list):
            best_params_to_print[k] = f"custom list ({len(v)} words)" if len(v) > 0 else None
        elif k == 'vectorizer__stop_words' and v == 'english':
            best_params_to_print[k] = 'english'
        elif k == 'vectorizer__stop_words' and v is None:
             best_params_to_print[k] = None
        else:
            best_params_to_print[k] = v
    print(best_params_to_print)
    print(f"\nBest cross-validation accuracy score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


def visualize_cv_results(cv_results, top_n=20):
    """Visualize cross-validation results"""
    print("\nProcessing CV results for visualization...")
    results_df = pd.DataFrame(cv_results)
    
    # Function to make param values more readable, especially for stopword lists
    def format_params(param_dict):
        formatted_params = {}
        for k, v in param_dict.items():
            if k == 'vectorizer__stop_words':
                if isinstance(v, list):
                    formatted_params[k] = f'custom_{len(v)}'
                else: # None or 'english'
                    formatted_params[k] = str(v)
            elif isinstance(v, (TfidfVectorizer, CountVectorizer, RandomForestClassifier)):
                 formatted_params[k] = type(v).__name__ # Just class name
            else:
                formatted_params[k] = v
        return str(formatted_params)

    results_df['params_str'] = results_df['params'].apply(format_params)

    # Corrected column selection
    key_columns = ['params_str', 'mean_test_accuracy', 'std_test_accuracy', 
                   'mean_train_accuracy', 'std_train_accuracy', 
                   'mean_test_mcc', 'std_test_mcc',
                   'mean_fit_time', 'mean_score_time']
    # Filter out columns that might not exist if return_train_score=False for some reason
    existing_columns = [col for col in key_columns if col in results_df.columns]
    results_df_filtered = results_df[existing_columns]
    
    results_df_sorted = results_df_filtered.sort_values(by='mean_test_accuracy', ascending=False)

    print(f"\nTop {top_n} configurations based on Mean CV Accuracy:")
    print(results_df_sorted.head(top_n))

    results_df_sorted.to_csv('gridsearchcv_rf_results.csv', index=False)
    print(f"\nFull GridSearchCV results saved to 'gridsearchcv_rf_results.csv'")

    plt.figure(figsize=(15, 8)) # Increased height for readability
    top_results = results_df_sorted.head(top_n)
    # Use params_str for ticks to give more info
    tick_labels = [p[:100] + '...' if len(p) > 100 else p for p in top_results['params_str']] 

    plt.errorbar(range(top_n), top_results['mean_test_accuracy'], 
                 yerr=top_results['std_test_accuracy'] if 'std_test_accuracy' in top_results else None, 
                 fmt='o-', label='Mean Test Accuracy')
    plt.xticks(range(top_n), tick_labels, rotation=90, ha='right')
    plt.xlabel('Top Configurations (Sorted by Accuracy)')
    plt.ylabel('Accuracy')
    plt.title(f'Top {top_n} RandomForest Configurations from GridSearchCV (CV Accuracy)')
    plt.legend()
    plt.tight_layout() # Adjust layout to make room for labels
    plt.savefig('gridsearchcv_rf_top_configs_accuracy.png')
    print(f"Plot of top {top_n} config accuracies saved to 'gridsearchcv_rf_top_configs_accuracy.png'")
    plt.close()

    return results_df_sorted


def evaluate_final_model(best_model_pipeline, X_val, y_val):
    """Evaluate the best model pipeline on the validation set."""
    print("\nEvaluating the best model on the validation set...")
    start_time = time()
    y_val_pred = best_model_pipeline.predict(X_val)
    pred_time = time() - start_time
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    error_rate = 1 - val_accuracy
    mcc = matthews_corrcoef(y_val, y_val_pred)
    class_report_str = classification_report(y_val, y_val_pred) # string report
    class_report_dict = classification_report(y_val, y_val_pred, output_dict=True) # dict report
    conf_matrix = confusion_matrix(y_val, y_val_pred)

    print(f"\nValidation Set Performance:")
    print(f"Prediction Time: {pred_time:.2f} seconds")
    print(f'Accuracy: {val_accuracy:.4f}')
    print(f'Error Rate: {error_rate:.4f}')
    print(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}')
    print("\nClassification Report:")
    print(class_report_str)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=best_model_pipeline.classes_,
                yticklabels=best_model_pipeline.classes_)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix for Best RandomForest Model (Validation Set)')
    plt.tight_layout()
    plt.savefig('best_rf_model_confusion_matrix.png')
    print("Confusion matrix plot saved to 'best_rf_model_confusion_matrix.png'")
    plt.close()

    try:
        vectorizer = best_model_pipeline.named_steps['vectorizer']
        classifier = best_model_pipeline.named_steps['classifier']
        
        if hasattr(classifier, 'feature_importances_'):
            feature_names = vectorizer.get_feature_names_out()
            importances = classifier.feature_importances_
            
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

            print("\nTop 20 most important features:")
            print(feature_importance_df.head(20))
            
            feature_importance_df.to_csv('best_rf_model_feature_importances.csv', index=False)
            print("Feature importances saved to 'best_rf_model_feature_importances.csv'")
        else:
            print("\nThe classifier does not support feature_importances_.")
    except Exception as e:
        print(f"\nCould not perform feature importance analysis: {e}")

    final_metrics = {
        'accuracy': val_accuracy,
        'error_rate': error_rate,
        'mcc': mcc,
        'report': class_report_dict,
        'confusion_matrix': conf_matrix.tolist(),
        'pred_time': pred_time
    }
    return final_metrics


def save_model(model, model_name='best_rf_model.pkl'):
   print(f"\nSaving best model pipeline as '{model_name}'...")
   joblib.dump(model, model_name)
   print("Model saved successfully.")


def save_final_report(metrics, filename='final_rf_validation_report.json'):
    print(f"Saving final validation report to {filename}...")
    try:
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=4)
        print("Final report saved successfully.")
    except Exception as e:
        print(f"Error saving final report: {e}")


def main():
    print("Starting Random Forest Classification with GridSearchCV...")
    total_start_time = time()

    X_train, y_train, X_val, y_val = load_data()

    # Define different top_n values for custom stopwords
    stopwords_counts_to_try = [10, 20, 30, 50] # Adjust as needed
    all_custom_stopwords_options = []
    print("\nGenerating custom stopword lists for GridSearch...")
    for count in stopwords_counts_to_try:
        custom_list = build_custom_stopwords(X_train, top_n=count)
        all_custom_stopwords_options.append(custom_list) # Add the list itself

    best_model, best_params, cv_results = perform_grid_search(
        X_train, y_train, all_custom_stopwords_options=all_custom_stopwords_options
    )

    df_cv_results = visualize_cv_results(cv_results)
    final_metrics = evaluate_final_model(best_model, X_val, y_val)
    save_model(best_model, model_name='best_random_forest_pipeline.pkl')
    save_final_report(final_metrics, filename='final_random_forest_validation_report.json')

    total_end_time = time()
    print("\n===== Final Summary =====")
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds")
    print("\nBest Hyperparameters found via GridSearchCV:")
    
    # More detailed print for best_params
    for param_name, param_value in best_params.items():
        if param_name == 'vectorizer__stop_words':
            if isinstance(param_value, list):
                print(f"  {param_name}: custom list ({len(param_value)} words)")
            elif param_value is None:
                 print(f"  {param_name}: None")
            else: # 'english'
                print(f"  {param_name}: '{param_value}'")
        elif isinstance(param_value, (TfidfVectorizer, CountVectorizer, RandomForestClassifier)):
            print(f"  {param_name}: {type(param_value).__name__}(...)")
        else:
            print(f"  {param_name}: {param_value}")
             
    if not df_cv_results.empty:          
        print(f"\nBest Cross-Validation Accuracy (mean across folds): {df_cv_results['mean_test_accuracy'].iloc[0]:.4f}")
        if 'mean_test_mcc' in df_cv_results.columns:
             print(f"Corresponding Cross-Validation MCC (mean across folds): {df_cv_results['mean_test_mcc'].iloc[0]:.4f}")
    else:
        print("\nCV results DataFrame is empty, cannot report CV scores.")


    print("\nFinal Performance on Independent Validation Set:")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  MCC: {final_metrics['mcc']:.4f}")
    print(f"  Error Rate: {final_metrics['error_rate']:.4f}")

    print("\nArtifacts saved:")
    print("  - gridsearchcv_rf_results.csv (Full CV results)")
    print("  - gridsearchcv_rf_top_configs_accuracy.png (Plot of top CV accuracies)")
    print("  - best_rf_model_confusion_matrix.png (Validation set confusion matrix)")
    print("  - best_rf_model_feature_importances.csv (Feature importances from best model)")
    print("  - best_random_forest_pipeline.pkl (Saved best model pipeline)")
    print("  - final_random_forest_validation_report.json (Metrics on validation set)")
    print("========================")


if __name__ == "__main__":
    main()
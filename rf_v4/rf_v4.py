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

def format_grid_params_for_display(param_dict_from_gs):
    """
    Formats a parameter dictionary from GridSearchCV's cv_results_['params']
    into a more readable string representation.
    Handles specific formatting for stop_words and estimator class names.
    """
    formatted_params = {}
    for k, v in param_dict_from_gs.items():
        # For top-level 'vectorizer' or 'classifier' keys, which hold the estimator objects
        if k in ['vectorizer', 'classifier'] and hasattr(v, '__class__') and \
           not isinstance(v, (int, float, str, bool, list, dict, tuple, set)): # Check it's an actual estimator object
            formatted_params[k] = type(v).__name__
        # Specific handling for 'vectorizer__stop_words' sub-parameter
        elif k == 'vectorizer__stop_words':
            if isinstance(v, list) or isinstance(v, set):
                formatted_params[k] = f'custom_{len(v) if v else 0}' # Handles empty list/set
            elif v is None:
                formatted_params[k] = 'None'
            else: # 'english' or other string
                formatted_params[k] = str(v)
        else: # For all other parameters (e.g., classifier__n_estimators, vectorizer__min_df)
            formatted_params[k] = v
    # Create a consistent string representation by sorting main keys
    return str(dict(sorted(formatted_params.items())))

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

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, # cv=3 for faster run, recommend 5 for real use
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


def visualize_cv_results(cv_results_dict, top_n_display=20, top_n_save_summary=10, sort_metric='mean_test_accuracy'):
    """
    Processes and visualizes cross-validation results from GridSearchCV.
    Saves the full CV results and a summary of the top N configurations to CSV files.
    Plots the performance of the top N configurations.

    Args:
        cv_results_dict (dict): The cv_results_ attribute from a fitted GridSearchCV object.
        top_n_display (int): Number of top configurations to print to console and include in the plot.
        top_n_save_summary (int): Number of top configurations to save in a separate summary CSV file.
        sort_metric (str): The metric in cv_results_dict to use for sorting (e.g., 'mean_test_accuracy').
    
    Returns:
        pandas.DataFrame: The full, sorted DataFrame of CV results, or an empty DataFrame on error.
    """
    print("\n--- Processing GridSearchCV Cross-Validation Results ---")
    if not isinstance(cv_results_dict, dict) or not cv_results_dict:
        print("Error: cv_results_dict is empty or not a dictionary. Cannot process.")
        return pd.DataFrame()

    try:
        results_df = pd.DataFrame(cv_results_dict)
    except Exception as e:
        print(f"Error converting cv_results_dict to DataFrame: {e}")
        return pd.DataFrame()

    if results_df.empty:
        print("CV results DataFrame is empty after conversion.")
        return results_df

    if 'params' not in results_df.columns:
        print("Error: 'params' column missing in cv_results_dict. Cannot format parameters.")
        return results_df # Or handle more gracefully depending on needs

    # Format parameters for readability
    results_df['params_str'] = results_df['params'].apply(format_grid_params_for_display)

    # Define key columns that are expected from GridSearchCV with multi-metric scoring
    # (and train scores if return_train_score=True)
    expected_base_metrics = ['accuracy', 'mcc'] # Metrics used in scoring
    score_types = ['mean_test', 'std_test', 'mean_train', 'std_train', 'rank_test']
    
    key_columns_for_report = ['params_str', 'params', 'mean_fit_time', 'mean_score_time']
    for metric_name in expected_base_metrics:
        for score_type in score_types:
            key_columns_for_report.append(f'{score_type}_{metric_name}')
            
    # Filter to keep only existing columns in the DataFrame
    existing_columns_for_report = [col for col in key_columns_for_report if col in results_df.columns]
    if not existing_columns_for_report:
        print("Error: No relevant data columns found in CV results DataFrame.")
        return results_df
        
    results_df_filtered = results_df[existing_columns_for_report]

    # Sort by the specified metric
    if sort_metric not in results_df_filtered.columns:
        print(f"Warning: Specified sort_metric '{sort_metric}' not found in CV results. Results will not be sorted reliably.")
        results_df_sorted = results_df_filtered # Proceed with unsorted or default sorted data
    else:
        results_df_sorted = results_df_filtered.sort_values(by=sort_metric, ascending=False)

    # --- 1. Print top N configurations to console ---
    print(f"\nTop {top_n_display} configurations (based on {sort_metric}):")
    display_cols_console = ['params_str', sort_metric]
    # Add MCC to console output if available and different from sort_metric
    if 'mean_test_mcc' in results_df_sorted.columns and sort_metric != 'mean_test_mcc':
        display_cols_console.append('mean_test_mcc')
    if 'mean_fit_time' in results_df_sorted.columns:
        display_cols_console.append('mean_fit_time')
    
    existing_display_cols_console = [col for col in display_cols_console if col in results_df_sorted.columns]
    print(results_df_sorted[existing_display_cols_console].head(top_n_display))

    # --- 2. Save the FULL GridSearchCV results ---
    full_results_filename = 'gridsearchcv_all_configurations_results.csv'
    try:
        results_df_sorted.to_csv(full_results_filename, index=False)
        print(f"\nFull GridSearchCV results saved to '{full_results_filename}'")
    except Exception as e:
        print(f"Error saving full CV results to CSV: {e}")

    # --- 3. Save a summary of the top N configurations (CV performance) ---
    if top_n_save_summary > 0:
        print(f"\nSaving summary of top {top_n_save_summary} configurations (CV performance) to CSV...")
        top_summary_df = results_df_sorted.head(top_n_save_summary)
        
        columns_for_top_summary = ['params_str', 'mean_test_accuracy', 'mean_test_mcc', 'mean_fit_time']
        # Ensure these columns exist, especially if sorting by a different metric
        existing_columns_for_top_summary = [col for col in columns_for_top_summary if col in top_summary_df.columns]
        
        if not existing_columns_for_top_summary or 'params_str' not in existing_columns_for_top_summary:
             print("Warning: Essential columns for top CV configurations summary CSV are missing. Skipping save.")
        else:
            top_summary_to_save = top_summary_df[existing_columns_for_top_summary]
            top_summary_filename = f'top_{top_n_save_summary}_cv_configurations_summary.csv'
            try:
                top_summary_to_save.to_csv(top_summary_filename, index=False)
                print(f"Top {top_n_save_summary} CV configurations summary saved to '{top_summary_filename}'")
            except Exception as e:
                print(f"Error saving top CV configurations summary to CSV: {e}")
    
    # --- 4. Plotting performance of top N configurations ---
    if top_n_display > 0 and sort_metric in results_df_sorted.columns:
        print(f"\nGenerating plot for top {top_n_display} configurations...")
        plt.figure(figsize=(15, max(10, top_n_display * 0.6))) # Adjusted height
        top_results_for_plot = results_df_sorted.head(top_n_display)
        
        # Use params_str for ticks, truncate if too long for display
        tick_labels = [
            (p[:100] + '...' if len(p) > 100 else p) 
            for p in top_results_for_plot['params_str']
        ]

        y_values = top_results_for_plot[sort_metric]
        y_err_values = None
        std_dev_col = sort_metric.replace('mean_', 'std_') # e.g., 'std_test_accuracy'
        if std_dev_col in top_results_for_plot:
            y_err_values = top_results_for_plot[std_dev_col]

        plt.errorbar(range(len(top_results_for_plot)), y_values,
                     yerr=y_err_values,
                     fmt='o-', label=f'{sort_metric} (CV)', capsize=5, elinewidth=1, markeredgewidth=1)
        
        # Optionally plot MCC on the same graph if it's different and available
        mcc_metric_plot = 'mean_test_mcc'
        if mcc_metric_plot in top_results_for_plot.columns and mcc_metric_plot != sort_metric:
            y_values_mcc = top_results_for_plot[mcc_metric_plot]
            y_err_mcc = None
            std_dev_mcc_col = mcc_metric_plot.replace('mean_', 'std_')
            if std_dev_mcc_col in top_results_for_plot:
                 y_err_mcc = top_results_for_plot[std_dev_mcc_col]
            plt.errorbar(range(len(top_results_for_plot)), y_values_mcc,
                         yerr=y_err_mcc,
                         fmt='s--', label=f'{mcc_metric_plot} (CV)', capsize=5, elinewidth=1, markeredgewidth=1, alpha=0.7)

        plt.xticks(range(len(top_results_for_plot)), tick_labels, rotation=90, ha='right', fontsize=8)
        plt.xlabel(f'Top {top_n_display} Configurations (Sorted by {sort_metric})')
        plt.ylabel('Score')
        plt.title(f'Top {top_n_display} Configurations from GridSearchCV (CV Performance)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        plot_filename = f'gridsearchcv_top_{top_n_display}_configs_performance.png'
        try:
            plt.savefig(plot_filename)
            print(f"Plot of top {top_n_display} configurations saved to '{plot_filename}'")
        except Exception as e:
            print(f"Error saving plot: {e}")
        plt.close()
    else:
        print("Plotting skipped: No data to plot or sort_metric column missing.")
        
    print("--- Finished processing CV results ---")
    return results_df_sorted

def evaluate_top_n_on_validation(X_train, y_train, X_val, y_val, cv_results_dict, 
                                 top_n=10, refit_metric_name='accuracy', 
                                 filename='top_n_configs_validation_performance.csv'):
    """
    Retrains the top N configurations from GridSearchCV on the full training set
    and evaluates them on the validation set.

    Args:
        X_train, y_train: Full training data.
        X_val, y_val: Validation data.
        cv_results_dict: The cv_results_ attribute from GridSearchCV.
        top_n: Number of top configurations to validate.
        refit_metric_name: The metric name used for 'refit' in GridSearchCV (e.g., 'accuracy').
                           This determines how "top" configurations are ranked.
        filename: Name of the CSV file to save validation results.
    Returns:
        pandas.DataFrame: DataFrame containing validation performance of top N models.
    """
    print(f"\n--- Validating Top {top_n} Configurations on Validation Set ---")
    validation_results_list = []
    
    df_cv = pd.DataFrame(cv_results_dict)

    # Determine sorting column based on refit_metric_name
    # GridSearchCV ranks are 'rank_test_METRIC_NAME' (lower is better)
    rank_col = f'rank_test_{refit_metric_name}'
    mean_score_col = f'mean_test_{refit_metric_name}'

    if rank_col in df_cv.columns:
        df_cv_sorted = df_cv.sort_values(by=rank_col, ascending=True)
    elif mean_score_col in df_cv.columns:
        print(f"Warning: Rank column '{rank_col}' not found. Sorting by '{mean_score_col}' descending.")
        df_cv_sorted = df_cv.sort_values(by=mean_score_col, ascending=False)
    else:
        print(f"Error: Neither rank ('{rank_col}') nor score ('{mean_score_col}') column for metric '{refit_metric_name}' found. Cannot select top configurations.")
        return pd.DataFrame()

    top_n_selected_configs = df_cv_sorted.head(top_n)

    for idx, row in top_n_selected_configs.iterrows():
        params_for_this_config = row['params'] # Original params dict
        cv_rank = row.get(rank_col, idx + 1) # Use rank if available, else fallback to index
        
        print(f"\nProcessing Configuration (CV Rank: {cv_rank})")
        # print(f"  Parameters: {format_grid_params_for_display(params_for_this_config)}") # Can be verbose

        # Reconstruct the pipeline using the estimator objects from this specific param set
        # These objects (params_for_this_config['vectorizer'], params_for_this_config['classifier'])
        # are already configured with their specific sub-parameters (e.g., min_df, n_estimators)
        # as per this particular grid point from GridSearchCV.
        pipeline_to_validate = Pipeline([
            ('vectorizer', params_for_this_config['vectorizer']),
            ('classifier', params_for_this_config['classifier'])
        ])

        # Train this specific pipeline configuration on the FULL training set
        print(f"  Training on full training set...")
        train_start_time = time()
        pipeline_to_validate.fit(X_train, y_train)
        train_time_val = time() - train_start_time

        # Predict on the validation set
        print(f"  Predicting on validation set...")
        pred_start_time = time()
        y_val_pred = pipeline_to_validate.predict(X_val)
        pred_time_val = time() - pred_start_time

        # Calculate validation metrics
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_mcc = matthews_corrcoef(y_val, y_val_pred)

        print(f"    CV Mean Accuracy: {row.get(f'mean_test_accuracy', float('nan')):.4f}, CV Mean MCC: {row.get(f'mean_test_mcc', float('nan')):.4f}")
        print(f"    Validation Accuracy: {val_accuracy:.4f}, Validation MCC: {val_mcc:.4f}")
        print(f"    Training Time: {train_time_val:.2f}s, Prediction Time: {pred_time_val:.2f}s")

        validation_results_list.append({
            'cv_rank': cv_rank,
            'params_str': format_grid_params_for_display(params_for_this_config),
            'cv_mean_accuracy': row.get(f'mean_test_accuracy', float('nan')),
            'cv_mean_mcc': row.get(f'mean_test_mcc', float('nan')),
            'validation_accuracy': val_accuracy,
            'validation_mcc': val_mcc,
            'training_time_full_train_sec': train_time_val,
            'prediction_time_val_sec': pred_time_val,
        })

    df_validation_summary = pd.DataFrame(validation_results_list)
    df_validation_summary.to_csv(filename, index=False)
    print(f"\nValidation performance of top {len(df_validation_summary)} configurations saved to '{filename}'")
    
    return df_validation_summary

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
    df_top_n_validation_perf = evaluate_top_n_on_validation(
        X_train, y_train, X_val, y_val, cv_results, 
        top_n=10,  # Or another number like 5 or 20
        refit_metric_name='accuracy' 
    )
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

    if not df_top_n_validation_perf.empty:
        print("\nTop Configurations Performance on Validation Set (sorted by CV rank):")
        # Select a few key columns to display
        cols_to_show = ['cv_rank', 'params_str', 'cv_mean_accuracy', 'validation_accuracy', 'validation_mcc']
        existing_cols_to_show = [col for col in cols_to_show if col in df_top_n_validation_perf.columns]
        print(df_top_n_validation_perf[existing_cols_to_show])
             
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
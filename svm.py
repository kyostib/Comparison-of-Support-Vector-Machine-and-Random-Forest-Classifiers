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
from collections import Counter
import re
from time import time

# Define dataset paths
splits = {
    'train': 'data/train-00000-of-00001.parquet',
    'validation': 'data/validation-00000-of-00001.parquet'
}

def tokenize(text):
    """Basic tokenizer that lowercases and extracts words"""
    return re.findall(r'\b[a-z]{2,}\b', text.lower())

def build_custom_stopwords(corpus, top_n=50):
    """Build a stopword list of the most frequent words in the training corpus"""
    word_counts = Counter()
    for doc in corpus:
        tokens = tokenize(doc)
        word_counts.update(tokens)
    return set([word for word, _ in word_counts.most_common(top_n)])

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

def create_text_pipeline(vectorizer_type="tfidf", stop_words=None):
    """Create a text processing and classification pipeline
    
    Args:
        vectorizer_type (str): Type of vectorizer to use - "tfidf" or "bow"
        stop_words: None, 'english', or custom list of stopwords
    """
    # Common vectorizer parameters
    vectorizer_params = {
        'min_df': 5,  # Minimum document frequency
        'max_df': 0.5,  # Maximum document frequency (to remove very common words)
        'stop_words': stop_words
    }
    vectorizer_params_bow = {
        'stop_words': stop_words
    }
    
    # Select vectorizer based on type
    if vectorizer_type.lower() == "tfidf":
        vectorizer = TfidfVectorizer(**vectorizer_params)
    else:  # Default to Bag of Words
        vectorizer = CountVectorizer(**vectorizer_params_bow)
    
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', LinearSVC(dual="auto", class_weight='balanced'))
    ])
    
    return pipeline

def evaluate_configuration(X_train, y_train, X_val, y_val, config):
    """Evaluate a specific configuration"""
    vectorizer_type = config['vectorizer_type']
    stopwords_option = config['stopwords_option']
    stopwords_count = config.get('stopwords_count', 0)
    c_value = config['c_value']
    
    # Prepare stopwords
    stop_words = None
    if stopwords_option == 'english':
        stop_words = 'english'
    elif stopwords_option == 'custom':
        stop_words = list(build_custom_stopwords(X_train, top_n=stopwords_count))
    
    # Create pipeline
    pipeline = create_text_pipeline(vectorizer_type, stop_words)
    
    # Set C value for LinearSVC
    pipeline.named_steps['classifier'].C = c_value
    pipeline.named_steps['classifier'].max_iter = 2000  # Higher iteration limit
    
    # Train the model
    print(f"\nTraining model with configuration: {config}")
    start_time = time()
    pipeline.fit(X_train, y_train)
    train_time = time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Evaluate on validation set
    start_time = time()
    y_val_pred = pipeline.predict(X_val)
    pred_time = time() - start_time
    
    # Calculate metrics
    val_accuracy = accuracy_score(y_val, y_val_pred)
    error_rate = 1 - val_accuracy
    mcc = matthews_corrcoef(y_val, y_val_pred)
    
    # Generate classification report
    report = classification_report(y_val, y_val_pred, output_dict=True)
    
    results = {
        'config': config,
        'accuracy': val_accuracy,
        'error_rate': error_rate,
        'mcc': mcc,
        'train_time': train_time,
        'pred_time': pred_time,
        'report': report,
        'pipeline': pipeline
    }
    
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Error Rate: {error_rate:.4f}')
    print(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}')
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred))
    
    return results

def grid_search_configurations(X_train, y_train, X_val, y_val):
    """Search through different configurations to find the optimal one"""
    # Define configuration grid
    vectorizer_types = ['bow', 'tfidf']
    stopwords_options = [None, 'english', 'custom']
    stopwords_counts = [10, 20, 30, 40, 50, 100, 200, 500]
    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    results = []
    best_accuracy = 0
    best_config = None
    best_model = None
    
    # Store full reports for each configuration
    all_reports = {}
    
    for vectorizer_type in vectorizer_types:
        for stopwords_option in stopwords_options:
            if stopwords_option == 'custom':
                for stopwords_count in stopwords_counts:
                    for c_value in c_values:
                        config = {
                            'vectorizer_type': vectorizer_type,
                            'stopwords_option': stopwords_option,
                            'stopwords_count': stopwords_count,
                            'c_value': c_value
                        }
                        
                        config_name = f"{vectorizer_type}_{stopwords_option}_{stopwords_count}_{c_value}"
                        print(f"\n{'='*50}")
                        print(f"Evaluating configuration: {config_name}")
                        print(f"{'='*50}")
                        
                        result = evaluate_configuration(X_train, y_train, X_val, y_val, config)
                        results.append(result)
                        all_reports[config_name] = result['report']
                        
                        if result['accuracy'] > best_accuracy:
                            best_accuracy = result['accuracy']
                            best_config = config
                            best_model = result['pipeline']
            else:
                for c_value in c_values:
                    config = {
                        'vectorizer_type': vectorizer_type,
                        'stopwords_option': stopwords_option,
                        'c_value': c_value
                    }
                    
                    config_name = f"{vectorizer_type}_{stopwords_option}_{c_value}"
                    print(f"\n{'='*50}")
                    print(f"Evaluating configuration: {config_name}")
                    print(f"{'='*50}")
                    
                    result = evaluate_configuration(X_train, y_train, X_val, y_val, config)
                    results.append(result)
                    all_reports[config_name] = result['report']
                    
                    if result['accuracy'] > best_accuracy:
                        best_accuracy = result['accuracy']
                        best_config = config
                        best_model = result['pipeline']
    
    return results, best_config, best_model, all_reports

def visualize_results(results):
    """Visualize the results of different configurations"""
    # Extract data for visualization
    df_results = pd.DataFrame([
        {
            'vectorizer_type': r['config']['vectorizer_type'],
            'stopwords_option': r['config']['stopwords_option'],
            'stopwords_count': r['config'].get('stopwords_count', 0),
            'c_value': r['config']['c_value'],
            'accuracy': r['accuracy'],
            'mcc': r['mcc'],
            'train_time': r['train_time']
        }
        for r in results
    ])
    
    # Create a configuration string for each row
    df_results['config_str'] = df_results.apply(
        lambda row: f"{row['vectorizer_type']}_" + 
                   f"{row['stopwords_option']}_" + 
                   (f"{int(row['stopwords_count'])}_" if row['stopwords_option'] == 'custom' else "") + 
                   f"C{row['c_value']}", 
        axis=1
    )
    
    # Plot accuracy by configuration
    plt.figure(figsize=(15, 10))
    
    # Filter for custom stopwords
    custom_df = df_results[df_results['stopwords_option'] == 'custom']
    sns.lineplot(data=custom_df, x='stopwords_count', y='accuracy', 
                 hue='vectorizer_type', style='c_value', markers=True, dashes=False)
    plt.title('Accuracy by Number of Stopwords (Custom)')
    plt.xlabel('Number of Stopwords')
    plt.ylabel('Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig('accuracy_by_stopwords_count.png')
    
    # Plot accuracy by C value
    plt.figure(figsize=(15, 10))
    sns.barplot(data=df_results, x='c_value', y='accuracy', hue='vectorizer_type')
    plt.title('Accuracy by C Value')
    plt.xlabel('C Value')
    plt.ylabel('Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig('accuracy_by_c_value.png')
    
    # Plot comparing all configurations
    plt.figure(figsize=(20, 10))
    
    # Sort by accuracy
    df_results = df_results.sort_values('accuracy')
    
    # Plot
    sns.barplot(data=df_results, x='config_str', y='accuracy')
    plt.title('Accuracy by Configuration')
    plt.xlabel('Configuration')
    plt.ylabel('Validation Accuracy')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig('accuracy_by_configuration.png')
    
    # NEW: Plot top 10 models by accuracy
    plt.figure(figsize=(16, 8))
    top10_acc = df_results.sort_values('accuracy', ascending=False).head(10)
    
    # Create a horizontal bar chart
    ax = sns.barplot(data=top10_acc, y='config_str', x='accuracy', palette='viridis')
    plt.title('Top 10 Models by Accuracy', fontsize=16)
    plt.xlabel('Accuracy', fontsize=12)
    plt.ylabel('Configuration', fontsize=12)
    
    # Add accuracy values as text
    for i, v in enumerate(top10_acc['accuracy']):
        ax.text(v + 0.001, i, f' {v:.4f}', va='center')
    
    plt.tight_layout()
    plt.grid(True, alpha=0.3, axis='x')
    plt.savefig('top10_models_by_accuracy.png')
    print("Top 10 models by accuracy visualization saved")
    
    # NEW: Plot top 10 models by MCC
    plt.figure(figsize=(16, 8))
    top10_mcc = df_results.sort_values('mcc', ascending=False).head(10)
    
    # Create a horizontal bar chart
    ax = sns.barplot(data=top10_mcc, y='config_str', x='mcc', palette='plasma')
    plt.title('Top 10 Models by Matthews Correlation Coefficient', fontsize=16)
    plt.xlabel('MCC', fontsize=12)
    plt.ylabel('Configuration', fontsize=12)
    
    # Add MCC values as text
    for i, v in enumerate(top10_mcc['mcc']):
        ax.text(v + 0.001, i, f' {v:.4f}', va='center')
    
    plt.tight_layout()
    plt.grid(True, alpha=0.3, axis='x')
    plt.savefig('top10_models_by_mcc.png')
    print("Top 10 models by MCC visualization saved")
    
    # NEW: Generate a table with top 10 models by accuracy
    print("\nTop 10 Models by Accuracy:")
    top10_acc_table = top10_acc[['config_str', 'accuracy', 'mcc', 'train_time']]
    print(top10_acc_table.to_string(index=False))
    
    # NEW: Generate a table with top 10 models by MCC
    print("\nTop 10 Models by MCC:")
    top10_mcc_table = top10_mcc[['config_str', 'mcc', 'accuracy', 'train_time']]
    print(top10_mcc_table.to_string(index=False))
    
    # Export top models data to CSV
    top10_acc.to_csv('top10_models_by_accuracy.csv', index=False)
    top10_mcc.to_csv('top10_models_by_mcc.csv', index=False)
    
    print("Visualizations and data files saved.")
    
    return df_results

def evaluate_best_model(model, X_val, y_val):
    """Detailed evaluation of the best model"""
    print("\nEvaluating best model on validation set...")
    y_val_pred = model.predict(X_val)
    
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
    plt.title('Confusion Matrix for Best Model')
    plt.tight_layout()
    plt.savefig('best_model_confusion_matrix.png')
    print("Confusion matrix visualization saved as 'best_model_confusion_matrix.png'")
    
    # Feature importance analysis
    if hasattr(model['classifier'], 'coef_'):
        feature_names = model['vectorizer'].get_feature_names_out()
        coefficients = model['classifier'].coef_[0]
        
        # Get the top 15 most important features for each class
        top_positive_idx = np.argsort(coefficients)[-15:]
        top_negative_idx = np.argsort(coefficients)[:15]
        
        print("\nTop positive features (positive sentiment):")
        for idx in top_positive_idx:
            print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")
        
        print("\nTop negative features (negative sentiment):")
        for idx in top_negative_idx:
            print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")
            
    return conf_matrix

def save_model(model, model_name='best_svm_model.pkl'):
    """Save the trained model pipeline"""
    print(f"\nSaving best model as '{model_name}'...")
    joblib.dump(model, model_name)
    print("Model saved successfully.")

def save_reports(all_reports, filename='all_classification_reports.json'):
    """Save all classification reports to a JSON file"""
    import json
    
    # Convert numpy values to Python types
    for config_name, report in all_reports.items():
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    report[class_name][metric] = float(value)
    
    with open(filename, 'w') as f:
        json.dump(all_reports, f, indent=4)
    
    print(f"All classification reports saved to {filename}")

def save_all_models_report(df_results, filename='all_models_comparison.csv'):
    """Save all model results to a CSV file for further analysis"""
    df_results.sort_values('accuracy', ascending=False).to_csv(filename, index=False)
    print(f"Full models comparison saved to {filename}")

def main():
    """Main function to execute the entire workflow"""
    print("Starting sentiment analysis model optimization...")
    
    # Load data
    X_train, y_train, X_val, y_val = load_data()
    
    # Grid search across configurations
    results, best_config, best_model, all_reports = grid_search_configurations(X_train, y_train, X_val, y_val)
    
    # Visualize results
    df_results = visualize_results(results)
    
    # Evaluate best model in detail
    conf_matrix = evaluate_best_model(best_model, X_val, y_val)
    
    # Save best model
    save_model(best_model, f"best_model_{best_config['vectorizer_type']}_{best_config.get('stopwords_option', 'none')}.pkl")
    
    # Save all classification reports
    save_reports(all_reports)
    
    # Save full model comparison
    save_all_models_report(df_results)
    
    # Find models with highest accuracy and MCC
    best_acc_config = df_results.sort_values('accuracy', ascending=False).iloc[0]
    best_mcc_config = df_results.sort_values('mcc', ascending=False).iloc[0]
    
    # Final report
    print("\n===== Final Report =====")
    print("Best Configuration by Accuracy:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
        
    best_result = [r for r in results if r['config'] == best_config][0]
    print(f"\nBest Validation Accuracy: {best_result['accuracy']:.4f}")
    print(f"Error Rate: {best_result['error_rate']:.4f}")
    print(f"Matthews Correlation Coefficient: {best_result['mcc']:.4f}")
    
    print("\nBest Configuration by MCC:")
    print(f"  Configuration: {best_mcc_config['config_str']}")
    print(f"  MCC: {best_mcc_config['mcc']:.4f}")
    print(f"  Accuracy: {best_mcc_config['accuracy']:.4f}")
    
    # Analysis of optimal parameters
    print("\nParameter Analysis:")
    
    # Vectorizer type analysis
    vectorizer_analysis = df_results.groupby('vectorizer_type')['accuracy'].mean().reset_index()
    best_vectorizer = vectorizer_analysis.loc[vectorizer_analysis['accuracy'].idxmax()]['vectorizer_type']
    print(f"Best vectorizer type: {best_vectorizer} (avg accuracy: {vectorizer_analysis['accuracy'].max():.4f})")
    
    # Stopwords analysis
    stopwords_analysis = df_results.groupby('stopwords_option')['accuracy'].mean().reset_index()
    best_stopwords = stopwords_analysis.loc[stopwords_analysis['accuracy'].idxmax()]['stopwords_option']
    print(f"Best stopwords option: {best_stopwords} (avg accuracy: {stopwords_analysis['accuracy'].max():.4f})")
    
    # Custom stopwords count analysis (if relevant)
    if 'custom' in df_results['stopwords_option'].values:
        custom_df = df_results[df_results['stopwords_option'] == 'custom']
        count_analysis = custom_df.groupby('stopwords_count')['accuracy'].mean().reset_index()
        best_count = count_analysis.loc[count_analysis['accuracy'].idxmax()]['stopwords_count']
        print(f"Best stopwords count: {int(best_count)} (avg accuracy: {count_analysis['accuracy'].max():.4f})")
    
    # C value analysis
    c_analysis = df_results.groupby('c_value')['accuracy'].mean().reset_index()
    best_c = c_analysis.loc[c_analysis['accuracy'].idxmax()]['c_value']
    print(f"Best C value: {best_c} (avg accuracy: {c_analysis['accuracy'].max():.4f})")
    
    print("========================")

if __name__ == "__main__":
    main()
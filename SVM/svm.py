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
from collections import Counter
import re
from time import time

# Dataset paths
splits = {
    'train': 'data/train-00000-of-00001.parquet',
    'validation': 'data/validation-00000-of-00001.parquet'
}

def tokenize(text):
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
    print("Loading datasets...")
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

# text processing and classification pipeline
def create_text_pipeline(vectorizer_type="tfidf", stop_words=None):

    vectorizer_params = {
        'min_df': 5,
        'max_df': 0.5,
        'stop_words': stop_words
    }
    vectorizer_params_bow = {
        'stop_words': stop_words
    }
    
    if vectorizer_type.lower() == "tfidf":
        vectorizer = TfidfVectorizer(**vectorizer_params)
    else:  
        vectorizer = CountVectorizer(**vectorizer_params_bow)
    
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', LinearSVC(dual="auto", class_weight='balanced'))
    ])
    
    return pipeline

# Evaluate a specific configuration
def evaluate_configuration(X_train, y_train, X_val, y_val, config):
    vectorizer_type = config['vectorizer_type']
    stopwords_option = config['stopwords_option']
    stopwords_count = config.get('stopwords_count', 0)
    c_value = config['c_value']
    
    stop_words = None
    if stopwords_option == 'english':
        stop_words = 'english'
    elif stopwords_option == 'custom':
        stop_words = list(build_custom_stopwords(X_train, top_n=stopwords_count))
    
    pipeline = create_text_pipeline(vectorizer_type, stop_words)
    
    # C value for LinearSVC
    pipeline.named_steps['classifier'].C = c_value
    pipeline.named_steps['classifier'].max_iter = 2000
    
    print(f"\nTraining model with configuration: {config}")
    start_time = time()
    pipeline.fit(X_train, y_train)
    train_time = time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    start_time = time()
    y_val_pred = pipeline.predict(X_val)
    pred_time = time() - start_time
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    error_rate = 1 - val_accuracy
    mcc = matthews_corrcoef(y_val, y_val_pred)
    
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

# Search through different configurations to find the optimal one
# Can't use GridsearchCV as I want the top n models stats and GridsearchCV only gives the CV scores.
def grid_search_configurations(X_train, y_train, X_val, y_val):

    vectorizer_types = ['bow', 'tfidf']
    stopwords_options = [None, 'english', 'custom']
    stopwords_counts = [10, 20, 30, 40, 50, 100, 200, 500]
    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    results = []
    best_accuracy = 0
    best_config = None
    best_model = None
    
    all_reports = {}

    # Again, bit ugly but it works.
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


# Visualization
def visualize_results(results):

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
    
    df_results['config_str'] = df_results.apply(
        lambda row: f"{row['vectorizer_type']}_" + 
                   f"{row['stopwords_option']}_" + 
                   (f"{int(row['stopwords_count'])}_" if row['stopwords_option'] == 'custom' else "") + 
                   f"C{row['c_value']}", 
        axis=1
    )
    
    # Sort by accuracy
    df_results = df_results.sort_values('accuracy')
    
    top10_acc = df_results.head(10)
    top10_mcc = df_results.sort_values('mcc', ascending=False).head(10)
    
    top10_acc.to_csv('top10_models_by_accuracy.csv', index=False)
    top10_mcc.to_csv('top10_models_by_mcc.csv', index=False)

    return df_results

# All the top 10 models are extracted but only the best one is evaluated
# (confusion matrix and feature importance)
def evaluate_best_model(model, X_val, y_val):

    y_val_pred = model.predict(X_val)
    
    conf_matrix = confusion_matrix(y_val, y_val_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix for Best Model')
    plt.tight_layout()
    plt.savefig('best_model_confusion_matrix.png')
    print("Confusion matrix saved")
            
    return conf_matrix

# Every runs' report saved
def save_reports(all_reports, filename='all_classification_reports.json'):
    import json
    
    for config_name, report in all_reports.items():
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    report[class_name][metric] = float(value)
    
    with open(filename, 'w') as f:
        json.dump(all_reports, f, indent=4)
    
    print(f"All classification reports saved to {filename}")

# Saving model stats
def save_all_models_report(df_results, filename='all_models_comparison.csv'):
    """Save all model results to a CSV file for further analysis"""
    df_results.sort_values('accuracy', ascending=False).to_csv(filename, index=False)
    print(f"Full models comparison saved to {filename}")

def main():
    
    X_train, y_train, X_val, y_val = load_data()
    
    results, best_config, best_model, all_reports = grid_search_configurations(X_train, y_train, X_val, y_val)
    
    df_results = visualize_results(results)
    
    conf_matrix = evaluate_best_model(best_model, X_val, y_val)
    
    save_reports(all_reports)
    
    save_all_models_report(df_results)
    
    best_acc_config = df_results.sort_values('accuracy', ascending=False).iloc[0]
    best_mcc_config = df_results.sort_values('mcc', ascending=False).iloc[0]
    
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
    
    print("\nParameter Analysis:")
    
    # Vectorizer type analysis
    vectorizer_analysis = df_results.groupby('vectorizer_type')['accuracy'].mean().reset_index()
    best_vectorizer = vectorizer_analysis.loc[vectorizer_analysis['accuracy'].idxmax()]['vectorizer_type']
    print(f"Best vectorizer type: {best_vectorizer} (avg accuracy: {vectorizer_analysis['accuracy'].max():.4f})")
    
    # Stopwords analysis
    stopwords_analysis = df_results.groupby('stopwords_option')['accuracy'].mean().reset_index()
    best_stopwords = stopwords_analysis.loc[stopwords_analysis['accuracy'].idxmax()]['stopwords_option']
    print(f"Best stopwords option: {best_stopwords} (avg accuracy: {stopwords_analysis['accuracy'].max():.4f})")
    
    print("========================")

if __name__ == "__main__":
    main()
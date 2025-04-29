import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
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
# This is done because sklearn's 'english' stop-words list is quite limited
def build_custom_stopwords(corpus, top_n=50):
    word_counts = Counter()
    for doc in corpus:
        tokens = tokenize(doc)
        word_counts.update(tokens)
    return set([word for word, _ in word_counts.most_common(top_n)])


# Loading data and splitting it into features and labels
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
        'min_df': 5,
        'max_df': 0.5,
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
    # Using RandomForestClassifier with class_weight='balanced'
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', RandomForestClassifier(class_weight='balanced'))
    ])
    
    return pipeline

def evaluate_configuration(X_train, y_train, X_val, y_val, config):
    """Evaluate a specific configuration"""
    vectorizer_type = config['vectorizer_type']
    stopwords_option = config['stopwords_option']
    stopwords_count = config.get('stopwords_count', 0)
    n_estimators = config['n_estimators']
    max_depth = config['max_depth']
    ccp_alpha = config['ccp_alpha']  # Cost complexity pruning parameter
    
    # Prepare stopwords
    stop_words = None
    if stopwords_option == 'english':
        stop_words = 'english'
    elif stopwords_option == 'custom':
        stop_words = list(build_custom_stopwords(X_train, top_n=stopwords_count))
    
    # Create pipeline
    pipeline = create_text_pipeline(vectorizer_type, stop_words)
    
    # Set RandomForest parameters
    pipeline.named_steps['classifier'].n_estimators = n_estimators
    pipeline.named_steps['classifier'].max_depth = max_depth
    pipeline.named_steps['classifier'].ccp_alpha = ccp_alpha
    pipeline.named_steps['classifier'].random_state = 42  # For reproducibility
    
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
    # Define configuration grid for Random Forest with cost complexity pruning
    vectorizer_types = ['bow', 'tfidf']
    stopwords_options = [None, 'english', 'custom']
    stopwords_counts = [10, 20, 30, 40, 50, 100, 200, 500]
    n_estimators_values = [100, 200, 300]  # Number of trees
    max_depth_values = [None, 10, 20, 30]  # Max depth of trees (None means unlimited)
    ccp_alpha_values = [0.0, 0.001, 0.01, 0.05, 0.1]  # Cost complexity pruning parameter
    
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
                    for n_estimators in n_estimators_values:
                        for max_depth in max_depth_values:
                            for ccp_alpha in ccp_alpha_values:
                                config = {
                                    'vectorizer_type': vectorizer_type,
                                    'stopwords_option': stopwords_option,
                                    'stopwords_count': stopwords_count,
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'ccp_alpha': ccp_alpha
                                }
                                
                                config_name = f"{vectorizer_type}_{stopwords_option}_{stopwords_count}_trees{n_estimators}_depth{max_depth}_ccp{ccp_alpha}"
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
                for n_estimators in n_estimators_values:
                    for max_depth in max_depth_values:
                        for ccp_alpha in ccp_alpha_values:
                            config = {
                                'vectorizer_type': vectorizer_type,
                                'stopwords_option': stopwords_option,
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'ccp_alpha': ccp_alpha
                            }
                            
                            config_name = f"{vectorizer_type}_{stopwords_option}_trees{n_estimators}_depth{max_depth}_ccp{ccp_alpha}"
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
    # Extract data for visualization
    df_results = pd.DataFrame([
        {
            'vectorizer_type': r['config']['vectorizer_type'],
            'stopwords_option': r['config']['stopwords_option'],
            'stopwords_count': r['config'].get('stopwords_count', 0),
            'n_estimators': r['config']['n_estimators'],
            'max_depth': r['config']['max_depth'],
            'ccp_alpha': r['config']['ccp_alpha'],
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
                   f"trees{row['n_estimators']}_" +
                   f"depth{row['max_depth']}_" +
                   f"ccp{row['ccp_alpha']}", 
        axis=1
    )
    
    # Sort by accuracy (descending)
    df_results = df_results.sort_values('accuracy', ascending=False)
    
    # top-10 models by accuracy and mcc
    top10_acc = df_results.head(10)
    top10_mcc = df_results.sort_values('mcc', ascending=False).head(10)
    
    # Export top models data to CSV
    top10_acc.to_csv('top10_models_by_accuracy.csv', index=False)
    top10_mcc.to_csv('top10_models_by_mcc.csv', index=False)

    # Visualization for ccp_alpha effect
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='ccp_alpha', y='accuracy', data=df_results)
    plt.title('Effect of Cost Complexity Pruning on Accuracy')
    plt.xlabel('ccp_alpha')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig('ccp_alpha_effect.png')
    plt.close()
    
    # Visualization for max_depth effect
    plt.figure(figsize=(12, 8))
    # Convert None to string for plotting
    df_results['max_depth_str'] = df_results['max_depth'].apply(lambda x: 'None' if x is None else str(x))
    sns.boxplot(x='max_depth_str', y='accuracy', data=df_results)
    plt.title('Effect of Max Depth on Accuracy')
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig('max_depth_effect.png')
    plt.close()
    
    # Visualization of n_estimators effect
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='n_estimators', y='accuracy', data=df_results)
    plt.title('Effect of Number of Estimators (Trees) on Accuracy')
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig('n_estimators_effect.png')
    plt.close()

    return df_results

def evaluate_best_model(model, X_val, y_val):
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
    plt.title('Confusion Matrix for Best Random Forest Model')
    plt.tight_layout()
    plt.savefig('rf_best_model_confusion_matrix.png')
    print("Confusion matrix saved")
    
    # Feature importance analysis for Random Forest
    if hasattr(model['classifier'], 'feature_importances_'):
        feature_names = model['vectorizer'].get_feature_names_out()
        feature_importances = model['classifier'].feature_importances_
        
        # Get the top 15 most important features
        top_features_idx = np.argsort(feature_importances)[-15:]
        
        # Create a dataframe of feature importances
        feature_df = pd.DataFrame({
            'Feature': feature_names[top_features_idx],
            'Importance': feature_importances[top_features_idx]
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 15 important features:")
        for idx, row in feature_df.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_df)
        plt.title('Top 15 Feature Importances in Random Forest Model')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        # Analyze tree structures
        rf_classifier = model['classifier']
        print("\nRandom Forest Analysis:")
        print(f"Number of trees in the forest: {rf_classifier.n_estimators}")
        
        # Get tree statistics
        depths = []
        leaves = []
        
        for tree in rf_classifier.estimators_:
            depths.append(tree.get_depth())
            leaves.append(tree.get_n_leaves())
        
        avg_depth = np.mean(depths)
        avg_leaves = np.mean(leaves)
        
        print(f"Average tree depth: {avg_depth:.2f}")
        print(f"Average number of leaves: {avg_leaves:.2f}")
        
        # Plot tree depth distribution
        plt.figure(figsize=(10, 6))
        plt.hist(depths, bins=range(min(depths), max(depths) + 2), alpha=0.7)
        plt.title('Distribution of Tree Depths in Random Forest')
        plt.xlabel('Depth')
        plt.ylabel('Number of Trees')
        plt.grid(True, alpha=0.3)
        plt.savefig('tree_depth_distribution.png')
        plt.close()
            
    return conf_matrix

def save_reports(all_reports, filename='all_classification_reports.json'):
    # Convert numpy values to Python types
    for config_name, report in all_reports.items():
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    report[class_name][metric] = float(value)
    
    with open(filename, 'w') as f:
        json.dump(all_reports, f, indent=4)
    
    print(f"All classification reports saved to {filename}")

def save_all_models_report(df_results, filename='rf_all_models_comparison.csv'):
    """Save all model results to a CSV file for further analysis"""
    df_results.to_csv(filename, index=False)
    print(f"Full models comparison saved to {filename}")

def main():
    print("Starting Random Forest sentiment analysis with cost complexity pruning...")
    
    # Load data
    X_train, y_train, X_val, y_val = load_data()
    
    # Grid search across configurations
    results, best_config, best_model, all_reports = grid_search_configurations(X_train, y_train, X_val, y_val)
    
    # Visualize results
    df_results = visualize_results(results)
    
    # Evaluate best model in detail
    conf_matrix = evaluate_best_model(best_model, X_val, y_val)
    
    # Save all classification reports
    save_reports(all_reports, 'rf_all_classification_reports.json')
    
    # Save full model comparison
    save_all_models_report(df_results, 'rf_all_models_comparison.csv')
    
    # Find models with highest accuracy and MCC
    best_acc_config = df_results.iloc[0]  # Already sorted by accuracy desc
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
    
    # Random forest specific parameter analysis
    # n_estimators analysis
    n_estimators_analysis = df_results.groupby('n_estimators')['accuracy'].mean().reset_index()
    best_n_estimators = n_estimators_analysis.loc[n_estimators_analysis['accuracy'].idxmax()]['n_estimators']
    print(f"Best number of trees: {best_n_estimators} (avg accuracy: {n_estimators_analysis['accuracy'].max():.4f})")
    
    # max_depth analysis
    # Handle None values for aggregation
    df_results['max_depth_str'] = df_results['max_depth'].apply(lambda x: 'None' if x is None else str(x))
    max_depth_analysis = df_results.groupby('max_depth_str')['accuracy'].mean().reset_index()
    best_max_depth = max_depth_analysis.loc[max_depth_analysis['accuracy'].idxmax()]['max_depth_str']
    print(f"Best max depth: {best_max_depth} (avg accuracy: {max_depth_analysis['accuracy'].max():.4f})")
    
    # ccp_alpha analysis
    ccp_alpha_analysis = df_results.groupby('ccp_alpha')['accuracy'].mean().reset_index()
    best_ccp_alpha = ccp_alpha_analysis.loc[ccp_alpha_analysis['accuracy'].idxmax()]['ccp_alpha']
    print(f"Best ccp_alpha (pruning parameter): {best_ccp_alpha} (avg accuracy: {ccp_alpha_analysis['accuracy'].max():.4f})")
    
    print("========================")

if __name__ == "__main__":
    main()
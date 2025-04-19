import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Define dataset paths
splits = {
    'train': 'data/train-00000-of-00001.parquet',
    'validation': 'data/validation-00000-of-00001.parquet'
}

def load_data():
    """Load train and validation datasets"""
    print("Loading datasets...")
    base_path = "hf://datasets/stanfordnlp/sst2/"
    df_train = pd.read_parquet(base_path + splits["train"])
    df_val = pd.read_parquet(base_path + splits["validation"])
    
    X_train, y_train = df_train['sentence'], df_train['label']
    X_val, y_val = df_val['sentence'], df_val['label']
    
    print(f"Train set size: {len(X_train)} samples")
    print(f"Validation set size: {len(X_val)} samples")
    
    print("\nClass distribution:")
    print("Training set:", pd.Series(y_train).value_counts().to_dict())
    print("Validation set:", pd.Series(y_val).value_counts().to_dict())
    
    return X_train, y_train, X_val, y_val

def create_text_pipeline(classifier_type="svm", vectorizer_type="tfidf", use_stop_words=True):
    """Create a text processing and classification pipeline"""
    vectorizer_params = {
        'min_df': 5,
        'max_df': 0.8,
    }
    
    if not use_stop_words:
        vectorizer_params['stop_words'] = 'english'
    
    if vectorizer_type.lower() == "tfidf":
        vectorizer = TfidfVectorizer(**vectorizer_params)
        vectorizer_suffix = "tfidf"
    else:
        vectorizer = CountVectorizer(**vectorizer_params)
        vectorizer_suffix = "bow"
    
    if classifier_type.lower() == "svm":
        classifier = LinearSVC(dual="auto", class_weight='balanced')
        classifier_suffix = "svm"
    else:
        classifier = RandomForestClassifier(class_weight='balanced')
        classifier_suffix = "rf"
    
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    stop_words_suffix = "no_stop" if not use_stop_words else "with_stop"
    model_suffix = f"{classifier_suffix}_{vectorizer_suffix}_{stop_words_suffix}"
    
    return pipeline, model_suffix

def get_best_ccp_alphas(X_train, y_train, max_alphas=25):
    """Compute possible ccp_alpha values from pruning path"""
    vectorizer = TfidfVectorizer()  # or use CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    clf = DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas
    # Optional: Limit the number of values to avoid too many models
    if len(alphas) > max_alphas:
        step = len(alphas) // max_alphas
        alphas = alphas[::step]
    return list(set(alphas))  # Remove duplicates

def plot_ccp_alpha_vs_impurity(X_train, y_train):
    vectorizer = TfidfVectorizer()  # or use CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    clf = DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    plt.plot(path.ccp_alphas[:-1], path.impurities[:-1], marker='o', drawstyle='steps-post')
    plt.xlabel("Effective alpha")
    plt.ylabel("Total impurity of leaves")
    plt.title("Total Impurity vs Effective Alpha")
    plt.grid(True)
    plt.show()

def main():
    X_train, y_train, X_val, y_val = load_data()
    ccp_alphas = get_best_ccp_alphas(X_train, y_train)
    print("Best candidate ccp_alpha values:", ccp_alphas)
    plot_ccp_alpha_vs_impurity(X_train, y_train)

#if __name__ == "__main__":
#    main()
import sys
import pickle
import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'chapter\s+\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

def load_texts(folder, label):
    # Reads all .txt files in a folder and returns a dataframe with text and label.
    dataset = []
    for fname in os.listdir(folder):
        if fname.endswith('.txt'):
            try:
                with open(os.path.join(folder, fname), 'r', encoding='utf-8') as file:
                    text = file.read()
                    text = clean_text(text)
                    dataset.append({"text_id": fname.split('.')[0], "text": text, "label": label})
            except Exception as e:
                print(f"Error reading {fname}: {e}")
    return pd.DataFrame(dataset)

def load_data():
    romantic_df = load_texts('data/romance', label=1)
    non_romantic_df = load_texts('data/non_romance', label=0)
    full_df = pd.concat([romantic_df, non_romantic_df], ignore_index=True)
    return full_df

def split_data(data, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
    train_df, dev_df = train_test_split(train_df, test_size=0.25, random_state=42, stratify=train_df['label'])
    return train_df, test_df, dev_df


def train_model(train_df, dev_df, test_df):
    X_train, y_train = train_df['text'], train_df['label']
    X_dev, y_dev = dev_df['text'], dev_df['label']
    X_test, y_test = test_df['text'], test_df['label']

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=10000,
        ngram_range=(1, 1),
        stop_words='english',
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_dev_tfidf = vectorizer.transform(X_dev)
    X_test_tfidf = vectorizer.transform(X_test)

    # SGDClassifier with Logistic Regression

    classifier = SGDClassifier(
        loss='log_loss',  # Logistic Regression
        penalty='l2',
        alpha=1e-4,
        random_state=42,
        max_iter=1000,
    )
    classifier.fit(X_train_tfidf, y_train)
    # Evaluation on dev set
    dev_pred = classifier.predict(X_dev_tfidf)
    dev_acc = accuracy_score(y_dev, dev_pred)
    print(f"Dev Accuracy: {dev_acc:.4f}")
    print(classification_report(y_dev, dev_pred))

    # Final evaluation on test set
    test_pred = classifier.predict(X_test_tfidf)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(classification_report(y_test, test_pred))

    # final training on full training data (train + dev)
    X_full_train = vectorizer.fit_transform(pd.concat([X_train, X_dev]))
    y_full_train = pd.concat([y_train, y_dev])

    classifier.fit(X_full_train, y_full_train)

    print("Model saved to model.dat")
    return classifier, vectorizer


if __name__ == "__main__":
    data = load_data()
    train_df, test_df, dev_df = split_data(data)

    model, vectorizer = train_model(train_df, dev_df, test_df)
    # Save the model

    with open("model.dat", 'wb') as file:
        pickle.dump((vectorizer, model), file)
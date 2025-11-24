# CSCI 4541 Natural Language Processing HW 4 (Prof. Andy Exley - University of Minnesota Twin Cities)
# Adithya Saravu, Angela Lenling

# Our results on the test set are: 
# Test Accuracy: 0.8375
#               precision    recall  f1-score   support

#            0       0.85      0.82      0.83       360
#            1       0.82      0.86      0.84       360

#     accuracy                           0.84       720
#    macro avg       0.84      0.84      0.84       720
# weighted avg       0.84      0.84      0.84       720

import sys
import pickle
import os

def load_model(model_path):
    with open(model_path, 'rb') as file:
        vectorizer, classifier = pickle.load(file)
    return vectorizer, classifier

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def classify(text, model, vectorizer):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    label = "Romance" if prediction == 1 else "Non-Romance"
    return label


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python classify.py <model_path> <text_path>")
        sys.exit(1)

    model_name, test_doc = sys.argv[1], sys.argv[2]

    if not os.path.exists(model_name):
        print(f"Model not found: {model_name}")
        sys.exit(1)

    if not os.path.exists(test_doc):
        print(f"Test document not found: {test_doc}")
        sys.exit(1)
    
    #load model and classify
    vectorizer, model = load_model(model_name)
    text = load_text(test_doc)

    label = classify(text, model, vectorizer)
    print(f"\nClassification result:  {label}\n")


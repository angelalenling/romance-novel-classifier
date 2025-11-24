Romance vs Non-Romance Book Classifier
======================================

Authors: Adithya Saravu & Angela Lenling
Class Project: CSCI 4541 – Natural Language Processing (Prof. Andy Exley)

--------------------------------------
Project Overview
--------------------------------------

This project implements a binary text classifier that predicts whether a novel is:

- Romance (1)
- Non-Romance (0)

We use a stochastic gradient descent (SGD) classifier with logistic regression from scikit-learn to classify full-text novels from Project Gutenberg.

The repository contains:

- classifier_train.py – trains the model and saves it as model.dat
- classify.py – loads a trained model and classifies a new .txt document as Romance or Non-Romance
- model.dat – pickled (vectorizer, classifier) pair
- data/romance/ – romance novels (.txt)
- data/non_romance/ – non-romance novels (.txt)
- 
Note: Raw text data is not included in this repository; see the Data section below.

--------------------------------------
Data
--------------------------------------

The raw training data (full novels) from Project Gutenberg is NOT included in this repository due to size and distribution concerns.

At training time, the code expects the following local folder structure:

    data/
      romance/       # Romance novels as .txt files
      non_romance/   # Non-romance novels as .txt files

In this GitHub repo, these folders only contain placeholder ".gitkeep" files so that the expected structure is visible. To retrain the model, download your own corpus from Project Gutenberg (or another source), label it as Romance (1) or Non-Romance (0), and place the `.txt` files into `data/romance` and `data/non_romance` before running `classifier_train.py`.

--------------------------------------
Classification Task
--------------------------------------

Our binary decision is:

Romance (1) vs Non-Romance (0)

We randomly selected:

- 1800 romance novels
- 1800 non-romance novels

These labels were derived from pg_catalog.csv of the Project Gutenberg data feed:
https://www.gutenberg.org/cache/epub/feeds/

--------------------------------------
Data Splits
--------------------------------------

We shuffle and combine all labeled texts, then split:

- 80% for train + dev
  - From this, 25% is held out as dev
  - Effective: 60% train, 20% dev
- 20% held out as test

All splits are stratified by label to preserve class balance.

--------------------------------------
Method
--------------------------------------

Vectorization (TF–IDF):

- lowercase = True
- max_features = 5000
- ngram_range = (1, 2) (unigrams + bigrams)
- stop_words = "english"

Classifier (SGDClassifier configured as logistic regression):

- loss = "log_loss"
- penalty = "l2"
- alpha = 1e-4
- random_state = 42
- max_iter = 1000

Workflow:

1. Fit the TF–IDF vectorizer on the training set.
2. Train SGDClassifier on the training vectors.
3. Evaluate on the dev set.
4. Evaluate final model on the test set.
5. Retrain on train + dev and save (vectorizer, classifier) to model.dat using pickle.

--------------------------------------
Results
--------------------------------------

Test set performance (on 720 held-out novels):

Test Accuracy: 0.8375 (≈ 0.84)

--------------------------------------
Academic Context
--------------------------------------

This repository was created as part of a homework assignment for:

CSCI 4541 – Natural Language Processing  
University of Minnesota  
Instructor: Prof. Andy  
Partners: Adithya Saravu & Angela Lenling

# Spam Text Classifier

This project implements a text classification model trained on SMS/text message data to distinguish between spam and non-spam (ham) messages. The model applies natural language preprocessing and TF-IDF vectorization before training a Support Vector Classifier to make binary predictions on unlabeled message data, outputting results to a `predictions.csv` file.

## Capabilities

- **Text Preprocessing**: Converts text to lowercase, removes punctuation, and strips digits to normalize input data
- **TF-IDF Vectorization**: Converts cleaned text into numerical feature vectors using `TfidfVectorizer` with English stop-word removal and a 5,000-feature limit
- **Support Vector Classification**: Trains a `LinearSVC` model to classify messages as spam or ham based on learned text patterns
- **Prediction Export**: Generates a `predictions.csv` file with binary labels (`TRUE` for spam, `FALSE` for ham) for each test message

## Usage

### Terminal MAC Run Script

```bash
bash run.sh
```

or directly:

```bash
python3 model.py
```

**Required input files:**
- `data_train_hw4_problem1.csv` — Labeled training data with `spam` and `text` columns
- `data_test_hw4_problem1.csv` — Unlabeled test data with `text` column

**Output:**
- `predictions.csv` — Binary classification results with `spam` column (`TRUE` or `FALSE`)

## Use Cases

### Personal Message Filtering

Evaluate SMS or text message datasets to identify unwanted spam messages for personal or organizational message triage.

### Research on Text Classification

Serve as a baseline or reference implementation for studying linear SVM-based text classification pipelines using TF-IDF features.

## Research Purposes

Designed for research purposes. Penn State University (PSU), IST 557 Data Mining. Fall 2025.

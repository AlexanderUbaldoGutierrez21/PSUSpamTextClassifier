import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer
import re

# LOAD TRAINING DATA
train_df = pd.read_csv('data_train_hw4_problem1.csv', encoding='latin1')
train_df['spam'] = train_df['spam'].astype(int)

# LOAD TEST DATA
test_df = pd.read_csv('data_test_hw4_problem1.csv', encoding='latin1')

# PREPROCESS FUNCTION
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# CLEAN TEXTS
train_df['text'] = train_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)

# VECTORIZE
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])
y_train = train_df['spam']

# TRAIN MODEL
model = LinearSVC(random_state=42, max_iter=10000)
model.fit(X_train, y_train)

# CROSS-VALIDATION BALANCED ACCURACY
scores = cross_val_score(model, X_train, y_train, cv=5, scoring=make_scorer(balanced_accuracy_score))
print(f"Cross-validation balanced accuracy: {scores.mean():.4f}")

# PREDICT
predictions = model.predict(X_test)

# OUTPUT PREDICTIONS
import csv
with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['spam'])
    for pred in predictions:
        writer.writerow(['TRUE' if pred == 1 else 'FALSE'])
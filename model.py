import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
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

# PREDICT
predictions = model.predict(X_test)

# OUTPUT PREDICTIONS
for pred in predictions:
    print('TRUE' if pred == 1 else 'FALSE')
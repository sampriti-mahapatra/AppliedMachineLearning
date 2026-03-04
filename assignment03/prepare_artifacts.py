"""
Prepare model artifacts for AML_3.
Loads the trained RandomForestClassifier from AML_2,
re-fits the TfidfVectorizer on training data, and saves both as .pkl files.
"""
import pandas as pd
import joblib
import cloudpickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load training data
train_df = pd.read_csv('/Users/sampriti/Downloads/cmi/AML_2/train.csv')
print(f'Training data: {train_df.shape}')

# Re-fit the vectorizer with same params as train.ipynb
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
tfidf.fit(train_df['cleaned_message'].fillna(''))
print(f'Vectorizer fitted: {len(tfidf.vocabulary_)} features')

# Save vectorizer
joblib.dump(tfidf, 'vectorizer.pkl')
print('Saved vectorizer.pkl')

# Load the trained model from AML_2 mlruns
model_path = (
    '/Users/sampriti/Downloads/cmi/AML_2/mlruns/306097714134317456/'
    'models/m-b08c8520fd38426286c4b2c7ee2c8896/artifacts/model.pkl'
)
with open(model_path, 'rb') as f:
    model = cloudpickle.load(f)
print(f'Model loaded: {type(model).__name__}')

# Save model
joblib.dump(model, 'model.pkl')
print('Saved model.pkl')

# Quick sanity test
X_test = tfidf.transform(['Free cash prize winner call now'])
proba = model.predict_proba(X_test)[0]
print(f'Test prediction proba (ham, spam): {proba}')

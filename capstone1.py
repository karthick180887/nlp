import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the dataset
df = pd.read_csv('tweets.csv')

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Function for tokenization and stemming
def preprocess(text):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    stems = [stemmer.stem(token) for token in tokens if token.isalpha()]  # Stem and filter out non-alphabetic tokens
    return ' '.join(stems)

# Apply preprocessing to the dataset
df['processed_text'] = df['text'].apply(preprocess)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])
y = df['sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training using Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

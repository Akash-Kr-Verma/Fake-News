import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

# Load the datasets
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

# Load the datasets
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

# Add a 'class' column to distinguish fake (0) and true (1) news
data_fake["class"] = 0
data_true["class"] = 1

# Combine the datasets
data = pd.concat([data_fake, data_true], axis=0)

# Drop unnecessary columns
data = data.drop(['title', 'subject', 'date'], axis=1, errors='ignore')

# Balance the dataset (if needed)
fake_news = data[data['class'] == 0]
true_news = data[data['class'] == 1]
true_news_oversampled = resample(true_news, replace=True, n_samples=len(fake_news), random_state=42)
balanced_data = pd.concat([fake_news, true_news_oversampled])
balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Apply preprocessing to the 'text' column
balanced_data['text'] = balanced_data['text'].apply(wordopt)

# Split the data into features (x) and labels (y)
x = balanced_data['text']
y = balanced_data['class']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Vectorize the text data using TF-IDF
vectorization = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train the Logistic Regression model
LR = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')
LR.fit(xv_train, y_train)

# Evaluate the model
y_pred = LR.predict(xv_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Function to predict and print the result
def predict_news(news):
    news = wordopt(news)
    news_vectorized = vectorization.transform([news])
    prediction = LR.predict(news_vectorized)
    return "Fake News" if prediction[0] == 0 else "Not Fake News"

# Test the function with user input
if __name__ == "__main__":
    news = input("Enter the news text: ")
    result = predict_news(news)
    print(f"Prediction: {result}")
# Add a 'class' column to distinguish fake (0) and true (1) news
data_fake["class"] = 0
data_true["class"] = 1

# Combine the datasets
data = pd.concat([data_fake, data_true], axis=0)

# Drop unnecessary columns
data = data.drop(['title', 'subject', 'date'], axis=1, errors='ignore')

# Balance the dataset (if needed)
fake_news = data[data['class'] == 0]
true_news = data[data['class'] == 1]
true_news_oversampled = resample(true_news, replace=True, n_samples=len(fake_news), random_state=42)
balanced_data = pd.concat([fake_news, true_news_oversampled])
balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Apply preprocessing to the 'text' column
balanced_data['text'] = balanced_data['text'].apply(wordopt)

# Split the data into features (x) and labels (y)
x = balanced_data['text']
y = balanced_data['class']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Vectorize the text data using TF-IDF
vectorization = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train the Logistic Regression model
LR = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')
LR.fit(xv_train, y_train)

# Evaluate the model
y_pred = LR.predict(xv_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Function to predict and print the result
def predict_news(news):
    news = wordopt(news)
    news_vectorized = vectorization.transform([news])
    prediction = LR.predict(news_vectorized)
    return "Fake News" if prediction[0] == 0 else "Not Fake News"

# Test the function with user input
if __name__ == "__main__":
    news = input("Enter the news text: ")
    result = predict_news(news)
    print(f"Prediction: {result}")
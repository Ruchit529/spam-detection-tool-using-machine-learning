import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
nltk.download('stopwords')


df=pd.read_csv("spam_ham_dataset.csv")
df=df.drop(columns=["Unnamed: 0","label_num"])
df['spam']=np.where(df['label']=='spam', 1,0)
df.isnull().sum()

df=df.dropna()
df=df.drop_duplicates()

# print(f"Duplicated values in a dataset: {df.duplicated().sum()}")

def clean_text(text):

    text=''.join([char for char in text if char not in string.punctuation])
    words=text.split()
    stop_words=stopwords.words("english")
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
    return ' '.join(words)


df['clean_text'] = df['text'].apply(clean_text)

df=df.drop(columns=['text'])

vectorizer = CountVectorizer()
X=vectorizer.fit_transform(df["clean_text"])
y=df["spam"]

X_train, X_text, y_train, y_test=train_test_split(X, y ,test_size=0.2,random_state=42)

# Define models
models = {
    "Naive Bayes": MultinomialNB()
    # "SVM": SVC(),
}

# best_results = {}
# best_model_name = None
# best_accuracy = 0.0

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)  # Training  the model
    y_pred = model.predict(X_text)  # Predict on the test set

with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
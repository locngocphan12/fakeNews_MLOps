import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def prepare_data():
    df_fake = pd.read_csv("mlflow/data/News _dataset/Fake.csv")
    df_real = pd.read_csv("mlflow/data/News _dataset/Real.csv")
    df_fake['label'] = "Fake"
    df_real['label'] = "Real"
    df = pd.concat([df_fake, df_real], ignore_index = True)
    X = df.drop(column = ['label'])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize a TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    # Fit and transform train set, transform test set
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    with open("mlflow/data/X_train.pkl", "wb") as f:
        pickle.dump(tfidf_train, f)
    with open("mlflow/data/X_test.pkl", "wb") as f:
        pickle.dump(tfidf_test, f)
    with open("mlflow/data/y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
    with open("mlflow/data/y_test.pkl", "wb") as f:
        pickle.dump(y_test, f)


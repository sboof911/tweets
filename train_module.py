import pandas as pd
from typing import Tuple
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

def clean_Data(folderPath : str, filesName : str) -> pd.DataFrame:
    def load_data() -> pd.DataFrame:
        DataFrame = pd.DataFrame(columns=["sentiment", "text"])
        for filename in filesName:
            filename : str = filename
            with open(folderPath + filename, 'r') as file:
                content = file.read()
                text_list = content.split(',')
            temp_df = pd.DataFrame({'text': text_list})
            temp_df["sentiment"] = filename.removeprefix("processed").removesuffix(".csv")
            DataFrame = pd.concat([DataFrame, temp_df], ignore_index=True)
        return DataFrame

    DataFrame = load_data()
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    def preprocess_text(text : str):
        text = text.lower()  # Lowercasing
        text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Removing speacial characters
        return ' '.join([word for word in text.split() if word not in stop_words])  # Removing stopwords

    DataFrame['text'] = DataFrame['text'].apply(preprocess_text)
    DataFrame = DataFrame.drop_duplicates(subset=['text', 'sentiment'])

    return DataFrame

def VectorizeWords(DataFrame : pd.DataFrame) -> Tuple[TfidfVectorizer, np.ndarray, pd.Series]:
    def wordToVector():
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(DataFrame['text']).toarray()
        y = DataFrame['sentiment']
        return vectorizer, x, y

    vectorizer, x, y = wordToVector()
    return vectorizer, x, y

def Top10Similarity(DataFrame : pd.DataFrame, x : np.ndarray) -> None:
    def compute_similarity_pairs(matrix : np.ndarray):
        pairs = []
        num_tweets = matrix.shape[0]
        for i, j in combinations(range(num_tweets), 2):
            similarity = matrix[i, j]
            pairs.append((i, j, similarity))
        return pairs

    similarity_matrix = cosine_similarity(x)
    pairs = compute_similarity_pairs(similarity_matrix)

    # Get the top-10 most similar pairs
    top_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:10]
    # Print the top-10 most similar pairs
    for i, j, sim in top_pairs:
        tweet1 = DataFrame.iloc[i]['text']
        tweet2 = DataFrame.iloc[j]['text']
        print(f"Similarity: {sim:.4f}")
        print(f"Tweet 1: {tweet1}")
        print(f"Tweet 2: {tweet2}")
        print("-" * 80)

def TrainModel(vectorizer : TfidfVectorizer, x : np.ndarray, y : pd.Series) -> None:
    # Split data into training and testing sets
    model = LogisticRegression(max_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))
    with open('predictions.pkl', 'wb') as file:
        pickle.dump((vectorizer, model), file)


if __name__ == "__main__":
    import os

    try:
        folderPath = "./p00_tweets/"
        filesName = ["processedNegative.csv", "processedNeutral.csv", "processedPositive.csv"]
        for filename in filesName:
            if not os.path.exists(folderPath + filename):
                raise Exception(f"The file '{folderPath + filename}' does not exist.")

        DataFrame = clean_Data(folderPath, filesName)
        vectorizer, x, y = VectorizeWords(DataFrame)
        Top10Similarity(DataFrame, x)
        TrainModel(vectorizer, x, y)
    except Exception as e:
        print(f"Exeption Error: {e}")

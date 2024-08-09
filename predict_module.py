from nltk.corpus import stopwords
import pickle
import pandas as pd

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def predict_sentiment(text):
    with open('predictions.pkl', 'rb') as file:
        vectorizer, model = pickle.load(file)

    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    sentiment = model.predict(vectorized_text)
    return sentiment[0]


if __name__ == "__main__":
    import sys, os

    try:
        if len(sys.argv) != 2:
            raise Exception("U need to insert the DataSetTest Path.")
        if not sys.argv[1].endswith(".csv"):
            raise Exception(f"The file '{sys.argv[1]}' must be a CSV file.")

        if not os.path.exists(sys.argv[1]):
            raise Exception(f"The file '{sys.argv[1]}' does not exist.")
        with open(sys.argv[1], 'r') as file:
                content = file.read()
                text_list = content.split(',')
        DataFrame = pd.DataFrame({'text': text_list})
        DataFrame['sentiment'] = DataFrame['text'].apply(predict_sentiment)
        sentiments = DataFrame['sentiment']

        # Save the selected column to a CSV file
        sentiments.to_csv('sentiments.csv', index=False, header=True)
    except Exception as e:
        print(f"Exeption Error: {e}")

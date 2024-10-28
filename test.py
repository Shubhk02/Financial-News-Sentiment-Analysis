import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gradio as gr

# Load and preprocess the first dataset for training
train_data = pd.read_csv("C:/Users/nisha/Downloads/archive (1)/News_sentiment_Jan2017_to_Apr2021.csv")
train_data = train_data.drop(columns=['Unnamed: 5'])
train_data['sentiment'] = train_data['sentiment'].map({'POSITIVE': 1, 'NEGATIVE': 0})
train_data = train_data.dropna()

# Tokenization and Padding for training data
max_vocab_size = 10000
max_sequence_length = 100

tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(train_data['Title'])

train_sequences = tokenizer.texts_to_sequences(train_data['Title'])
train_padded_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length)

train_labels = train_data['sentiment'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(train_padded_sequences, train_labels, test_size=0.2, random_state=42)

# Define the LSTM model
embedding_dim = 128
lstm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
lstm_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Load the second dataset
second_data = pd.read_csv("C:/Users/nisha/Downloads/cnbc_headlines.csv")

# Remove "ET" timezone information from Time column, convert to datetime, and extract month and year
second_data['Time'] = second_data['Time'].str.replace("ET", "", regex=False).str.strip()
second_data['Date'] = pd.to_datetime(second_data['Time'], errors='coerce')  # Specify format if possible
second_data['Month_Year'] = second_data['Date'].dt.to_period('M')  # Extract month and year for aggregation

# Ensure no NaN values in 'Description' and convert to strings for processing
second_data['Description'] = second_data['Description'].fillna("").astype(str)

# Tokenize and pad the descriptions in the second dataset
second_sequences = tokenizer.texts_to_sequences(second_data['Description'])
second_padded_sequences = pad_sequences(second_sequences, maxlen=max_sequence_length)

# Predict sentiment on the second dataset
second_data['Sentiment_Score'] = lstm_model.predict(second_padded_sequences).flatten()

# Group by month and year to calculate the average sentiment
monthly_sentiment = second_data.groupby('Month_Year')['Sentiment_Score'].mean().reset_index()
monthly_sentiment['Month_Year'] = monthly_sentiment['Month_Year'].dt.to_timestamp()

# Function to predict sentiment of a single headline
def predict_sentiment(news_title):
    seq = tokenizer.texts_to_sequences([news_title])
    padded = pad_sequences(seq, maxlen=max_sequence_length)
    pred_prob = lstm_model.predict(padded)[0][0]
    pred_label = "Positive" if pred_prob > 0.5 else "Negative"
    confidence = np.round(pred_prob * 100, 2) if pred_prob > 0.5 else np.round((1 - pred_prob) * 100, 2)
    return f"Sentiment: {pred_label}, Confidence: {confidence}%"

# Function to plot the monthly average sentiment graph
def plot_sentiment_graph():
    plt.figure(figsize=(14, 7))
    plt.plot(monthly_sentiment['Month_Year'], monthly_sentiment['Sentiment_Score'], marker='o', color='b', linestyle='-')
    plt.title("Average Financial News Sentiment Score (2017-2020)")
    plt.xlabel("Month-Year")
    plt.ylabel("Average Sentiment Score")
    plt.grid()
    plt.tight_layout()
    plt.savefig("monthly_sentiment.png")  # Save plot as an image
    return "monthly_sentiment.png"  # Return path to saved image for Gradio

# Gradio Interface
with gr.Blocks() as interface:
    headline_input = gr.Textbox(lines=2, placeholder="Enter news headline here...", label="Financial News Headline")
    headline_output = gr.Textbox(label="Sentiment Prediction")
    graph_output = gr.Image(type="filepath", label="Monthly Average Sentiment")

    def process_input(news_title):
        sentiment, graph_path = predict_sentiment(news_title), plot_sentiment_graph()
        return sentiment, graph_path

    headline_input.change(process_input, inputs=headline_input, outputs=[headline_output, graph_output])

# Launch the Gradio interface
interface.launch()

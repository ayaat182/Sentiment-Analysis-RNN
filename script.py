import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

train_df = pd.read_csv("train_tweets.csv", encoding="utf-8")
test_df = pd.read_csv("test_tweets.csv", encoding="utf-8")

def clean_text(text):
    text = re.sub(r"@\w+", "", text) 
    text = re.sub(r"#\w+", "", text) 
    text = re.sub(r"http\S+", "", text)  
    text = re.sub(r"[^\w\s]", "", text)  
    return text.lower()

train_df["tweet"] = train_df["tweet"].astype(str).apply(clean_text)
test_df["tweet"] = test_df["tweet"].astype(str).apply(clean_text)

vocab_size = 10000
max_length = 100

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_df["tweet"])

X_train = pad_sequences(tokenizer.texts_to_sequences(train_df["tweet"]), maxlen=max_length)
y_train = np.array(train_df["label"])

X_test = pad_sequences(tokenizer.texts_to_sequences(test_df["tweet"]), maxlen=max_length)


model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(1, activation="sigmoid")
])

model.build(input_shape=(None, max_length))  
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()


history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

train_loss, train_acc = model.evaluate(X_train, y_train)
print(f"Training Accuracy: {train_acc:.4f}")

model.save("sentiment_model.h5")

predictions = model.predict(X_test)
test_df["sentiment"] = (predictions > 0.5).astype(int) 

if "label" in test_df.columns:
    y_test = np.array(test_df["label"])
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

test_df.to_csv("predicted_test_tweets.csv", index=False)


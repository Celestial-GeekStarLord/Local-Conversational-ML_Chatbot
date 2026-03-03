import tensorflow as tf
import pickle
import numpy as np

# Load model
model = tf.keras.models.load_model("dialog_model.keras")

with open("dialog_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

index_word = {v:k for k,v in tokenizer.word_index.items()}

print("Chatbot ready! type quit to exit")

while True:
    text = input("You: ").lower()
    if text == "quit":
        break

    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)

    pred = model.predict(padded, verbose=0)[0]   # take first batch

    response_words = []

    for timestep in pred:
        predicted_index = np.argmax(timestep)

        if predicted_index == 0:
            continue

        word = index_word.get(predicted_index)
        if word:
            response_words.append(word)

    response = " ".join(response_words).strip()

    if response == "":
        response = "I don't know what to say."

    print("Bot:", response)
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle

print("Loading dataset...")


# Load dialog dataset (TAB separated!)

inputs = []
responses = []

with open("dialogs.txt", encoding="utf-8") as f:
    for line in f:
        if "\t" in line:   # <-- FIXED (was comma before)
            inp, resp = line.strip().split("\t", 1)
            inputs.append(inp.lower())
            responses.append("<start> " + resp.lower() + " <end>")

print(f"Loaded {len(inputs)} dialog pairs.")


# Tokenizer

tokenizer = Tokenizer(filters='', oov_token="<OOV>")
tokenizer.fit_on_texts(inputs + responses)

input_seq = tokenizer.texts_to_sequences(inputs)
response_seq = tokenizer.texts_to_sequences(responses)

max_len = max(
    max(len(s) for s in input_seq),
    max(len(s) for s in response_seq)
)

input_seq = pad_sequences(input_seq, maxlen=max_len, padding="post")
response_seq = pad_sequences(response_seq, maxlen=max_len, padding="post")

vocab_size = len(tokenizer.word_index) + 1

print("Vocabulary size:", vocab_size)
print("Max sequence length:", max_len)


# Build Seq2Seq Model

model = Sequential([
    Embedding(vocab_size, 128, mask_zero=True),
    LSTM(256, return_sequences=True),
    Dense(vocab_size, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

model.summary()


# Prepare labels

# shape must be (samples, timesteps, 1)
y = np.expand_dims(response_seq, -1)


# Train

print("Training model...")
model.fit(
    input_seq,
    y,
    epochs=200,
    batch_size=32
)


# Save model + tokenizer

model.save("dialog_model.keras")

with open("dialog_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("max_len.pkl", "wb") as f:
    pickle.dump(max_len, f)

print("✅ Dialog model trained and saved!")
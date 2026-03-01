import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
with open('intents.json') as file:
    data = json.load(file)

sentences = []
labels = []
all_labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(intent['tag'])

    responses.append(intent['responses'])

    if intent['tag'] not in all_labels:
        all_labels.append(intent['tag'])

# Encode labels
lbl_encoder = LabelEncoder()
lbl_encoder.fit(labels)
labels = lbl_encoder.transform(labels)

# Tokenization
vocab_size = 1000
max_len = 20
embedding_dim = 16

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_len)

# Build model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(len(all_labels), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(padded, np.array(labels), epochs=300)

# Save model + tokenizer + encoder
model.save("chat_model.keras")

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(lbl_encoder, f)

print("Training complete!")
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Bidirectional
from keras.optimizers import Adam, AdamW
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
import pickle

data = pd.read_csv('clean_wordsNO.csv', names=['word', 'transcription'])

char_to_index = {}
index_to_char = {}

def create_mapping(data):
    global char_to_index, index_to_char
    unique_chars = set(''.join(data['word']) + ''.join(data['transcription']))
    char_to_index = {char: idx + 1 for idx, char in enumerate(unique_chars)}
    index_to_char = {idx + 1: char for idx, char in enumerate(unique_chars)}
    char_to_index['<PAD>'] = 0
    index_to_char[0] = '<PAD>'

create_mapping(data)

def encode_sequence(sequence, max_length):
    return [char_to_index[char] for char in sequence] + [char_to_index['<PAD>']] * (max_length - len(sequence))

max_word_length = max(data['word'].apply(len))
max_transcription_length = max(data['transcription'].apply(len))
max_sequence_length = max(max_word_length, max_transcription_length)

X = np.array([encode_sequence(word, max_sequence_length) for word in data['word']])
y = np.array([encode_sequence(transcription, max_sequence_length) for transcription in data['transcription']])

y = tf.keras.utils.to_categorical(y, num_classes=len(char_to_index))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    Embedding(input_dim=len(char_to_index), output_dim=128, input_length=max_sequence_length),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=True)),
    TimeDistributed(Dense(256, activation='relu')),
    Dropout(0.3),
    TimeDistributed(Dense(len(char_to_index), activation='softmax'))
])

model.compile(optimizer=AdamW(learning_rate=0.005), loss='categorical_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=5,          
    restore_best_weights=True  
)


model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=15,
    epochs=50,  
    callbacks=[early_stopping] 
)

def predict_transcription(word):
    encoded_word = encode_sequence(word, max_sequence_length)
    encoded_word = np.expand_dims(encoded_word, axis=0)
    predicted_indices = model.predict(encoded_word)
    predicted_indices = np.argmax(predicted_indices, axis=-1)[0]
    transcription = ''.join(index_to_char[idx] for idx in predicted_indices if idx != 0)
    return transcription.strip()

# сохранение 
model.save('work1.keras')
with open('work1.pkl', 'wb') as f:
    pickle.dump({'char_to_index': char_to_index, 'index_to_char': index_to_char}, f)


example_word = "пример"
predicted_transcription = predict_transcription(example_word)
print(f"'{example_word}' -> '{predicted_transcription}'")
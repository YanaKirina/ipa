import numpy as np
import pickle
from tensorflow.keras.models import load_model
import re

example_sentence = """Добрый день, извините, Вы не могли бы мне помочь? Кажется, я заблудился. 
Интернет перестал ловить и не загружается карта. Я не знаю, где сейчас нахожусь.

Добрый день, не волнуйтесь, всё хорошо, я Вам помогу. Мы находимся на Ворошиловском проспекте. А Вам куда нужно?

Мне нужно добраться до студенческого общежития номер десять Донского государственного технического университета.

Это общежитие находится возле главного корпуса ДГТУ на площади Гагарина один. Вам нужно идти прямо, никуда не сворачивать. Прямо перед Вами будет площадь Гагарина с фонтаном и здание университета. Обогните главный корпус слева и увидите высокое круглое здание. Это и есть общежитие номер десять.

Большое спасибо за помощь!"""

model = load_model('work1.keras')


with open('work1.pkl', 'rb') as f:
    data = pickle.load(f)
    char_to_index = data['char_to_index']
    index_to_char = data['index_to_char']


def clean_text(text):

    return re.sub(r'[^\w\s]', '', text)


def encode_sequence(word, max_sequence_length):
    encoded = [char_to_index.get(char, 0) for char in word]  
    if len(encoded) < max_sequence_length:
        encoded += [0] * (max_sequence_length - len(encoded))  
    return np.array(encoded[:max_sequence_length])  

def predict_transcription(word, max_sequence_length=50):
    
    encoded_word = encode_sequence(word, max_sequence_length)
    encoded_word = np.expand_dims(encoded_word, axis=0)  

    predicted_indices = model.predict(encoded_word)
    predicted_indices = np.argmax(predicted_indices, axis=-1)[0]

    transcription = ''.join(index_to_char[idx] for idx in predicted_indices if idx != 0)
    return transcription.strip()


def predict_sentence_transcription(sentence, max_sequence_length=50):
    
    cleaned_sentence = clean_text(sentence)
    words = cleaned_sentence.split()  


    predicted_words = [predict_transcription(word, max_sequence_length) for word in words]
    

    return ' '.join(predicted_words)

predicted_sentence = predict_sentence_transcription(example_sentence)
print(f"'{example_sentence}' -> '{predicted_sentence}'")

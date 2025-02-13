import numpy as np
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import pickle

def train_and_save_model():
    # Загрузка данных
    max_words = 10000
    max_len = 500
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words)
    
    # Создание и обучение токенизатора
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts([" ".join([str(word) for word in text]) for text in x_train])
    
    # Сохранение токенизатора
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Преобразование последовательностей
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    
    # Преобразование меток
    num_classes = 46
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # Создание модели
    model = Sequential([
        Dense(512, input_shape=(max_len,), activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    
    # Обучение модели
    model.fit(x_train, y_train,
              batch_size=32,
              epochs=5,
              validation_split=0.1)
    
    # Сохранение модели
    model.save('news_classifier_model.h5')
    
    # Сохранение параметров модели
    model_params = {
        'max_words': max_words,
        'max_len': max_len,
        'num_classes': num_classes
    }
    
    with open('model_params.pickle', 'wb') as handle:
        pickle.dump(model_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    train_and_save_model()
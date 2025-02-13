import tensorflow as tf
import numpy as np
import json
import librosa
from tensorflow.keras import layers, models
from typing import List, Dict, Tuple

# Константы для обработки аудио
SAMPLE_RATE = 16000
FRAME_LENGTH = 256
FRAME_STEP = 160
FFT_LENGTH = 384

# Параметры модели
OUTPUT_DIM = 100  # Размер словаря (нужно настроить под ваш датасет)

class AudioPreprocessor:
    def __init__(self):
        # Создаем токенизатор для русского текста
        self.char_to_num = {}
        self.num_to_char = {}
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        # Создаем словарь для русских букв и специальных токенов
        chars = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя '
        self.char_to_num = {char: idx for idx, char in enumerate(chars)}
        self.char_to_num['<blank>'] = len(self.char_to_num)
        self.num_to_char = {v: k for k, v in self.char_to_num.items()}
        
    def process_audio(self, audio_path: str) -> np.ndarray:
        # Загрузка и обработка аудио
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Извлечение спектрограммы
        spectrogram = tf.signal.stft(
            audio,
            frame_length=FRAME_LENGTH,
            frame_step=FRAME_STEP,
            fft_length=FFT_LENGTH
        )
        
        # Преобразование в мел-спектрограмму
        mel_spectrogram = tf.abs(spectrogram)
        mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        
        return mel_spectrogram.numpy()

    def process_text(self, text: str) -> np.ndarray:
        # Преобразование текста в последовательность чисел
        return np.array([self.char_to_num.get(c, self.char_to_num['<blank>']) 
                        for c in text.lower()])

class DeepSpeechModel:
    def __init__(self):
        self.model = self._build_model()
    
    def _build_model(self) -> models.Model:
        # Входной слой
        input_spectrogram = layers.Input(shape=(None, FFT_LENGTH // 2 + 1))
        
        # CNN слои
        x = layers.Reshape((-1, FFT_LENGTH // 2 + 1, 1))(input_spectrogram)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        # Преобразование для RNN
        x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
        
        # Bidirectional RNN слои
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(0.5)(x)
        
        # Выходной слой
        output = layers.Dense(OUTPUT_DIM + 1)(x)  # +1 для blank token
        
        return models.Model(input_spectrogram, output)
    
    def compile_model(self):
        # Компиляция модели с CTC loss
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=self.ctc_loss
        )
    
    def ctc_loss(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        return tf.keras.backend.ctc_batch_cost(
            y_true, y_pred, input_length, label_length
        )

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_paths: List[Dict], preprocessor: AudioPreprocessor, 
                 batch_size: int = 32):
        self.data_paths = data_paths
        self.preprocessor = preprocessor
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.data_paths) // self.batch_size
    
    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        batch_data = self.data_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X = []
        y = []
        
        for item in batch_data:
            # Обработка аудио
            audio_features = self.preprocessor.process_audio(item['audio_filepath'])
            X.append(audio_features)
            
            # Обработка текста
            text_features = self.preprocessor.process_text(item['text'])
            y.append(text_features)
        
        # Паддинг последовательностей
        X = tf.keras.preprocessing.sequence.pad_sequences(
            X, padding='post', dtype='float32'
        )
        y = tf.keras.preprocessing.sequence.pad_sequences(
            y, padding='post', dtype='int32'
        )
        
        return X, y

def load_dataset(json_path: str) -> List[Dict]:
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data
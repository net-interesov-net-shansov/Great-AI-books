from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pickle
import numpy as np

app = Flask(__name__)

# Загрузка сохраненных файлов
model = load_model('news_classifier_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('model_params.pickle', 'rb') as handle:
    model_params = pickle.load(handle)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']
        
        # Предобработка текста
        sequence = tokenizer.texts_to_sequences([text])
        padded = sequence.pad_sequences(sequence, maxlen=model_params['max_len'])
        
        # Предсказание
        prediction = model.predict(padded)
        category = np.argmax(prediction[0])
        confidence = float(prediction[0][category])
        
        return jsonify({
            'category': int(category),
            'confidence': confidence,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'status': 'success',
        **model_params
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
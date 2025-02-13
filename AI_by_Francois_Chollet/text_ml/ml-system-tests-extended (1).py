# tests/test_integration.py
import pytest
import requests
import time
import numpy as np
from locust import HttpUser, task, between
from tensorflow.keras.models import load_model
import json
import pickle

class TestIntegration:
    """Интеграционные тесты для проверки взаимодействия компонентов"""
    
    @pytest.fixture(scope="class")
    def setup_system(self):
        """Подготовка системы для интеграционных тестов"""
        # Загрузка необходимых компонентов
        model = load_model('news_classifier_model.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer

    def test_end_to_end_flow(self, setup_system):
        """Тест полного процесса от предобработки до получения результата"""
        model, tokenizer = setup_system
        
        # Тестовый текст
        test_text = "This is a test news article about technology and innovation"
        
        # Предобработка
        sequences = tokenizer.texts_to_sequences([test_text])
        assert len(sequences) > 0, "Ошибка токенизации"
        
        # Предсказание через модель напрямую
        padded_sequence = sequence.pad_sequences(sequences, maxlen=500)
        direct_prediction = model.predict(padded_sequence)
        
        # Предсказание через API
        api_response = requests.post(
            "http://localhost:5000/predict",
            json={"text": test_text}
        )
        api_prediction = api_response.json()
        
        # Проверка согласованности результатов
        assert np.argmax(direct_prediction[0]) == api_prediction['category']

# tests/test_performance.py
class TestPerformance:
    """Тесты производительности системы"""
    
    def test_model_inference_time(self):
        """Тест времени инференса модели"""
        model = load_model('news_classifier_model.h5')
        dummy_input = np.zeros((1, 500))
        
        # Прогрев модели
        model.predict(dummy_input)
        
        # Измерение времени выполнения
        times = []
        for _ in range(100):
            start_time = time.time()
            model.predict(dummy_input)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        assert avg_time < 0.1, f"Среднее время инференса ({avg_time:.3f}с) превышает порог"

    def test_api_response_time(self):
        """Тест времени отклика API"""
        test_text = "Sample news text for testing"
        times = []
        
        for _ in range(50):
            start_time = time.time()
            response = requests.post(
                "http://localhost:5000/predict",
                json={"text": test_text}
            )
            end_time = time.time()
            times.append(end_time - start_time)
            
            assert response.status_code == 200
        
        avg_time = np.mean(times)
        assert avg_time < 0.5, f"Среднее время ответа API ({avg_time:.3f}с) превышает порог"

# tests/test_security.py
class TestSecurity:
    """Тесты безопасности системы"""
    
    def test_sql_injection(self):
        """Тест на SQL-инъекции"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'); DROP TABLE messages; --"
        ]
        
        for text in malicious_inputs:
            response = requests.post(
                "http://localhost:5000/predict",
                json={"text": text}
            )
            assert response.status_code in [200, 400, 500]
            assert 'error' in response.json() or 'category' in response.json()

    def test_xss_prevention(self):
        """Тест на XSS-атаки"""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "<img src='x' onerror='alert(1)'>",
            "javascript:alert('xss')"
        ]
        
        for text in malicious_inputs:
            response = requests.post(
                "http://localhost:5000/predict",
                json={"text": text}
            )
            assert response.status_code in [200, 400, 500]

    def test_input_validation(self):
        """Тест валидации входных данных"""
        invalid_inputs = [
            {"text": "a" * 1000000},  # слишком длинный текст
            {"text": None},  # null
            {"text": ["array", "of", "strings"]},  # неправильный тип
            {"wrong_key": "text"},  # неправильный ключ
        ]
        
        for data in invalid_inputs:
            response = requests.post(
                "http://localhost:5000/predict",
                json=data
            )
            assert response.status_code in [400, 500]

# tests/test_load.py
class LoadTest(HttpUser):
    """Нагрузочное тестирование с использованием Locust"""
    wait_time = between(1, 2)
    
    @task
    def predict_news(self):
        """Тест эндпоинта предсказания под нагрузкой"""
        self.client.post(
            "/predict",
            json={"text": "Sample news text for load testing"}
        )

    @task
    def get_model_info(self):
        """Тест эндпоинта информации о модели под нагрузкой"""
        self.client.get("/model-info")

# tests/test_error_handling.py
class TestErrorHandling:
    """Тесты обработки ошибок"""
    
    def test_model_error_handling(self):
        """Тест обработки ошибок модели"""
        with pytest.raises(Exception):
            # Попытка загрузить несуществующую модель
            load_model('nonexistent_model.h5')
    
    def test_tokenizer_error_handling(self):
        """Тест обработки ошибок токенизатора"""
        with pytest.raises(Exception):
            # Попытка загрузить несуществующий токенизатор
            with open('nonexistent_tokenizer.pickle', 'rb') as handle:
                pickle.load(handle)

    @pytest.mark.parametrize("bad_request", [
        {},  # пустой запрос
        {"wrong_key": "value"},  # неправильный ключ
        {"text": None},  # null значение
        {"text": ""},  # пустой текст
    ])
    def test_api_error_handling(self, bad_request):
        """Тест обработки ошибок API"""
        response = requests.post(
            "http://localhost:5000/predict",
            json=bad_request
        )
        assert response.status_code in [400, 500]
        assert 'error' in response.json()

# tests/conftest.py
import pytest
import os
import logging

@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Настройка логирования для тестов"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='test_logs.log'
    )

@pytest.fixture(scope="session")
def test_data():
    """Фикстура с тестовыми данными"""
    return {
        "short_text": "This is a short news article",
        "long_text": "This is a very long news article " * 100,
        "special_chars": "!@#$%^&*()",
        "numbers": "123456789",
        "mixed": "Text with numbers 123 and special chars !@#"
    }
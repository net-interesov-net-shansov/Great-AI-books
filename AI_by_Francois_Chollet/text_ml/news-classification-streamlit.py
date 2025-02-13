import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Классификатор новостей",
    page_icon="📰",
    layout="wide"
)

# Функции для работы с API
def get_model_info():
    try:
        return requests.get('http://localhost:5000/model-info').json()
    except:
        return None

def classify_text(text):
    try:
        return requests.post(
            'http://localhost:5000/predict',
            json={'text': text}
        ).json()
    except:
        return None

# Интерфейс
st.title("📰 Классификатор новостей")

# Боковая панель
with st.sidebar:
    st.header("ℹ️ Информация о модели")
    model_info = get_model_info()
    if model_info and model_info['status'] == 'success':
        st.success("API подключено")
        for key, value in model_info.items():
            if key != 'status':
                st.write(f"{key}: {value}")
    else:
        st.error("❌ Ошибка подключения к API")

# Основной интерфейс
text_input = st.text_area(
    "Введите текст новости:",
    height=200
)

if st.button("Классифицировать"):
    if text_input:
        with st.spinner("Обрабатываем..."):
            result = classify_text(text_input)
            if result and result['status'] == 'success':
                st.success(f"Категория: {result['category']}")
                
                # Визуализация уверенности
                fig = px.bar(
                    pd.DataFrame({
                        'Уверенность': [result['confidence']],
                        'Предсказание': ['Результат']
                    }),
                    x='Предсказание',
                    y='Уверенность',
                    range_y=[0, 1]
                )
                st.plotly_chart(fig)
            else:
                st.error("Ошибка при классификации")
    else:
        st.warning("Введите текст для классификации")
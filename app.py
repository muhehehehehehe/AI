import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Загрузка модели и векторизатора ---
try:
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    st.success("Модель и векторизатор успешно загружены.")
except FileNotFoundError:
    st.error("Ошибка: Файлы модели ('sentiment_model.pkl') или векторизатора ('tfidf_vectorizer.pkl') не найдены.")
    st.stop() # Останавливаем приложение
except Exception as e:
    st.error(f"Произошла ошибка при загрузке модели или векторизатора: {e}")
    st.stop()

# --- Функция для предсказания ---
def predict_sentiment(text):
    if not text.strip():
        return "Введите текст для анализа."
    try:
        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)
        sentiment = "Позитивный" if prediction[0] == 1 else "Негативный"
        return f"Тональность: {sentiment}"
    except Exception as e:
        return f"Ошибка анализа: {e}"

# --- Интерфейс Streamlit ---
st.title("Анализатор тональности текста")
st.write("Введите текст ниже, чтобы определить его тональность.")

user_input = st.text_area("Ваш текст:")

if st.button("Проанализировать"):
    if user_input:
        result = predict_sentiment(user_input)
        st.write(result)
    else:
        st.warning("Пожалуйста, введите текст.")

# --- Запуск приложения ---
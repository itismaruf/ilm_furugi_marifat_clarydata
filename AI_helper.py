import os
import requests
import streamlit as st
from dotenv import load_dotenv

# === Загрузка переменных окружения ===
load_dotenv()
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ API-ключ не найден. Убедитесь, что он задан в .env или secrets.")

# === История диалога и контекст проекта ===
chat_history = [
    {
        "role": "system",
        "content": "Ты помощник на русском языке. Отвечай чётко, по делу, кратко при необходимости и точно."
    }
]
context = {}

def update_context(key, value):
    """Добавление или обновление глобального контекста проекта."""
    context[key] = value

# === Универсальная функция с учётом контекста ===
def get_chatgpt_response(prompt, model="mistralai/devstral-small-2505:free"):
    """Запрос в ИИ с подстановкой глобального контекста."""
    if not prompt or not isinstance(prompt, str):
        return "❌ Пустой или некорректный запрос."

    context_info = "\n".join([f"{k}: {v}" for k, v in context.items()])
    full_prompt = f"Контекст:\n{context_info}\n\n{prompt}, не когда не давай код, просто отвечай на то что просять конкретно, коротко если надо!"

    chat_history.append({"role": "user", "content": full_prompt})

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": chat_history},
            timeout=20
        )
        data = resp.json()
        if "choices" in data and data["choices"]:
            reply = data["choices"][0]["message"]["content"]
            chat_history.append({"role": "assistant", "content": reply})
            return reply
        return f"❌ Пустой ответ от API: {data}"
    except Exception as e:
        return f"❌ Ошибка при запросе: {e}"

# === Чат без автоконтекста (раздел "Чат") ===
def chat_only(message, model="mistralai/devstral-small-2505:free"):
    """Отправка сообщения в ИИ без контекста проекта, но с сохранением истории."""
    if not message or not isinstance(message, str):
        return "❌ Пустой или некорректный запрос."

    chat_history.append({"role": "user", "content": message})

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": chat_history},
            timeout=20
        )
        data = resp.json()
        if "choices" in data and data["choices"]:
            reply = data["choices"][0]["message"]["content"]
            chat_history.append({"role": "assistant", "content": reply})
            return reply
        return f"❌ Пустой ответ от API: {data}"
    except Exception as e:
        return f"❌ Ошибка при запросе: {e}"
    

def notify_ai_dataset_structure(df, get_fn=get_chatgpt_response):
    """
    Отправляет в ИИ краткую информацию о датасете.
    """
    try:
        # Краткое описание данных
        info = (
            f"Датасет: {df.shape[0]} строк, {df.shape[1]} столбцов. "
            f"Колонки: {', '.join(df.columns)}. "
            f"Типы: {', '.join(f'{c} ({str(df[c].dtype)})' for c in df.columns)}."
        )

        # Отправляем ИИ
        get_fn(f"[DATASET STRUCTURE]\n{info}")

        # Можно сохранить в контекст, если используешь update_context
        try:
            update_context("dataset_structure", info)
        except:
            pass

        return "✅ ИИ в успешно подключился."
    except Exception as e:
        return f"Ошибка при отправке данных в ИИ: {e}"



# === Отправка корреляций ===
def send_correlation_to_ai(df):
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return "📉 Недостаточно числовых переменных для корреляции."

    corr = numeric_df.corr().abs().unstack().sort_values(ascending=False)
    corr = corr[corr < 1].drop_duplicates()
    top_corr = corr.head(10)

    formatted_corr = "\n".join([f"{a} и {b}: корреляция {v:.2f}" for (a, b), v in top_corr.items()])
    prompt = f"Топ-10 корреляций между переменными:\n{formatted_corr}"
    return get_chatgpt_response(prompt)

# === Отправка сводной таблицы ===
def send_pivot_to_ai(pivot_df, index_col, value_col, agg_func):
    try:
        if pivot_df is None:
            return "❌ Невозможно отправить пустую сводную таблицу."

        top_rows = pivot_df.head(10).to_dict(orient="records")
        formatted = "\n".join(map(str, top_rows))
        prompt = f"Сводная таблица по {index_col}, агрегируя {value_col} методом {agg_func}:\n{formatted}"
        return get_chatgpt_response(prompt)
    except Exception as e:
        return f"❌ Ошибка при отправке сводной таблицы: {e}"
# =====  ->СТР Загрузка данных =====
import pandas as pd
import streamlit as st
import re
from typing import Callable

from AI_helper import update_context
# Функция для загрузки и предварительной обработки данных 

def looks_like_number(s: str) -> bool:
    s = s.strip().replace(",", ".")
    return bool(re.match(r"^-?\d+(\.\d+)?$", s))

def load_data(uploaded_file) -> pd.DataFrame:
    """
    Читает CSV/XLSX/XLS, приводит object-столбцы к числам, 
    сохраняет лог преобразований и имя файла в st.session_state.
    """
    # Сохраняем оригинальное имя файла
    st.session_state["original_filename"] = uploaded_file.name  

    fname = uploaded_file.name.lower()
    if fname.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif fname.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        st.error("Неподдерживаемый формат файла")
        raise ValueError

    conversion_log = []
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "object":
            df[col] = df[col].astype(str).str.strip().str.replace(",", ".")
            mask = df[col].apply(looks_like_number)
            rate = mask.mean()
            if rate > 0.9:
                try:
                    df[col] = pd.to_numeric(df[col], errors="raise")
                    conversion_log.append(f"{col}: object → float ({rate:.0%})")
                except:
                    conversion_log.append(f"{col}: оставлен как текст")
            else:
                conversion_log.append(f"{col}: текст ({rate:.0%} чисел)")
        else:
            conversion_log.append(f"{col}: {dtype}")

    st.session_state["conversion_log"] = conversion_log
    return df


def get_base_info(df: pd.DataFrame) -> dict:
    """Возвращает базовую статистику по DataFrame."""
    return {
        "Строк": df.shape[0],
        "Столбцов": df.shape[1],
        "Пропусков": int(df.isnull().sum().sum()),
        "Дубликатов": int(df.duplicated().sum()),
        "Числовых": len(df.select_dtypes("number").columns),
        "Категориальных": len(df.select_dtypes("object").columns),
    }


def display_preview(df: pd.DataFrame, n: int = 5):
    """Показывает первые n строк и скрытую описательную статистику с пояснениями."""
    st.markdown(f"### 🧾 Пример данных (первые {n} строк):")
    st.dataframe(df.head(n), use_container_width=True)

    with st.expander("📑 Описательная статистика (describe)", expanded=False):
        # формируем таблицу describe
        desc = df.describe(include="all").round(3).transpose()
        desc.index.name = "Признак"
        st.dataframe(desc, use_container_width=True)

        # краткое пояснение к отсутствующим полям
        st.markdown(
            "Некоторые ячейки могут быть пустыми (None) —\n"
            "- для числовых признаков поля `unique`, `top` и `freq` не вычисляются;\n"
            "- для категориальных полей отсутствуют `min`, `25%`, `50%`, `75%`, `max`, так как они неприменимы.\n"
            "Это нормально и указывает на то, что данный показатель просто не рассчитан для такого типа данных."
        )


def display_base_info(base_info: dict):
    """Красиво отображает ключевые метрики DataFrame."""
    st.subheader("📊 Базовая информация")
    cols = st.columns(len(base_info))
    for col, (label, value) in zip(cols, base_info.items()):
        col.metric(label=label, value=value)


def interpret_with_ai(
    data_summary: str,
    user_desc: str,
    df: pd.DataFrame,
    get_ai_fn: Callable[[str], str]
) -> None:
    """
    Формирует подробный prompt на основе переданной сводки, целей и образца данных,
    отправляет его в ИИ и красиво выводит полученную интерпретацию.
    """
    # 1. Обновляем глобальный контекст ИИ
    update_context("data_summary", data_summary)
    update_context("user_goal", user_desc)

    # 2. Собираем «первые 5 строк» и статистику пропусков
    sample_csv = df.head(5).to_csv(index=False)
    missing = df.isna().sum().to_dict()

    # 3. Формируем подробный prompt
    prompt = (
        f"У тебя есть следующие данные:\n"
        f"{data_summary}\n\n"
        f"Пропуски по столбцам: {missing}\n\n"
        f"Первые 5 строк (CSV):\n{sample_csv}\n\n"
        f"Цель анализа: {user_desc}\n\n"
        "Просто это учитовай, на всякие случи\n"
        "Ответь так чтобы не скучно было читаь, и не длинно!" \
        "Консентрируйся на что что просят!"
    )

    # 4. Отправляем запрос и получаем ответ
    try:
        with st.spinner("✨ Генерируем интерпретацию от ИИ..."):
            ai_response = get_ai_fn(prompt)
    except Exception as e:
        st.error(f"Ошибка при получении интерпретации от ИИ: {e}", icon="🚫")
        return ai_response

    # 5. Выводим ответ
    st.markdown("---")
    st.subheader("💡 Интерпретация от ИИ")
    st.info(ai_response, icon="🤖")

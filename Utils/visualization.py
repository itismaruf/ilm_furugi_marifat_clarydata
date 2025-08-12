from AI_helper import get_chatgpt_response

import streamlit as st
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go 
from typing import Any, List, Optional, Tuple, Dict


# ==== Визуализация ====
def safe_selectbox(
    label: str,
    options: List[Any],
    index: int = 0,
    default: Any = None,
    **kwargs
) -> Any:
    """
    Обёртка для st.selectbox:
    - если options пуст, выводит предупреждение и возвращает default.
    - если index вне диапазона, подставляет 0.
    """
    if not options:
        st.warning(f"Нет доступных вариантов для «{label}».")
        return default
    idx = index if 0 <= index < len(options) else 0
    return st.selectbox(label, options, index=idx, **kwargs)


def apply_numeric_filters(
    df: pd.DataFrame,
    numeric_filters: Optional[Dict[str, Tuple[float, float]]]
) -> pd.DataFrame:
    """
    Фильтрует числовые колонки по диапазонам, заданным пользователем.
    Пропускает фильтры, если min == max или колонка не числовая.
    Ошибки не прерывают выполнение.
    """
    if not numeric_filters:
        return df

    try:
        for col, (min_val, max_val) in numeric_filters.items():
            if col not in df:
                st.warning(f"Колонка «{col}» не найдена в данных.")
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                st.warning(f"Колонка «{col}» не является числовой.")
                continue

            if min_val == max_val:
                st.info(f"Пропущен фильтр по «{col}»: диапазон ({min_val}, {max_val}) не имеет смысла.")
                continue

            df = df[df[col].between(min_val, max_val)]

    except Exception as e:
        st.warning(f"Ошибка при применении числовых фильтров: {e}")

    return df



# === Вспомогательная функция для фильтрации топ-N ===
def filter_top_n(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    top_values = df[col].value_counts().nlargest(n).index
    return df[df[col].isin(top_values)]


# === Вспомогательная функция для определения временного признака ===
def is_temporal(column_name: str, series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    keywords = ["date", "time", "year", "month"]
    return any(key in column_name.lower() for key in keywords)


# === Улучшенная ручная визуализация ===
def generate_manual_chart(
    df: pd.DataFrame,
    x: str,
    y: Optional[str] = None,
    chart_type: str = "Гистограмма",
    top_n: Optional[int] = None
) -> Optional[go.Figure]:
    try:
        if x not in df.columns or (y and y not in df.columns):
            st.warning("Выбранные столбцы отсутствуют в данных.")
            return None

        if df[x].nunique() == len(df):
            st.warning("X — уникальный идентификатор, визуализировать бессмысленно.")
            return None

        if top_n:
            df = filter_top_n(df, x, top_n)
            if y and not pd.api.types.is_numeric_dtype(df[y]):
                df = filter_top_n(df, y, top_n)

        x_num = pd.api.types.is_numeric_dtype(df[x])
        y_num = pd.api.types.is_numeric_dtype(df[y]) if y else False
        is_time_x = is_temporal(x, df[x])

        match chart_type:
            case "Гистограмма":
                return px.histogram(df, x=x, nbins=30 if x_num else None)

            case "Круговая диаграмма":
                if df[x].nunique() > 10:
                    st.warning("Слишком много категорий для круговой диаграммы (>10).")
                    return None
                counts = df[x].value_counts().nlargest(top_n) if top_n else df[x].value_counts()
                return px.pie(names=counts.index, values=counts.values)

            case "Точечный график":
                if not y:
                    st.warning("Укажите Y для точечного графика.")
                    return None
                return px.scatter(df, x=x, y=y, color=(y if not y_num else None))

            case "Boxplot":
                if not y:
                    st.warning("Укажите Y для boxplot.")
                    return None
                return px.box(df, x=x, y=y, color=(x if not x_num else None))

            case "Bar-график":
                if not y:
                    st.warning("Укажите Y для bar-графика.")
                    return None
                return px.bar(df, x=x, y=y, color=(x if not x_num else None))

            case "Лайнплот":
                if not is_time_x:
                    st.warning("Лайнплот применяется только к временным данным.")
                    return None
                if not y:
                    st.warning("Укажите Y для лайнплота.")
                    return None
                return px.line(df, x=x, y=y)

            case _:
                st.warning(f"Неизвестный тип графика: {chart_type}")
                return None

    except Exception as e:
        st.warning(f"Невозможно построить график «{chart_type}»: {e}")
        return None
    


# === Улучшенная авто-визуализация ===
def generate_auto_chart(
    df: pd.DataFrame,
    x: str,
    y: Optional[str] = None,
    top_n: Optional[int] = None
) -> Optional[go.Figure]:
    try:
        if x not in df.columns or (y and y not in df.columns):
            st.warning("Выбранные столбцы отсутствуют в данных.")
            return None

        if df[x].nunique() == len(df):
            st.warning("X — уникальный идентификатор, визуализировать бессмысленно.")
            return None

        if top_n:
            df = filter_top_n(df, x, top_n)
            if y and not pd.api.types.is_numeric_dtype(df[y]):
                df = filter_top_n(df, y, top_n)

        x_num = pd.api.types.is_numeric_dtype(df[x])
        y_num = pd.api.types.is_numeric_dtype(df[y]) if y else False
        is_time_x = is_temporal(x, df[x])

        if y:
            if x_num and y_num:
                return px.line(df, x=x, y=y) if is_time_x else px.scatter(df, x=x, y=y)

            if not x_num and y_num:
                if df[x].nunique() <= 5:
                    agg = df.groupby(x)[y].mean().reset_index()
                    return px.bar(agg, x=x, y=y)
                else:
                    return px.box(df, x=x, y=y)

            if not x_num and not y_num:
                return px.histogram(df, x=x, color=y, barmode="group")

            return px.bar(df, x=x, y=y)

        else:
            if x_num:
                return px.histogram(df, x=x)
            if df[x].nunique() <= 10:
                counts = df[x].value_counts()
                return px.pie(names=counts.index, values=counts.values)
            return px.histogram(df, x=x)

    except Exception as e:
        st.warning(f"Невозможно построить автоматическую визуализацию: {e}")
        return None
    


# === Обёртка для отображения графика с фильтрами и обработкой ошибок ===
def plot_data_visualizations(
    df: pd.DataFrame,
    x: str,
    y: Optional[str] = None,
    top_n: Optional[int] = None,
    numeric_filters: Optional[Dict[str, Tuple[float, float]]] = None,
    chart_type: str = "Автоматически"
) -> None:
    if x not in df.columns:
        st.warning("Не выбрана переменная X или она отсутствует в данных.")
        return
    if y and y not in df.columns:
        st.warning("Переменная Y отсутствует в данных.")
        return
    if x == y:
        st.warning("Переменные X и Y не должны совпадать.")
        return

    df_filtered = apply_numeric_filters(df, numeric_filters or {})
    fig = (
        generate_auto_chart(df_filtered, x, y, top_n)
        if chart_type == "Автоматически"
        else generate_manual_chart(df_filtered, x, y, chart_type, top_n)
    )

    if fig is None:
        st.info("Визуализация недоступна для выбранных параметров.")
    else:
        st.plotly_chart(fig, use_container_width=True)



def suggest_visualization_combinations(df_info: str) -> str:
    try:
        prompt = (
            "Предложи 2–3 интересные комбинации для визуализации (X и Y), и коротко скажи почему,"
            "чтобы выявить закономерности. Кратко, по одной на строку." \
            "Пример: X - ... а Y - ..."
        )
        return get_chatgpt_response(prompt)
    except Exception as e:
        return f"Не удалось получить рекомендации: {e}"
    

# ==== Корреляции ====
def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Строит тепловую карту корреляций для числовых переменных.
    """
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] < 2:
        st.warning("Недостаточно числовых переменных для корреляционного анализа.")
        return None

    corr = numeric_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="🔗 Тепловая карта корреляций"
    )
    return fig

# ==== Pivot ====
def generate_pivot_table(df: pd.DataFrame, index_col: str, value_col: str, agg_func: str = "mean"):
    """
    Строит сводную таблицу по index_col с агрегированием value_col.
    Поддерживает mean, sum, count.
    """
    if index_col not in df.columns or value_col not in df.columns:
        st.warning("Выберите корректные переменные для сводной таблицы.")
        return None

    # Группируем и агрегистрируем данные в зависимости от функции агрегации
    if agg_func == "mean":
        pivot = df.groupby(index_col, as_index=False)[value_col].mean()
    elif agg_func == "sum":
        pivot = df.groupby(index_col, as_index=False)[value_col].sum()
    elif agg_func == "count":
        pivot = df.groupby(index_col, as_index=False)[value_col].count()
    else:
        st.warning("Неизвестная агрегирующая функция.")
        return None

    # Проверка, есть ли два столбца после группировки
    if len(pivot.columns) == 2:  # Убедимся, что есть индекс и результат агрегации
        pivot.columns = [index_col, f"{agg_func}({value_col})"]
    else:
        st.warning(f"Не удалось создать сводную таблицу с выбранными параметрами.")
        return None

    return pivot
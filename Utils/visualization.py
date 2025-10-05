import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Optional, Tuple, Dict
from AI_helper import chat_with_context
import time


# eda_ui_blocks.py
# === Внутренние зависимости ===
def apply_numeric_filters(
    df: pd.DataFrame,
    numeric_filters: Optional[Dict[str, Tuple[float, float]]]
) -> pd.DataFrame:
    """Фильтрует числовые колонки по диапазонам."""
    if not numeric_filters:
        return df
    try:
        for col, (min_val, max_val) in numeric_filters.items():
            if col not in df or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            if min_val != max_val:
                df = df[df[col].between(min_val, max_val)]
    except Exception as e:
        st.warning(f"Ошибка при применении фильтров: {e}")
    return df

def is_temporal(column_name: str, series: pd.Series) -> bool:
    """Определяет, является ли колонка временной."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    keywords = ["date", "time", "year", "month"]
    return any(key in column_name.lower() for key in keywords)

# === Авто- и ручная визуализация ===
def generate_manual_chart(
    df: pd.DataFrame,
    x: str,
    y: Optional[str] = None,
    chart_type: str = "Гистограмма"
):
    """Строит график выбранного типа."""
    try:
        if x not in df.columns or (y and y not in df.columns):
            return None
        x_num = pd.api.types.is_numeric_dtype(df[x])
        y_num = pd.api.types.is_numeric_dtype(df[y]) if y else False
        is_time_x = is_temporal(x, df[x])

        if chart_type == "Гистограмма":
            return px.histogram(df, x=x, nbins=30 if x_num else None)
        elif chart_type == "Круговая диаграмма":
            counts = df[x].value_counts()
            return px.pie(names=counts.index, values=counts.values)
        elif chart_type == "Точечный график" and y:
            return px.scatter(df, x=x, y=y)
        elif chart_type == "Boxplot" and y:
            return px.box(df, x=x, y=y)
        elif chart_type == "Bar-график" and y:
            return px.bar(df, x=x, y=y)
        elif chart_type == "Лайнплот" and y and is_time_x:
            return px.line(df, x=x, y=y)
    except Exception as e:
        st.warning(f"Ошибка при построении графика: {e}")
    return None

def generate_auto_chart(
    df: pd.DataFrame,
    x: str,
    y: Optional[str] = None
):
    """Автоматически подбирает тип графика."""
    try:
        if x not in df.columns or (y and y not in df.columns):
            return None
        x_num = pd.api.types.is_numeric_dtype(df[x])
        y_num = pd.api.types.is_numeric_dtype(df[y]) if y else False
        is_time_x = is_temporal(x, df[x])

        if y:
            if x_num and y_num:
                return px.line(df, x=x, y=y) if is_time_x else px.scatter(df, x=x, y=y)
            if not x_num and y_num:
                return px.bar(df.groupby(x)[y].mean().reset_index(), x=x, y=y)
            if not x_num and not y_num:
                return px.histogram(df, x=x, color=y, barmode="group")
            return px.bar(df, x=x, y=y)
        else:
            if x_num:
                return px.histogram(df, x=x)
            counts = df[x].value_counts()
            return px.pie(names=counts.index, values=counts.values)
    except Exception as e:
        st.warning(f"Ошибка авто-визуализации: {e}")
    return None

def plot_data_visualizations(df, x, y=None, numeric_filters=None, chart_type="Автоматически"):
    if x not in df.columns or (y and y not in df.columns) or (x == y and y is not None):
        st.warning("Некорректный выбор переменных.")
        return None

    df_filtered = apply_numeric_filters(df, numeric_filters or {})

    if df_filtered.empty:
        st.info("После фильтрации данных не осталось.")
        return None

    fig = None
    if chart_type == "Автоматически":
        fig = generate_auto_chart(df_filtered, x, y)
    else:
        fig = generate_manual_chart(df_filtered, x, y, chart_type)

    if fig is None:
        st.info("Выбранный тип графика не подходит для этих данных. Попробуйте другой.")
    return fig


# === Рекомендации от ИИ ===
def suggest_visualization_combinations(df_info: str) -> str:
    """Запрашивает у ИИ рекомендации по визуализациям."""
    try:
        prompt = (
            "Предложи 2–3 интересные комбинации для визуализации (X и Y) "
            "и коротко поясни, что можно увидеть."
        )
        return chat_with_context(prompt)
    except Exception as e:
        return f"Не удалось получить рекомендации: {e}"

# === Корреляции ===
def plot_correlation_heatmap(df: pd.DataFrame):
    """Строит тепловую карту корреляций."""
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr().round(2)
    return px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="🔗 Тепловая карта корреляций"
    )

# === Pivot ===
def generate_pivot_table(df: pd.DataFrame, index_col: str, value_col: str, agg_func: str = "mean"):
    """Строит сводную таблицу по index_col с агрегированием value_col."""
    if index_col not in df.columns or value_col not in df.columns:
        return None

    if agg_func not in {"mean", "sum", "count"}:
        return None

    grouped = df.groupby(index_col, as_index=False)[value_col]

    if agg_func == "mean":
        pivot = grouped.mean()
    elif agg_func == "sum":
        pivot = grouped.sum()
    elif agg_func == "count":
        pivot = grouped.count()

    if pivot.shape[1] == 2:
        agg_col_name = f"{agg_func}({value_col})"
        pivot.columns = [index_col, agg_col_name]
        # 🔽 сортировка по убыванию
        pivot = pivot.sort_values(by=agg_col_name, ascending=False).reset_index(drop=True)
        return pivot
    else:
        return pivot
    


# Интеграция с ИИ (ожидается, что эти функции уже есть в проекте)
from AI_helper import send_correlation_to_ai, send_pivot_to_ai


def show_chart_tab(df: pd.DataFrame) -> None:
    """Вкладка: выбор переменных, тип графика, фильтры и построение графика."""
    st.subheader("🧭 Выбор переменных")

    # X и Y в одной строке
    col1, col2 = st.columns(2)
    with col1:
        x = st.selectbox(
            "🟥 Ось X",
            df.columns,
            index=st.session_state.get("eda_x_index", 0),
            key="eda_x",
        )
    with col2:
        y_options = ["— не выбрано —"] + list(df.columns)
        y = st.selectbox(
            "🟦 Ось Y (необязательно)",
            y_options,
            index=st.session_state.get("eda_y_index", 0),
            key="eda_y",
        )
        # Преобразуем выбор в None, если выбрано "— не выбрано —"
        if y == "— не выбрано —":
            y = None

    # Синхронизируем индексы выбора в session_state
    st.session_state["eda_x_index"] = list(df.columns).index(x)
    st.session_state["eda_y_index"] = y_options.index(y if y is not None else "— не выбрано —")

    # Защита от совпадения X и Y
    if x == y and y is not None:
        st.warning("Переменные X и Y не должны совпадать.")
        y = None

    st.markdown("---")

    # Тип графика (вынесено в экспандер)
    with st.expander("🎨 Тип графика", expanded=True):
        chart_options = [
            "Автоматически",
            "Гистограмма",
            "Круговая диаграмма",
            "Точечный график",
            "Boxplot",
            "Bar-график",
            "Лайнплот",
        ]
        chart_type = st.selectbox(
            "Выберите подходящий тип графика",
            options=chart_options,
            index=st.session_state.get("eda_chart_index", 0),
            key="eda_chart",
        )
        st.session_state["eda_chart_index"] = chart_options.index(chart_type)
        build_chart = st.button("📊 Построить график", key="build_chart")

    st.markdown("---")

    # График с фильтрами
    with st.expander("📈 График с фильтрами", expanded=True):
        filters = {}
        cols_to_filter = [x] + ([y] if y else [])
        for col in dict.fromkeys(cols_to_filter):
            if col and pd.api.types.is_numeric_dtype(df[col]):
                lo, hi = float(df[col].min()), float(df[col].max())
                if lo != hi:
                    sel = st.slider(
                        f"Фильтр по {col}",
                        min_value=lo,
                        max_value=hi,
                        value=st.session_state.get(f"slider_{col}", (lo, hi)),
                        key=f"slider_{col}",
                    )
                    filters[col] = sel

        if build_chart:
            with st.spinner("Построение графика..."):
                time.sleep(2.5)  # имитация работы
                fig = plot_data_visualizations(
                    df=df,
                    x=x,
                    y=y,
                    numeric_filters=filters,
                    chart_type=chart_type,
                )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Визуализация недоступна для выбранных параметров.")
        else:
            st.info("🎯 Выберите переменные и нажмите «Построить график».")            

def show_ai_suggestions(df: pd.DataFrame) -> None:
    """Блок с советами от ИИ по визуализациям (вынесен отдельно)."""
    with st.expander("💡 Получить советы для визуализации от ИИ"):
        if st.button("✨ Предложи комбинации", key="suggest_combinations"):
            df_info = f"Переменные: {', '.join(df.columns)}"
            with st.spinner("Генерируем рекомендации..."):
                time.sleep(2)
                st.session_state["eda_suggestion"] = suggest_visualization_combinations(df_info)

        if "eda_suggestion" in st.session_state:
            st.markdown("**📝 Рекомендации от ИИ:**")
            st.info(st.session_state["eda_suggestion"], icon="🤖")


def show_correlation_tab(df: pd.DataFrame) -> None:
    """Вкладка: тепловая карта корреляций и фиксация в ИИ."""
    st.subheader("❄️ Тепловая карта корреляций")
    fig = plot_correlation_heatmap(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 Чем ближе значение к 1 или -1, тем сильнее линейная связь между переменными.")

        st.caption("Нажмите «Зафиксировать в ИИ», чтобы сохранить результат в чат.")
        if st.button("📤 Зафиксировать корреляции в ИИ", key="fix_corr"):
            try:
                _ = send_correlation_to_ai(df)
                st.session_state["correlation_saved"] = True
                st.success("✅ Корреляции зафиксированы в ИИ.")
            except Exception as e:
                st.error(f"Не удалось отправить корреляции в ИИ: {e}")
        elif st.session_state.get("correlation_saved"):
            st.info("✅ Корреляции уже были зафиксированы.")
    else:
        st.info("Невозможно построить тепловую карту.")


import time

def show_pivot_tab(df: pd.DataFrame) -> None:
    """Вкладка: сводные таблицы (pivot) и фиксация результата в ИИ + визуализация."""
    st.subheader("📊 Сводные таблицы (Pivot)")

    col1, col2 = st.columns(2)
    with col1:
        index_col = st.selectbox(
            "Группировать по",
            df.columns,
            index=st.session_state.get("pivot_index_index", 0),
            key="pivot_index",
        )
        st.session_state["pivot_index_index"] = list(df.columns).index(index_col)

    with col2:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) == 0:
            st.warning("Нет числовых столбцов для агрегации.")
            return
        value_col = st.selectbox(
            "Агрегировать",
            num_cols,
            index=st.session_state.get("pivot_value_index", 0),
            key="pivot_value",
        )
        st.session_state["pivot_value_index"] = list(num_cols).index(value_col)

    agg_options = ["mean", "sum", "count"]
    agg_func = st.radio(
        "Метод агрегации",
        agg_options,
        index=st.session_state.get("pivot_agg_index", 0),
        horizontal=True,
        key="pivot_agg",
    )
    st.session_state["pivot_agg_index"] = agg_options.index(agg_func)

    pivot_table = generate_pivot_table(df, index_col, value_col, agg_func)
    if pivot_table is not None and not pivot_table.empty:
        st.dataframe(pivot_table, use_container_width=True)

        # === Кнопка фиксации в ИИ сразу после таблицы ===
        st.caption("Нажмите «Зафиксировать в ИИ», чтобы сохранить результат в чат.")
        if st.button("📤 Зафиксировать в ИИ", key="fix_pivot"):
            try:
                _ = send_pivot_to_ai(pivot_table, index_col, value_col, agg_func)
                st.session_state["pivot_saved"] = True
                st.success("✅ Сводная таблица зафиксирована в ИИ.")
            except Exception as e:
                st.error(f"Не удалось отправить сводную таблицу в ИИ: {e}")
        elif st.session_state.get("pivot_saved"):
            st.info("✅ Сводная таблица уже была зафиксирована.")

        # Подсказка для пользователя

        st.markdown("---")  # 🔽 Разделитель

        # === Блок выбора графика ===
        st.markdown("### 📉 Визуализация сводной таблицы")
        chart_type = st.selectbox(
            "Выберите тип графика",
            ["Bar", "Pie", "Line"],
            key="pivot_chart_type"
        )

        if st.button("Визуализировать", key="pivot_visualize"):
            # искусственная задержка для реалистичности
            with st.spinner("⏳ Строим визуализацию..."):
                time.sleep(1.5)

            if pivot_table.shape[1] < 2:
                st.warning("⚠️ Для визуализации нужно выбрать корректные переменные (группировка + числовая агрегация).")
            else:
                agg_col = pivot_table.columns[1]

                if chart_type == "Bar":
                    fig = px.bar(
                        pivot_table,
                        x=index_col,
                        y=agg_col,
                        text=agg_col,
                        title=f"{chart_type} график: {agg_func}({value_col}) по {index_col}"
                    )
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    fig.update_layout(yaxis_title=agg_col, xaxis_title=index_col)
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Pie":
                    if pivot_table[index_col].nunique() >= 10:
                        st.warning("⚠️ Слишком много категорий для круговой диаграммы (10 и более). Выберите другой тип графика.")
                    else:
                        fig = px.pie(
                            pivot_table,
                            names=index_col,
                            values=agg_col,
                            title=f"{chart_type} график: {agg_func}({value_col}) по {index_col}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Line":
                    fig = px.line(
                        pivot_table,
                        x=index_col,
                        y=agg_col,
                        markers=True,
                        title=f"{chart_type} график: {agg_func}({value_col}) по {index_col}"
                    )
                    fig.update_layout(yaxis_title=agg_col, xaxis_title=index_col)
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("⚠️ Возможно, вы выбрали одни и те же столбцы или таблица пустая.")
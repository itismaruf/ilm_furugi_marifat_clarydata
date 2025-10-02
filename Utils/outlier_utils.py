import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List
from scipy.stats import skew

import streamlit as st

def render_outlier_handling_info():
    """
    Рендерит скрытую секцию с краткой инструкцией по работе с выбросами.
    """
    with st.expander("ℹ️ Инструкция по работе с разделом обработки выбросов", expanded=False):
        st.markdown("""
        В этом разделе вы можете исследовать и обрабатывать выбросы в ваших данных.

        - Анализ выбросов с помощью IQR-метода или Z-score.
        - Автообработка выбросов (стандартный IQR-критерий).
        - Ручная очистка с выбором метода и границ, включая удаление по процентилям.
        """)



def detect_outliers_iqr(df: pd.DataFrame,
                        cols: list,
                        q_low: float = 0.25,
                        q_high: float = 0.75) -> dict:
    """
    Идентифицирует выбросы методом IQR.
    Возвращает dict: {column: boolean Series}, True там, где выброс.
    """
    masks = {}
    for col in cols:
        series = df[col].dropna()
        Q1 = series.quantile(q_low)
        Q3 = series.quantile(q_high)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        masks[col] = (df[col] < lower) | (df[col] > upper)
    return masks


def detect_outliers_zscore(df: pd.DataFrame,
                           cols: list,
                           z_thresh: float = 3.0) -> dict:
    """
    Идентифицирует выбросы по Z-score.
    Возвращает dict: {column: boolean Series}, True там, где |z| > z_thresh.
    """
    masks = {}
    for col in cols:
        series = df[col]
        mu = series.mean()
        sigma = series.std()
        # избегаем деления на ноль
        if sigma == 0 or pd.isna(sigma):
            masks[col] = pd.Series(False, index=df.index)
        else:
            z = (series - mu) / sigma
            masks[col] = z.abs() > z_thresh
    return masks


def plot_outliers_distribution(
    df: pd.DataFrame,
    masks: Dict[str, pd.Series],
    cols: List[str]
) -> go.Figure:
    """
    Строит scatter-фасеты и добавляет текст под графиком (вне plot area).
    """
    if not cols:
        fig = go.Figure()
        fig.update_layout(
            title="Нет выбранных столбцов для визуализации",
            xaxis={'visible': False},
            yaxis={'visible': False},
            margin=dict(b=200)  # увеличенный отступ низа
        )
        fig.add_annotation(
            text="Выберите хотя бы один столбец",
            xref="paper", yref="paper",
            x=0.5, y=-0.5,
            xanchor="center", yanchor="top",
            showarrow=False,
            font=dict(size=14)
        )
        return fig

    # Формируем датасет для Facet scatter
    plots = []
    for col in cols:
        plots.append(pd.DataFrame({
            "index": df.index,
            "value": df[col],
            "is_outlier": masks.get(col, pd.Series(False, index=df.index)),
            "feature": col
        }))
    long_df = pd.concat(plots, ignore_index=True)

    # Строим Facet scatter
    fig = px.scatter(
        long_df,
        x="index", y="value",
        color="is_outlier",
        facet_col="feature",
        color_discrete_map={False: "blue", True: "red"},
        title="Распределение значений и выбросов"
    )

    # Настраиваем отступы, чтобы текст не налазил
    fig.update_layout(
        showlegend=False,
        margin=dict(t=60, b=130)
    )

    # Текст-пояснение под графиком
    fig.add_annotation(
        text=(
            "Синие точки — нормальные значения; "
            "красные — выбросы (значения за пределами IQR/Z-score)."
        ),
        xref="paper", yref="paper",
        x=0.5, y=-0.3,
        xanchor="center", yanchor="top",
        showarrow=False,
        font=dict(size=12)
    )

    return fig


def plot_outlier_removal_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    cols: List[str]
) -> go.Figure:
    """
    Сравнение распределений 'до' и 'после' удаления выбросов.
    Для каждого столбца рисует наложенные гистограммы, нормированные в плотность.
    """
    # Готовим длинный DataFrame
    df_b = df_before[cols].copy().assign(dataset="before")
    df_a = df_after[cols].copy().assign(dataset="after")
    long_df = pd.concat([df_b, df_a], ignore_index=True)

    long_df = long_df.melt(
        id_vars="dataset",
        var_name="feature",
        value_name="value"
    )

    # Строим гистограммы с контрастными цветами
    fig = px.histogram(
        long_df,
        x="value",
        color="dataset",
        facet_col="feature",
        facet_col_wrap=3,
        opacity=0.6,
        barmode="overlay",
        histnorm="density",
        nbins=50,
        color_discrete_map={
            "before": "#2ca02c",   # зелёный для 'до'
            "after":  "#d62728"    # красный для 'после'
        }
    )

    # Оставляем только подпись столбца в заголовках фасетов
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    fig.update_layout(
        legend_title_text="Дата-сэт",
        margin=dict(t=60, b=20),
    )

    return fig



def outliers_summary(df: pd.DataFrame, masks: dict) -> pd.DataFrame:
    """
    Возвращает DataFrame с краткой статистикой выбросов:
    столбец, общее число выбросов, % выбросов.
    """
    records = []
    n = len(df)
    for col, mask in masks.items():
        cnt = int(mask.sum())
        records.append({
            "column": col,
            "total_outliers": cnt,
            "percent": round(cnt / n * 100, 2)
        })
    return pd.DataFrame(records)


def run_auto_outlier_removal(df: pd.DataFrame, z_thresh: float = 3.0):
    """
    Автоматически удаляет выбросы:
      - Если распределение числового признака ~нормальное (|skew| < 1) → Z-score
      - Иначе → IQR
    Возвращает:
      before_df  — DataFrame с количеством выбросов до очистки,
      log        — список словарей {"column", "method", "removed_count"},
      cleaned_df — итоговый DataFrame без выбросов.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    before = []
    log = []
    cleaned = df.copy()

    for col in numeric_cols:
        s = cleaned[col].dropna()
        if s.empty:
            continue

        try:
            skewness = skew(s)
            method = "Z-score" if abs(skewness) < 1 else "IQR"

            if method == "IQR":
                masks = detect_outliers_iqr(cleaned, [col], q_low=0.25, q_high=0.75)
                removed = int(masks[col].sum())
                if removed > 0:
                    before.append({"column": col, "removed_count": removed})
                    log.append({"column": col, "method": "IQR", "removed_count": removed})
                    cleaned = cleaned.loc[~masks[col]]

            elif method == "Z-score":
                mu, sigma = s.mean(), s.std()
                if sigma == 0 or pd.isna(sigma):
                    log.append({"column": col, "method": "Z-score", "removed_count": 0, "note": "std=0"})
                    continue

                z = (cleaned[col] - mu) / sigma
                mask = z.abs() > z_thresh
                removed = int(mask.sum())
                if removed > 0:
                    before.append({"column": col, "removed_count": removed})
                    log.append({"column": col, "method": "Z-score", "removed_count": removed})
                    cleaned = cleaned.loc[~mask]

        except Exception as e:
            log.append({"column": col, "method": "auto", "error": str(e)})

    before_df = pd.DataFrame(before)
    return before_df, log, cleaned



def render_outlier_rules_table():
    """
    Показывает таблицу с правилами автообработки выбросов и пояснениями.
    """
    rules = [
        {
            "Метод": "IQR (стандартный)",
            "Границы": "q_low=0.25, q_high=0.75",
            "Формула": "Q1 - 1.5⋅IQR, Q3 + 1.5⋅IQR"
        },
        {
            "Метод": "Z-score",
            "Границы": "threshold=3.0",
            "Формула": "|x - μ| / σ > threshold"
        }
    ]

    df_rules = pd.DataFrame(rules).set_index("Метод")
    st.markdown("### 📌 Правила автообработки выбросов")
    st.table(df_rules)

    st.markdown("**Примечание:** при нажатии «Автоочистка» будет выбран метод в зависимости от распределения данных:")
    st.markdown(
        "- Если данные имеют выраженную асимметрию или содержат экстремальные значения → IQR-метод\n"
        "- Если данные близки к нормальному распределению → Z-score-метод"
    )


def remove_outliers_iqr(df: pd.DataFrame,
                        cols: list,
                        q_low: float = 0.25,
                        q_high: float = 0.75) -> pd.DataFrame:
    """
    Удаляет выбросы по IQR-методу для указанных столбцов.
    """
    cleaned = df.copy()
    masks = detect_outliers_iqr(cleaned, cols, q_low, q_high)
    # объединяем все маски: строка удаляется, если в любом столбце выброс
    combined = np.logical_or.reduce([masks[c] for c in cols])
    return cleaned.loc[~combined]


def remove_outliers_zscore(df: pd.DataFrame,
                           cols: list,
                           z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Удаляет выбросы по Z-score для указанных столбцов.
    """
    cleaned = df.copy()
    masks = detect_outliers_zscore(cleaned, cols, z_thresh)
    combined = np.logical_or.reduce([masks[c] for c in cols])
    return cleaned.loc[~combined]


def cap_outliers(df: pd.DataFrame,
                 cols: list,
                 q_low: float = 0.25,
                 q_high: float = 0.75) -> pd.DataFrame:
    """
    «Каппит» выбросы по IQR-методу: заменяет значения ниже/выше границ
    на границы.
    """
    capped = df.copy()
    for col in cols:
        series = capped[col].dropna()
        Q1 = series.quantile(q_low)
        Q3 = series.quantile(q_high)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        capped[col] = capped[col].clip(lower, upper)
    return capped


def remove_outliers_percentile(df: pd.DataFrame,
                               cols: list,
                               p_low: float,
                               p_high: float) -> pd.DataFrame:
    """
    Удаляет строки, если значение в столбце выходит за заданные
    процентильные границы.
    """
    cleaned = df.copy()
    for col in cols:
        low_val = np.percentile(cleaned[col].dropna(), p_low)
        high_val = np.percentile(cleaned[col].dropna(), p_high)
        cleaned = cleaned[(cleaned[col] >= low_val) & (cleaned[col] <= high_val)]
    return cleaned

def show_outlier_summary(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    cols: List[str]
):
    """
    Показывает сравнительную таблицу для каждого столбца:
      mean, std, min, max – до и после очистки, 
      и общее число удалённых строк.
    Все числовые значения округлены до 2 знаков после точки.
    """
    summary = []
    for col in cols:
        stats_before = before_df[col].describe()
        stats_after  = after_df[col].describe()

        summary.append({
            "column":      col,
            "mean_before": stats_before["mean"],
            "mean_after":  stats_after["mean"],
            "std_before":  stats_before["std"],
            "std_after":   stats_after["std"],
            "min_before":  stats_before["min"],
            "min_after":   stats_after["min"],
            "max_before":  stats_before["max"],
            "max_after":   stats_after["max"],
        })

    df_summary = (
        pd.DataFrame(summary)
          .set_index("column")
          .round(2)     # <-- округляем все числовые столбцы
    )

    st.markdown("**Статистики до и после очистки выбросов**")
    st.table(df_summary)

    removed_rows = len(before_df) - len(after_df)
    st.write(f"Удалено строк всего: {removed_rows}")
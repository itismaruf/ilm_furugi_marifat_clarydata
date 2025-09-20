import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import ttest_ind, ttest_rel, f_oneway, chi2_contingency

# ==== Утилиты ====
def is_numeric(series: pd.Series) -> bool:
    """Проверяет, является ли серия числовой."""
    return pd.api.types.is_numeric_dtype(series)

def is_categorical(series: pd.Series) -> bool:
    """Проверяет, является ли серия категориальной."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series)

# ==== Сводка по группам ====
def group_summary(df: pd.DataFrame, num_col: str, cat_col: str) -> pd.DataFrame:
    """Возвращает DataFrame со средними, SD, SE и количеством наблюдений по группам."""
    summary = (
        df.groupby(cat_col)[num_col]
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={
            'mean': 'Среднее',
            'std': 'SD',
            'count': 'N',
            cat_col: 'Группа'
        })
    )
    summary['SE'] = summary['SD'] / summary['N']**0.5
    return summary

# ==== Вывод результатов теста ====
def display_test_result(test_name: str, stat_label: str, stat_value: float, p_value: float, alpha: float = 0.05):
    """Единый формат вывода результатов статистического теста."""
    if p_value < alpha:
        st.success(f"✅ {test_name}: различия значимы (p = {p_value:.4f})")
    else:
        st.info(f"ℹ️ {test_name}: различия незначимы (p = {p_value:.4f})")

    st.metric(label=stat_label, value=f"{stat_value:.4f}")
    st.metric(label="p‑value", value=f"{p_value:.4f}")

# ==== Bar chart со средними ====
def plot_group_means(summary_df: pd.DataFrame, title: str = "Сравнение средних значений"):
    """Строит bar chart со средними значениями и доверительными интервалами (SE)."""
    fig = px.bar(
        summary_df,
        x="Группа",
        y="Среднее",
        error_y="SE",
        text="Среднее",
        color="Группа",
        title=title
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# ==== Таблица под графиком ====
def display_summary_table(summary_df: pd.DataFrame):
    """Отображает таблицу со статистикой по группам."""
    st.dataframe(
        summary_df.style.format({
            "Среднее": "{:.2f}",
            "SD": "{:.2f}",
            "SE": "{:.2f}",
            "N": "{:d}"
        })
    )

# ==== Chi2 визуализация ====
def plot_chi2_table(table: pd.DataFrame, plot_choice: str = "Авто"):
    """Строит график для таблицы сопряжённости."""
    n_levels_x = table.shape[1]
    n_levels_y = table.shape[0]

    if plot_choice == "Авто":
        if n_levels_x <= 5 and n_levels_y <= 5:
            fig = px.imshow(
                table.values,
                labels=dict(
                    x=table.columns.name or "Col2",
                    y=table.index.name or "Col1",
                    color="Count"
                ),
                x=table.columns, y=table.index,
                text_auto=True, color_continuous_scale="Blues"
            )
        else:
            table_long = table.reset_index().melt(id_vars=table.index.name, value_name="count")
            fig = px.bar(
                table_long,
                x=table.columns.name, y="count", color=table.index.name,
                barmode="group", title="Сравнение категориальных частот"
            )
    elif plot_choice == "Heatmap":
        fig = px.imshow(
            table.values,
            labels=dict(
                x=table.columns.name or "Col2",
                y=table.index.name or "Col1",
                color="Count"
            ),
            x=table.columns, y=table.index,
            text_auto=True, color_continuous_scale="Blues"
        )
    elif plot_choice == "Stacked bar":
        table_long = table.reset_index().melt(id_vars=table.index.name, value_name="count")
        fig = px.bar(
            table_long,
            x=table.columns.name, y="count", color=table.index.name,
            barmode="stack", title="Stacked bar chart"
        )
    else:  # Clustered bar
        table_long = table.reset_index().melt(id_vars=table.index.name, value_name="count")
        fig = px.bar(
            table_long,
            x=table.columns.name, y="count", color=table.index.name,
            barmode="group", title="Clustered bar chart"
        )

    st.plotly_chart(fig, use_container_width=True)

# ==== T-test ====
def run_ttest(df: pd.DataFrame, col: str, group_col: str, paired: bool = False):
    """Выполняет t‑test (независимый или парный)."""
    if not is_numeric(df[col]):
        st.error("❌ Для t‑test нужен числовой признак.")
        return
    if not is_categorical(df[group_col]):
        st.error("❌ Группирующая переменная должна быть категориальной.")
        return

    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        st.error("❌ Для t‑test должно быть ровно 2 группы.")
        return

    g1 = df[df[group_col] == groups[0]][col].dropna()
    g2 = df[df[group_col] == groups[1]][col].dropna()

    if paired:
        if len(g1) != len(g2):
            st.error(f"❌ Для парного t‑test группы должны быть одинаковой длины. Сейчас: {len(g1)} и {len(g2)}.")
            return
        stat, p = ttest_rel(g1, g2)
    else:
        stat, p = ttest_ind(g1, g2)

    summary_df = group_summary(df, col, group_col)
    display_test_result("t‑test", "t‑статистика", stat, p)
    plot_group_means(summary_df)
    display_summary_table(summary_df)

# ==== ANOVA ====
def run_anova(df: pd.DataFrame, col: str, group_col: str):
    """Выполняет однофакторный дисперсионный анализ (ANOVA)."""
    if not is_numeric(df[col]):
        st.error("❌ Для ANOVA нужен числовой признак.")
        return
    if not is_categorical(df[group_col]):
        st.error("❌ Группирующая переменная должна быть категориальной.")
        return

    groups = [df[df[group_col] == g][col].dropna() for g in df[group_col].dropna().unique()]
    if len(groups) < 3:
        st.error("❌ Для ANOVA минимум 3 группы.")
        return

    stat, p = f_oneway(*groups)
    summary_df = group_summary(df, col, group_col)
    display_test_result("ANOVA", "F‑статистика", stat, p)
    plot_group_means(summary_df)
    display_summary_table(summary_df)

# ==== Chi-squared ====
def run_chi2(df: pd.DataFrame, col1: str, col2: str, plot_choice: str = "Авто"):
    """Выполняет критерий хи‑квадрат Пирсона."""
    if not (is_categorical(df[col1]) and is_categorical(df[col2])):
        st.error("❌ Для Chi‑square нужны два категориальных признака.")
        return

    table = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(table)

    display_test_result("Chi‑square", "Chi²‑статистика", chi2, p)
    plot_chi2_table(table, plot_choice)

    # Таблица наблюдаемых и ожидаемых значений
    expected_df = pd.DataFrame(expected, index=table.index, columns=table.columns)
    st.subheader("📊 Наблюдаемые значения")
    st.dataframe(table)
    st.subheader("📊 Ожидаемые значения")
    st.dataframe(expected_df.round(2))


# stats_tests_ui.py
def show_ttest_ui(df):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols_all = df.select_dtypes(exclude=["number"]).columns.tolist()

    if not num_cols:
        st.info("ℹ️ Нет числовых признаков для t‑test.")
        return
    if not cat_cols_all:
        st.info("ℹ️ Нет категориальных признаков для t‑test.")
        return

    target_col = st.selectbox("Числовой признак (метрика)", num_cols, key="ttest_num")
    group_col = st.selectbox("Категориальный признак", cat_cols_all, key="ttest_group")

    levels = df[group_col].dropna().unique().tolist()
    if len(levels) == 2:
        st.caption(f"Группы: {levels[0]!r} и {levels[1]!r}")
        paired = st.checkbox("Парный t‑test (paired)", value=False, key="ttest_paired")
    else:
        picked_levels = st.multiselect(
            "Выберите ДВА уровня для сравнения",
            options=levels, max_selections=2,
            key="ttest_levels"
        )
        paired = st.checkbox("Парный t‑test (paired)", value=False, key="ttest_paired")

    st.markdown("---")
    if st.button("Выполнить t‑test", type="primary"):
        if len(levels) == 2:
            run_ttest(df, target_col, group_col, paired)
        elif len(picked_levels) == 2:
            df_pair = df[df[group_col].isin(picked_levels)].copy()
            run_ttest(df_pair, target_col, group_col, paired)
        else:
            st.error("❌ Выберите ровно две категории.")


def show_anova_ui(df):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    if not num_cols:
        st.info("ℹ️ Нет числовых признаков для ANOVA.")
        return
    if not cat_cols:
        st.info("ℹ️ Нет категориальных признаков для ANOVA.")
        return

    target_col = st.selectbox("Числовой признак (метрика)", num_cols, key="anova_num")
    group_col = st.selectbox("Категориальный признак (3+ группы)", cat_cols, key="anova_group")

    st.markdown("---")
    if st.button("Выполнить ANOVA", type="primary"):
        run_anova(df, target_col, group_col)


def show_chi2_ui(df):
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    if len(cat_cols) < 2:
        st.info("ℹ️ Для Chi‑square нужно минимум два категориальных признака.")
        return

    col1 = st.selectbox("Категориальный признак №1", cat_cols, key="chi_col1")
    other_cats = [c for c in cat_cols if c != col1] or cat_cols
    col2 = st.selectbox("Категориальный признак №2", other_cats, key="chi_col2")

    plot_choice = st.radio(
        "Тип графика",
        ["Авто", "Heatmap", "Stacked bar", "Clustered bar"],
        horizontal=True, key="chi_plot"
    )

    st.markdown("---")
    if st.button("Выполнить Chi‑square", type="primary"):
        if col1 == col2:
            st.error("❌ Выберите разные категориальные признаки.")
        else:
            run_chi2(df, col1, col2, plot_choice)
import pandas as pd
import streamlit as st

def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает DataFrame со столбцами:
      - column        — имя колонки
      - missing_count — кол-во NaN
      - pct_missing   — % NaN (округлённый)
    Только по тем колонкам, где есть NaN.
    """
    nulls = df.isna().sum()
    stats = pd.DataFrame({
        "column":        nulls.index,
        "missing_count": nulls.values,
        "pct_missing":   (nulls / len(df) * 100).round(1).values
    })
    return stats[stats["missing_count"] > 0].reset_index(drop=True)


def is_categorical(series: pd.Series) -> bool:
    """Определяет, является ли колонка категориальной."""
    return (
        series.dtype == 'object'
        or pd.api.types.is_categorical_dtype(series)
        or series.nunique() < 20  # эвристика: мало уникальных значений
    )


def standard_auto_cleaning(df: pd.DataFrame, target_col: str = None):
    """
    Стандартная автоочистка:
      <5% NaN       → дроп строк
      <20% NaN      → заполнение (числ. → median, катег. → mode)
      ≥50% NaN      → дроп колонки
      NaN в target → дроп строк
    Возвращает (new_df, cleaning_log), где cleaning_log — список dict:
      column, missing_count, pct_missing, action
    """
    df_clean = df.copy()
    total = len(df_clean)
    log = []

    for col in df.columns:
        miss = df_clean[col].isna().sum()
        if miss == 0:
            continue

        pct = miss / total * 100
        pct_r = round(pct, 1)

        # Целевая переменная
        if target_col and col == target_col:
            before = len(df_clean)
            df_clean.dropna(subset=[col], inplace=True)
            dropped = before - len(df_clean)
            action = f"дроп строк в target ({dropped} шт.)"

        # <5% NaN — удаляем строки
        elif pct < 5:
            before = len(df_clean)
            df_clean.dropna(subset=[col], inplace=True)
            dropped = before - len(df_clean)
            action = f"удалено строк ({dropped} шт.)"

        # <20% NaN — заполняем
        elif pct < 20:
            s = df_clean[col]
            if pd.api.types.is_numeric_dtype(s) and not is_categorical(s):
                val = s.median()
                df_clean[col].fillna(val, inplace=True)
                action = f"заполнено median={val:.2f}"
            else:
                mode = s.mode()
                if not mode.empty:
                    val = mode[0]
                    df_clean[col].fillna(val, inplace=True)
                    action = f"заполнено mode='{val}'"
                else:
                    action = "не заполнено: mode пустой"

        # ≥50% NaN — удаляем колонку
        elif pct >= 50:
            df_clean.drop(columns=[col], inplace=True)
            action = f"колонка удалена (≥50% NaN)"

        # Остальное — на ручную проверку
        else:
            action = f"оставлено без изменений ({pct_r}% пропусков)"

        log.append({
            "column":        col,
            "missing_count": int(miss),
            "pct_missing":   pct_r,
            "action":        action
        })

    return df_clean, log


def run_auto_cleaning(df: pd.DataFrame, target_col: str = None):
    """
    Обёртка: сначала summarize_missing, потом standard_auto_cleaning.
    Возвращает (stats_before, cleaning_log, new_df).
    """
    stats_before = summarize_missing(df)
    new_df, cleaning_log = standard_auto_cleaning(df, target_col)
    return stats_before, cleaning_log, new_df


import streamlit as st

def render_cleaning_principles():
    """
    Отображает таблицу с принципами автоочистки пропусков.
    """
    with st.expander("📖 Принципы очистки данных и заполнения пропусков (для «Умной очистки данных»)", expanded=False):
        st.markdown("#### Как мы решаем проблемы с пропусками")
        st.markdown("""
<table style="width:100%">
<thead>
<tr>
  <th style="text-align:left">📊 Условие</th>
  <th style="text-align:left">🛠 Действие</th>
</tr>
</thead>
<tbody>
<tr><td>&lt; 5% пропусков</td><td>Удаляем строки с NaN</td></tr>
<tr><td>5–20% пропусков</td><td>Заполняем (числ. → median / кат. → mode)</td></tr>
<tr><td>20–50% пропусков</td><td>Оставляем на ручную проверку</td></tr>
<tr><td>≥ 50% пропусков</td><td>Удаляем весь столбец</td></tr>
<tr><td>Пропуски в целевой переменной `y`</td><td>Удаляем строки — без заполнения</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)




# ===== Ручная очистка =====

def drop_rows_na(df: pd.DataFrame, cols: list, target_col: str = None) -> pd.DataFrame:
    """
    Удаляет все строки, где в любых из cols есть NaN.
    Не затрагивает строки с NaN в target_col.
    """
    df_clean = df.copy()
    subset = [c for c in cols if c in df_clean.columns]
    df_clean.dropna(subset=subset, inplace=True)
    return df_clean

def drop_cols_na(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Удаляет указанные колонки целиком.
    """
    df_clean = df.copy()
    to_drop = [c for c in cols if c in df_clean.columns]
    df_clean.drop(columns=to_drop, inplace=True)
    return df_clean

def fill_na(
    df: pd.DataFrame,
    cols: list,
    method: str,
    constant_value=None
) -> pd.DataFrame:
    """
    Заполняет пропуски в указанных cols:
      - mean/median для числовых
      - mode для категориальных
      - constant — заданным значением
      - unknown — строкой 'unknown'
    """
    df_clean = df.copy()
    for col in cols:
        if col not in df_clean.columns:
            continue
        series = df_clean[col]
        if method == "mean" and pd.api.types.is_numeric_dtype(series):
            df_clean[col].fillna(series.mean(), inplace=True)
        elif method == "median" and pd.api.types.is_numeric_dtype(series):
            df_clean[col].fillna(series.median(), inplace=True)
        elif method == "mode":
            mode = series.mode()
            if not mode.empty:
                df_clean[col].fillna(mode[0], inplace=True)
        elif method == "constant":
            df_clean[col].fillna(constant_value, inplace=True)
        elif method == "unknown":
            df_clean[col].fillna("unknown", inplace=True)
    return df_clean

def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает DataFrame со столбцами:
      - column, missing_count, pct_missing
    Только для тех колонок, где есть NaN.
    """
    nulls = df.isna().sum()
    stats = pd.DataFrame({
        "column":        nulls.index,
        "missing_count": nulls.values,
        "pct_missing":   (nulls / len(df) * 100).round(1).values
    })
    return stats[stats["missing_count"] > 0].reset_index(drop=True)


def render_nan_handling_info():
    st.markdown("### 🧭 Как пользоваться этим разделом")
    with st.expander("ℹ️ Рекомендуем сначала прочитать — нажмите, чтобы раскрыть"):
        st.markdown("""
### Обработка пропущенных значений (NaN)

В данном разделе предусмотрены два подхода к обработке пропусков в наборе данных.

#### Методы очистки

- **Автоматическая очистка**  
  Осуществляется в соответствии с заранее заданными правилами, что обеспечивает оперативность и стандартизацию процесса.

- **Ручная очистка**  
  Предоставляет возможность пользователю самостоятельно определять способ заполнения или удаления пропущенных значений, обеспечивая гибкий контроль над результатом.

        """)


def render_nan_rules_table():
    st.markdown("### 📋 Рекомендованные действия в зависимости от процента пропусков")
    st.markdown(
                "| % пропусков | Действие                               |\n"
                "|------------:|----------------------------------------|\n"
                "| < 5%        | Удалить строки                         |\n"
                "| 5–20%       | Заполнить (числ.→median / кат.→mode)   |\n"
                "| 20–50%      | Оставить без изменений                 |\n"
                "| ≥ 50%       | Удалить столбец                        |\n"
                "| NaN в target| Удалить строки                         |"
            )


def drop_selected_cols(df, cols):
    """Удаляет указанные столбцы из DataFrame."""
    return df.drop(columns=cols, errors="ignore")

def show_na_summary(before: pd.DataFrame,
                    after: pd.DataFrame,
                    cols: list[str],
                    title_before="До",
                    title_after="После"):
    # Считаем пропуски
    cnt_before = before[cols].isna().sum()
    # Только те, где > 0
    cnt_before = cnt_before[cnt_before > 0]

    # Для after учитываем, что колонки могли удалиться
    common = [c for c in cols if c in after.columns]
    cnt_after = after[common].isna().sum() if common else pd.Series(dtype=int)
    cnt_after = cnt_after[cnt_after > 0]

    # Отображаем
    if cnt_before.empty:
        st.info("Нет пропусков до обработки")
    else:
        st.markdown(f"**{title_before}**")
        st.table(cnt_before.rename("NaN").to_frame())

    if cnt_after.empty:
        st.success("Нет пропусков после обработки")
    else:
        st.markdown(f"**{title_after}**")
        st.table(cnt_after.rename("NaN").to_frame())

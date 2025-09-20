import pandas as pd
import io
import os
import streamlit as st

# ======= Общие =======

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

# ======= Автоочистка =======

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




def render_nan_rules_table():
    """
    Отображает таблицу с правилами автоочистки NaN.
    """
    st.markdown(
        "| % пропусков | Действие                               |\n"
        "|------------:|----------------------------------------|\n"
        "| < 5%        | Удалить строки                         |\n"
        "| 5–20%       | Заполнить (числ.→median / кат.→mode)   |\n"
        "| 20–50%      | Оставить без изменений                 |\n"
        "| ≥ 50%       | Удалить столбец                        |\n"
        "| NaN в target| Удалить строки                         |"
    )


def run_auto_cleaning(df: pd.DataFrame, target_col: str = None):
    """
    Обёртка: сначала summarize_missing, потом standard_auto_cleaning.
    Возвращает (stats_before, cleaning_log, new_df).
    """
    stats_before = summarize_missing(df)
    new_df, cleaning_log = standard_auto_cleaning(df, target_col)
    return stats_before, cleaning_log, new_df

# ======= Ручная очистка =======

def drop_rows_na(df: pd.DataFrame, cols: list, target_col: str = None) -> pd.DataFrame:
    """Удаляет все строки, где в любых из cols есть NaN."""
    df_clean = df.copy()
    subset = [c for c in cols if c in df_clean.columns]
    df_clean.dropna(subset=subset, inplace=True)
    return df_clean

def drop_cols_na(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Удаляет указанные колонки целиком."""
    df_clean = df.copy()
    to_drop = [c for c in cols if c in df_clean.columns]
    df_clean.drop(columns=to_drop, inplace=True)
    return df_clean

def drop_selected_cols(df, cols):
    """Удаляет указанные столбцы из DataFrame."""
    return df.drop(columns=cols, errors="ignore")

def fill_na(df: pd.DataFrame, cols: list, method: str, constant_value=None) -> pd.DataFrame:
    """Заполняет пропуски в указанных cols."""
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

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет полностью одинаковые строки из DataFrame."""
    return df.drop_duplicates().reset_index(drop=True)

def apply_manual_cleaning(df, action, cols, target=None, method=None, value=None):
    """
    Универсальный обработчик для ручной очистки.
    """
    if action == "Удалить строки":
        return drop_rows_na(df, cols, target)
    elif action == "Удалить столбцы (с NaN)":
        return drop_cols_na(df, cols)
    elif action == "Удалить выбранные столбцы":
        return drop_selected_cols(df, cols)
    elif action == "Заполнить NaN":
        return fill_na(df, cols, method, value)
    elif action == "Удалить дубликаты":
        return remove_duplicates(df)
    return df

# ======= Сравнение до/после =======

def show_na_summary(before: pd.DataFrame,
                    after: pd.DataFrame,
                    cols: list[str],
                    title_before="До",
                    title_after="После"):
    """
    Показывает сводку по NaN до и после обработки.
    """
    import streamlit as st
    cnt_before = before[cols].isna().sum()
    cnt_before = cnt_before[cnt_before > 0]
    common = [c for c in cols if c in after.columns]
    cnt_after = after[common].isna().sum() if common else pd.Series(dtype=int)
    cnt_after = cnt_after[cnt_after > 0]

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


def prepare_csv_download(df: pd.DataFrame, original_filename: str = None):
    """
    Готовит CSV-файл в буфере для скачивания.
    Возвращает (file_name, buffer).
    """
    base_name = "data"
    if original_filename:
        base_name = os.path.splitext(original_filename)[0]

    file_name = f"{base_name}_cleaned.csv"

    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # сброс указателя в начало

    return file_name, csv_buffer
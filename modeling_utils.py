# modeling_utils.py
import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from typing import Dict, List, Tuple, Optional, Any

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve
)

import plotly.express as px
import plotly.graph_objects as go


# =========================
# Feature utils
# =========================

def split_features_by_type(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    """Разделяет признаки на числовые и категориальные."""
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    return num_cols, cat_cols


def prepare_features_and_target(df: pd.DataFrame, target_col: str):
    """
    Делит фичи и таргет, кодирует y в числа.
    Возвращает X, y_encoded, label_encoder, num_cols, cat_cols
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    num_cols, cat_cols = split_features_by_type(X, X.columns.tolist())
    return X, y_encoded, le, num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Строит препроцессор для числовых и категориальных фич."""
    numeric_transformer = StandardScaler(with_mean=False)
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols),
        ],
        remainder='drop'
    )


def transformed_name_maps(preprocessor: ColumnTransformer) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Возвращает полные и базовые имена признаков после трансформации."""
    names = preprocessor.get_feature_names_out()
    full_map, base_map = {}, {}

    for name in names:
        if name.startswith('num__'):
            orig = name.split('num__', 1)[1]
            full_map[name] = orig
            base_map[name] = orig
        elif name.startswith('cat__'):
            tail = name.split('cat__', 1)[1]
            if '_' in tail:
                base, val = tail.split('_', 1)
                full_map[name] = f"{base}_{val}"
                base_map[name] = base
            else:
                full_map[name] = tail
                base_map[name] = tail
        else:
            full_map[name] = name
            base_map[name] = name
    return full_map, base_map

# =========================
# Обучение модели
# =========================
def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    C: float = 1.0,
    penalty: str = "l2",
    class_weight: Optional[str] = None,
    max_iter: int = 1000,
    label_encoder: Optional[LabelEncoder] = None
) -> Tuple[Pipeline, Dict]:
    """Тренирует пайплайн препроцессор + логистическая регрессия."""
    feature_cols = list(X_train.columns)
    num_cols, cat_cols = split_features_by_type(X_train, feature_cols)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    clf = LogisticRegression(
        C=C,
        penalty=penalty,
        solver='liblinear',
        max_iter=max_iter,
        class_weight=class_weight
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])
    model.fit(X_train, y_train)

    full_map, base_map = transformed_name_maps(preprocessor)

    meta = {
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "transformed_names": preprocessor.get_feature_names_out(),
        "feature_full_map": full_map,
        "feature_base_map": base_map,
        "label_encoder": label_encoder
    }
    return model, meta

# =========================
# Оценка модели
# =========================
def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    meta: Dict,
    threshold: float = 0.5
):
    """Считает метрики и согласует типы y_true и y_pred."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_int = (y_proba >= threshold).astype(int)

    le = meta.get("label_encoder", None)
    # Определяем, нужно ли вернуть y_pred к строкам
    if le is not None and np.array(y_test).dtype.kind in ("U", "S", "O"):
        y_pred = le.inverse_transform(y_pred_int)
    else:
        y_pred = y_pred_int

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(
            le.transform(y_test) if le and np.array(y_test).dtype.kind in ("U", "S", "O") else y_test,
            y_proba
        )
    }

    fpr, tpr, _ = roc_curve(
        le.transform(y_test) if le and np.array(y_test).dtype.kind in ("U", "S", "O") else y_test,
        y_proba
    )
    precision, recall, _ = precision_recall_curve(
        le.transform(y_test) if le and np.array(y_test).dtype.kind in ("U", "S", "O") else y_test,
        y_proba
    )

    return metrics, (fpr, tpr), (precision, recall)


# =========================
# Importance & interpretation
# =========================
def compute_feature_importance(model: Pipeline, meta: Dict) -> pd.DataFrame:
    """
    Агрегирует важности с учётом one-hot: суммируем коэффициенты для исходного признака.
    Использует feature_base_map для группировки и feature_full_map для точного соответствия.
    """
    # Достаём обученную логистическую регрессию
    lr = model.named_steps['clf']
    coefs = lr.coef_[0]  # Соответствует порядку preprocessor.get_feature_names_out()

    # Имена трансформированных фич
    tnames = meta["transformed_names"]
    base_map = meta["feature_base_map"]   # базовые имена (для агрегации)
    full_map = meta["feature_full_map"]   # полные имена (если нужно вывести развёрнуто)

    # Таблица коэффициентов
    df = pd.DataFrame({
        "Transformed": tnames,
        "Coefficient": coefs,
        "OriginalFeature": [base_map[name] for name in tnames],
        "FullFeatureName": [full_map[name] for name in tnames]
    })

    # Группируем по исходному (базовому) признаку
    agg = df.groupby("OriginalFeature").agg(
        Coefficient=("Coefficient", "sum"),
        AbsCoefficient=("Coefficient", lambda x: np.sum(np.abs(x)))
    ).reset_index()

    # Знак влияния
    agg["Sign"] = np.where(
        agg["Coefficient"] > 0, "Положительное",
        np.where(agg["Coefficient"] < 0, "Отрицательное", "Слабое")
    )

    # Сортировка по силе влияния
    agg = agg.sort_values("AbsCoefficient", ascending=False)

    return agg.rename(columns={"OriginalFeature": "Feature"})


def plot_feature_importance(importance_df: pd.DataFrame):
    """
    Горизонтальный barplot по абсолютной важности с цветом по знаку суммарного влияния.
    """
    fig = px.bar(
        importance_df,
        x="AbsCoefficient",
        y="Feature",
        color="Sign",
        orientation="h",
        color_discrete_map={"Положительное": "blue", "Отрицательное": "red", "Слабое": "gray"},
        title="Влияние признаков на целевую переменную",
        hover_data={"Coefficient": True, "AbsCoefficient": True}
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def interpret_feature_importance(importance_df: pd.DataFrame, top_n: int = 3) -> str:
    """
    Краткий текст: какие признаки повышают/снижают шанс целевого события.
    """
    top = importance_df.head(max(top_n, 1))
    top_pos = top[top["Coefficient"] > 0]["Feature"].tolist()
    top_neg = top[top["Coefficient"] < 0]["Feature"].tolist()

    parts = []
    if top_pos:
        parts.append(f"Больше всего повышают вероятность: {', '.join(top_pos)}.")
    if top_neg:
        parts.append(f"Больше всего снижают вероятность: {', '.join(top_neg)}.")
    if not parts:
        parts.append("Сильных влияющих признаков не обнаружено в топе.")

    return " ".join(parts)


def make_roc_fig(fpr, tpr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(dash='dash')))
    fig.update_layout(title="ROC-кривая", xaxis_title="FPR", yaxis_title="TPR")
    return fig


def make_pr_fig(precision, recall):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR'))
    fig.update_layout(title="Precision-Recall кривая", xaxis_title="Recall", yaxis_title="Precision")
    return fig


# =========================
# Single prediction & validation
# =========================
def validate_and_prepare_single_input(
    df: pd.DataFrame,
    feature_cols: List[str],
    user_input: Dict[str, object]
) -> Tuple[Optional[pd.DataFrame], Dict[str, str]]:
    """
    Проверяет значения и формирует DataFrame из одного объекта.
    Возвращает (X_input_df, errors). Если ошибки есть — X_input_df = None.
    """
    errors = {}
    row = {}

    for feat in feature_cols:
        if feat not in user_input:
            errors[feat] = "Поле отсутствует."
            continue

        val = user_input[feat]
        series = df[feat]

        if pd.api.types.is_numeric_dtype(series):
            try:
                if val is None or (isinstance(val, str) and val.strip() == ""):
                    errors[feat] = "Числовое значение не задано."
                else:
                    row[feat] = float(val)
            except Exception:
                errors[feat] = f"Ожидалось число, получено: {val}"
        else:
            allowed = pd.Series(series.dropna().unique()).astype(str).tolist()
            sval = str(val)
            if sval not in allowed:
                errors[feat] = f"Недопустимая категория: {sval}. Допустимые: {', '.join(allowed[:20])}" + (" ..." if len(allowed) > 20 else "")
            else:
                row[feat] = sval

    if errors:
        return None, errors

    return pd.DataFrame([row], columns=feature_cols), {}


def predict_with_explanation(
    model: Pipeline,
    meta: Dict,
    X_input_df: pd.DataFrame,
    threshold: float = 0.5,
    top_k: int = 3
) -> Dict[str, object]:
    """
    Предсказание для одного объекта с объяснением на уровне исходных признаков.
    Агрегирует вклады one-hot компонент к базовым фичам.
    """
    # 1) Вероятность и класс (int)
    proba = float(model.predict_proba(X_input_df)[0, 1])
    pred_class_int = int(proba >= threshold)

    # 2) Восстановим «человеческую» метку класса, если есть LabelEncoder
    le = meta.get("label_encoder")
    pred_class = le.inverse_transform([pred_class_int])[0] if le is not None else pred_class_int

    # 3) Вклады в трансформированном пространстве
    preproc = model.named_steps['preprocessor']
    lr = model.named_steps['clf']

    x_trans = preproc.transform(X_input_df)
    if hasattr(x_trans, "toarray"):
        x_vec = x_trans.toarray()[0]
    else:
        x_vec = np.asarray(x_trans)[0]

    w = lr.coef_[0]
    contrib_transformed = x_vec * w  # вклад каждой трансформированной фичи

    # 4) Агрегация вкладов к исходным признакам
    tnames = meta["transformed_names"]
    base_map = meta["feature_base_map"]  # 'cat__gender_F' -> 'gender'

    grouped = {}
    for val, tname in zip(contrib_transformed, tnames):
        base = base_map[tname]
        grouped[base] = grouped.get(base, 0.0) + float(val)

    # 5) Топ влияющих признаков
    top = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    influence_text = ", ".join([f"{feat} ({'+' if v > 0 else ''}{v:.2f})" for feat, v in top])

    explanation = f"Решение объясняется вкладом признаков: {influence_text}."
    return {
        "pred_class": pred_class,          # строковая метка (если есть LabelEncoder) или 0/1
        "pred_class_int": pred_class_int,  # числовой класс
        "proba": proba,
        "explanation": explanation,
        "top_contributions": top
    }


# =========================
# Reporting & export
# =========================
def generate_markdown_report(
    target_col: str,
    metrics: Dict[str, float],
    importance_df: pd.DataFrame,
    threshold: float,
    model_params: Dict[str, object],
    top_n: int = 10
) -> str:
    lines = []
    lines.append(f"# Отчёт по модели (логистическая регрессия)")
    lines.append(f"- Целевая переменная: {target_col}")
    lines.append(f"- Порог классификации: {threshold:.2f}")
    lines.append(f"- Параметры модели: " + ", ".join([f"{k}={v}" for k, v in model_params.items()]))
    lines.append("")
    lines.append("## Метрики")
    for k, v in metrics.items():
        lines.append(f"- {k}: {v:.4f}")
    lines.append("")
    lines.append("## Топ признаков")
    top = importance_df.head(top_n)
    for _, r in top.iterrows():
        lines.append(f"- {r['Feature']}: coef={r['Coefficient']:.4f}, |coef|={r['AbsCoefficient']:.4f}, знак={r['Sign']}")
    return "\n".join(lines)


def serialize_model(model: Pipeline) -> bytes:
    return pickle.dumps(model)


def summarize_dataset_for_ai(
    target_col: str,
    metrics: Dict[str, float],
    importance_df: pd.DataFrame,
    top_n: int = 5
) -> str:
    """Краткая сводка для фиксации в ИИ: только ключевые моменты."""
    top_feats = ", ".join(importance_df.head(top_n)["Feature"].tolist())
    parts = [
        f"Целевая переменная: {target_col}",
        "Метрики: " + ", ".join([f"{k}={v:.3f}" for k, v in metrics.items()]),
        f"Топ-{top_n} признаков по важности: {top_feats}"
    ]
    return " | ".join(parts)



def get_ns(name: str) -> dict:
    """Гарантирует наличие неймспейса в session_state и возвращает его."""
    if name not in st.session_state:
        st.session_state[name] = {}
    return st.session_state[name]


def ensure_modeling_state(
    df: pd.DataFrame,
    default_target: Optional[str] = None,
    show_warning: bool = True
) -> dict:
    """
    Инициализирует и поддерживает 'липкое' состояние страницы моделирования.
    - Хранит выбранный таргет в st.session_state['modeling_state']['target'].
    - Не сбрасывает выбор при переходах между страницами.
    - Если столбец пропал, выбирает безопасный фолбэк и (опционально) предупреждает.
    """
    ms = get_ns("modeling_state")

    # Подпись данных (для мягкой проверки смены набора)
    df_sig = (tuple(df.columns), df.shape)
    dataset_changed = ms.get("df_sig") != df_sig
    if dataset_changed:
        ms["df_sig"] = df_sig
        # Если текущий таргет пропал — восстанавливаем
        if ms.get("target") not in df.columns:
            if show_warning and ms.get("target") is not None:
                st.warning(
                    f"Ранее выбранный таргет '{ms['target']}' отсутствует в текущем датасете. "
                    f"Выбрано значение по умолчанию."
                )
            ms["target"] = None
            ms["dirty"] = True

    # Первичная инициализация таргета
    if not ms.get("target"):
        if default_target and default_target in df.columns:
            ms["target"] = default_target
        elif len(df.columns) > 0:
            ms["target"] = df.columns[0]

    # Флаг 'конфиг изменён'
    if "dirty" not in ms:
        ms["dirty"] = False

    return ms


def sticky_selectbox(
    ns: str,
    key: str,
    label: str,
    options: List[Any],
    ui_key: Optional[str] = None,
    help: Optional[str] = None,
    format_func=None,
    index_fallback: int = 0,
) -> Tuple[Any, bool]:
    """
    Липкий selectbox: читает/пишет значение в session_state[ns][key],
    не сбрасывается при навигации. Возвращает (value, changed).
    """
    ns_dict = get_ns(ns)

    if not options:
        st.error("Нет доступных опций для выбора.")
        return None, False

    saved = ns_dict.get(key, options[index_fallback] if options else None)
    idx = options.index(saved) if saved in options else index_fallback

    if ui_key is None:
        ui_key = f"{ns}_{key}_ui"

    # Если format_func не задан, даём безопасный по умолчанию
    fmt = format_func if callable(format_func) else lambda x: x

    value = st.selectbox(
        label,
        options,
        index=idx,
        key=ui_key,
        help=help,
        format_func=fmt,
    )

    changed = value != saved
    if changed:
        ns_dict[key] = value
        ns_dict["dirty"] = True

    return value, changed


def mark_model_trained():
    """Сбрасывает флаг 'dirty' после успешного обучения."""
    ms = get_ns("modeling_state")
    ms["dirty"] = False



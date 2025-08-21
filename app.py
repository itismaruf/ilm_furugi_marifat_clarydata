import streamlit as st
import pandas as pd
import os
import time
import plotly.express as px
import joblib
import io
from sklearn.model_selection import train_test_split


from Utils.upload_utils import load_data, get_base_info, display_preview, display_base_info, interpret_with_ai
from Utils.automatic_data_processing import run_auto_cleaning, summarize_missing, run_auto_cleaning, \
        drop_rows_na, drop_cols_na, fill_na, render_nan_handling_info, render_nan_rules_table, drop_selected_cols, show_na_summary

from Utils.outlier_utils import render_outlier_handling_info, detect_outliers_iqr, detect_outliers_zscore, \
    plot_outliers_distribution, outliers_summary, run_auto_outlier_removal, render_outlier_rules_table, \
    remove_outliers_iqr, remove_outliers_zscore, cap_outliers, remove_outliers_percentile, plot_outlier_removal_comparison

from Utils.visualization import plot_data_visualizations, suggest_visualization_combinations, plot_correlation_heatmap, generate_pivot_table

from Utils.modeling_utils import *

from Utils.chat import *

from AI_helper import (
    get_chatgpt_response, update_context, send_correlation_to_ai, send_pivot_to_ai, chat_only, notify_ai_dataset_structure, reset_ai_conversation
)

# Конфигурация страницы
st.set_page_config(layout="wide")


# === Заставка ===
if "app_loaded" not in st.session_state:
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

            html, body {
                margin: 0;
                padding: 0;
                height: 100%;
                width: 100%;
                font-family: 'Inter', sans-serif;
                overflow: hidden;
            }

            .splash-container {
                position: fixed;
                top: 0; left: 0;
                width: 100vw;
                height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
                color: #0f172a;
                z-index: 9999;
                animation: fadeIn 1s ease-in-out;
                transition: opacity 1s ease-out;
            }

            .splash-container.fade-out {
                opacity: 0;
                pointer-events: none;
            }

            .ai-emoji {
                font-size: 3.2em;
                margin-bottom: 20px;
                animation: pulse 2s infinite;
            }

            .splash-title {
                font-size: 2.4em;
                font-weight: 700;
                text-align: center;
                opacity: 0;
                animation: fadeUp 1.2s ease-out forwards;
                animation-delay: 0.4s;
            }

            .splash-subtext {
                font-size: 1em;
                margin-top: 12px;
                color: #475569;
                opacity: 0;
                animation: fadeUp 1.4s ease-out forwards;
                animation-delay: 0.8s;
                text-align: center;
                max-width: 600px;
                padding: 0 16px;
            }

            .splash-footer {
                position: absolute;
                bottom: 18px;
                font-size: 0.8em;
                color: #64748b;
                text-align: center;
            }

            @keyframes fadeUp {
                0% { opacity: 0; transform: translateY(20px); }
                100% { opacity: 1; transform: translateY(0); }
            }

            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            @keyframes pulse {
                0%, 100% {
                    transform: scale(1);
                    opacity: 1;
                }
                50% {
                    transform: scale(1.15);
                    opacity: 0.75;
                }
            }
        </style>

        <div class="splash-container" id="splash">
            <div class="ai-emoji">✨</div>
            <div class="splash-title">ClariData</div>
            <div class="splash-subtext">Интеллектуальная система анализа данных<br>с автоочисткой, визуализацией, предсказаниями и пояснениями</div>
            <div class="splash-footer">© Created by Rahimov M.A.</div>
        </div>

        <script>
            setTimeout(() => {
                const splash = document.getElementById("splash");
                if (splash) splash.classList.add("fade-out");
            }, 3000);
        </script>
    """, unsafe_allow_html=True)

    time.sleep(4)
    st.session_state.app_loaded = True
    st.rerun()


# --- Установка API-ключа из секретов, если есть ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# --- Инициализация первой страницы при запуске ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'Загрузка данных'

st.markdown("""
    <style>
        /* Когда сайдбар открыт (aria-expanded="true"), основной контент смещается вправо */
        [data-testid="stSidebar"][aria-expanded="true"] ~ .main .block-container {
            margin-left: 300px;
            transition: margin-left 0.3s ease;
        }
        /* Когда сайдбар свернут (aria-expanded="false"), основной контент возвращается в исходное положение */
        [data-testid="stSidebar"][aria-expanded="false"] ~ .main .block-container {
            margin-left: 1rem;
            transition: margin-left 0.3s ease;
        }
    </style>
""", unsafe_allow_html=True)

if "_ai_session_inited" not in st.session_state:
    reset_ai_conversation()                 # сброс глобальной истории для этой сессии
    st.session_state["_ai_session_inited"] = True

# --- Функция переключения страниц ---
def set_page(page_name):
    st.session_state['page'] = page_name

# Сброс истории чата при запуске приложения
if "chat_history" in st.session_state:
    del st.session_state["chat_history"]
if "chat_initialized" in st.session_state:
    del st.session_state["chat_initialized"]

# --- Сайдбар с навигацией и стилем кнопок ---
st.sidebar.header("🔧 Навигация")
pages = {
    "Загрузка данных": "📥",
    "Автообработка данных": "⚙️",
    "Обработка выбросов": "🚩",
    "Визуальный анализ (EDA)": "📊",
    "Моделирование и предсказание": "📟",
    "Разъяснение результатов (с ИИ)": "💬",
    "Руководство пользователя": "📝"
}

# Настройка CSS для кнопок (цвета при наведении)
st.markdown("""
    <style>
        div.stButton > button {
            background-color: #f0f2f6;
            color: black;
            border: 1px solid #ccc;
            border-radius: 6px;
        }
        div.stButton > button:hover {
            background-color: #e0f0ff;
            color: #007BFF;
            border: 1px solid #007BFF;
        }
    </style>
""", unsafe_allow_html=True)

# Навигационные кнопки
for name, icon in pages.items():
    st.sidebar.button(f"{icon} {name}", on_click=set_page, args=(name,))

# Сброс истории чата при запуске приложения
if "chat_history" in st.session_state:
    del st.session_state["chat_history"]
if "chat_initialized" in st.session_state:
    del st.session_state["chat_initialized"]


# Кнопка для очистки всех данных
if st.sidebar.button("🔄 Очистить всё"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ===================== СТРАНИЦЫ =======================

if st.session_state['page'] == "Загрузка данных":
    st.caption('💡 Если вы не пользовались ClaryData, можете перейте в раздел "Руководство пользователя".')
    st.title("📥 Загрузка данных")

    # --- Загрузка данных ---
    if "df" not in st.session_state:
        uploaded_file = st.file_uploader("", type=["csv", "xlsx", "xls"])
        if not uploaded_file:
            st.info("⬆ Загрузите файл для анализа.", icon="📁")
        else:
            try:
                df = load_data(uploaded_file)
                st.session_state["df"] = df
                st.success("Данные успешно загружены", icon="✅")
            except Exception as e:
                st.error(f"Ошибка при обработке данных: {e}", icon="🚫")
    else:
        df = st.session_state["df"]
        st.success("Данные уже загружены ✅")

    # --- Если данные загружены ---
    if "df" in st.session_state:
        st.markdown("---")

        # Превью и метрики
        display_preview(df)
        base_info = get_base_info(df)
        display_base_info(base_info)

        # — Инициализация/обновление краткого summary (безопасно) —
        data_sig = (tuple(df.columns), df.shape)
        if st.session_state.get("_data_sig") != data_sig:
            # датасет новый или изменился — пересобираем summary
            summary = f"{df.shape[0]} строк, {df.shape[1]} столбцов; признаки: {', '.join(map(str, df.columns))}"
            st.session_state["_data_sig"] = data_sig
            st.session_state["data_summary"] = summary
            try:
                update_context("data_summary", summary)
            except Exception:
                pass
        else:
            # датасет тот же — берем сохранённое или формируем на лету
            summary = st.session_state.get(
                "data_summary",
                f"{df.shape[0]} строк, {df.shape[1]} столбцов; признаки: {', '.join(map(str, df.columns))}"
            )

        st.markdown("---")

        st.markdown("### Подключение ИИ")
        st.caption("Нажмите на кнопку ниже, чтобы позволить ИИ подключиться к анализу, получая нужную информацию о ваших данных.")

        if st.button("🔌 Подключить ИИ к анализу"):
            msg = notify_ai_dataset_structure(df)
            st.success(msg)

        st.markdown("---")

        # Поле для описания цели анализа
        user_desc = st.text_area(
            "📝 Уточните задачу анализа, чтобы ИИ мог более точно адаптировать свою помощь",
            placeholder="Например: Хочу проанализировать, как меняются цены на жильё по регионам",
            value=st.session_state.get("analysis_goal", ""),
            height=100
        )

        # Кнопка для интерпретации
        if st.button("✨Получить интерпретацию от AI"):
            if not user_desc.strip():
                st.warning("Пожалуйста, опишите цель анализа.")
            else:
                st.session_state["analysis_goal"] = user_desc
                ai_response = interpret_with_ai(
                    data_summary=summary,  # <-- используем локальную summary
                    user_desc=user_desc,
                    df=df,
                    get_ai_fn=get_chatgpt_response
                )
                st.session_state["ai_interpretation"] = ai_response




# === Автообработка данных ===
if st.session_state.get("page") == "Автообработка данных":
    st.title("⚙️ Автообработка данных")

    # Инициализация флага изменений
    if "data_changed" not in st.session_state:
        st.session_state["data_changed"] = False

    if "df" not in st.session_state:
        st.warning("📥 Загрузите данные", icon="⚠️")
    else:
        df = st.session_state["df"]

        # ℹ️ Краткая инструкция
        render_nan_handling_info()

        # 🎯 Выбор целевой переменной
        target = st.selectbox(
            "Целевая переменная (ее NaN будут удалены)",
            [None] + list(df.columns)
        )

        st.markdown("---")

        # 📊 Статистика пропусков
        st.subheader("📊 Пропуски в данных")
        missing = summarize_missing(df)

        if missing.empty:
            st.success("Нет пропусков в данных", icon="✅")
        else:
            st.table(
                missing
                .rename(columns={
                    "column": "Столбец",
                    "missing_count": "Кол-во",
                    "pct_missing": "% пропусков"
                })
                .set_index("Столбец")
            )

            st.markdown("---")

            # 🤖 Автоочистка
            st.subheader("🤖 Автоочистка")
            with st.expander("📌 Правила автоочистки"):
                render_nan_rules_table()

            if st.button("🚀 Запустить автоочистку"):
                before, log, new_df = run_auto_cleaning(df, target_col=target)
                st.session_state["df"] = new_df
                st.session_state["data_changed"] = True  # <-- Фиксируем изменения

                if before.empty:
                    st.info("Пропусков не найдено", icon="✅")
                else:
                    st.markdown("**До очистки**")
                    st.table(
                        before
                        .rename(columns={
                            "column": "Столбец",
                            "missing_count": "Кол-во",
                            "pct_missing": "% пропусков"
                        })
                        .set_index("Столбец")
                    )

                    with st.spinner("Автоочистка..."):
                        time.sleep(1)

                    report = (
                        pd.DataFrame(log)
                        .rename(columns={
                            "column": "Столбец",
                            "missing_count": "Кол-во",
                            "pct_missing": "% пропусков",
                            "action": "Действие"
                        })
                        .set_index("Столбец")
                    )
                    st.markdown("**Отчет автоочистки**")
                    st.table(report)

                    remaining = new_df.isna().sum().sum()
                    st.success(f"Готово! Осталось пропусков: {remaining}")

        st.markdown("---")

        # 🔧 Ручная очистка
        st.subheader("🔧 Ручная очистка")
        with st.expander("✍️ Панель ручной очистки"):
            cols = st.multiselect(
                "Столбцы для обработки:",
                [c for c in df.columns if c != target]
            )
            action = st.radio(
                "Действие:",
                ["Удалить строки", "Удалить столбцы (с NaN)", "Заполнить NaN", "Удалить выбранные столбцы"]
            )
            show_tables = st.checkbox("Показывать сводку по NaN", value=True)

            method = value = None
            if action == "Заполнить NaN":
                method = st.selectbox("Метод заполнения:", ["mean", "median", "mode", "constant"])
                if method == "constant":
                    value = st.text_input("Значение для заполнения:")

            if st.button("✅ Применить"):
                before = df.copy()

                if action == "Удалить строки":
                    new_df = drop_rows_na(df, cols, target)
                elif action == "Удалить столбцы (с NaN)":
                    new_df = drop_cols_na(df, cols)
                elif action == "Удалить выбранные столбцы":
                    new_df = drop_selected_cols(df, cols)
                elif action == "Заполнить NaN":
                    new_df = fill_na(df, cols, method, value)

                st.session_state["df"] = new_df
                st.session_state["data_changed"] = True  # <-- Фиксируем изменения
                st.success("✅ Обработка завершена")

                if show_tables and action != "Удалить выбранные столбцы":
                    show_na_summary(before, new_df, cols)
                elif show_tables and action == "Удалить выбранные столбцы":
                    st.markdown("**Размер до/после (строки, столбцы)**")
                    col1, col2 = st.columns(2)
                    col1.write(before.shape)
                    col2.write(new_df.shape)

        # === 📥 Кнопка скачивания, если были изменения ===
        if st.session_state.get("data_changed", False) and not st.session_state["df"].empty:
            st.markdown("---")
            st.subheader("📥 Скачать обработанные данные")

            # Получаем имя исходного файла, если оно есть
            base_name = "data"
            if "original_filename" in st.session_state:
                base_name = os.path.splitext(st.session_state["original_filename"])[0]

            file_name = f"{base_name}_cleaned.csv"

            # Готовим CSV в буфере
            csv_buffer = io.BytesIO()
            st.session_state["df"].to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)  # <-- ВАЖНО: сброс указателя в начало

            st.success("✅ Файл готов к скачиванию")
            st.download_button(
                label=f"💾 Скачать {file_name}",
                data=csv_buffer,
                file_name=file_name,
                mime="text/csv"
            )


# === Обработка выбросов ===
if st.session_state.get("page") == "Обработка выбросов":
    st.title("🚩 Обработка выбросов")

        # Инициализация флага изменений
    if "data_changed" not in st.session_state:
        st.session_state["data_changed"] = False

    if "df" not in st.session_state:
        st.warning("📥 Загрузите данные на предыдущей странице", icon="⚠️")
    else:
        df = st.session_state["df"]
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # Инструкция
        render_outlier_handling_info()
        st.markdown("---")

    if "df" not in st.session_state:
        st.warning("📥 Загрузите данные на предыдущей странице", icon="⚠️")
    else:
        df = st.session_state["df"]
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # Инструкция
        render_outlier_handling_info()
        st.markdown("---")

        # Анализ и визуализация выбросов
        st.subheader("🔍 Анализ выбросов")
        with st.expander("👁 Просмотреть распределение выбросов"):
            cols_viz = st.multiselect(
                "Выберите числовые столбцы для анализа",
                numeric_cols,
                key="out_viz_cols"
            )
            method_viz = st.radio(
                "Метод обнаружения выбросов",
                ["IQR-метод", "Z-score"],
                key="out_viz_method"
            )

            if method_viz == "IQR-метод":
                q_low, q_high = st.slider(
                    "Квантили для IQR",
                    0.0, 0.5, (0.25, 0.75),
                    step=0.05,
                    key="iqr_viz"
                )
            else:
                z_thresh = st.number_input(
                    "Порог Z-score",
                    min_value=1.0, max_value=5.0,
                    value=3.0, step=0.1,
                    key="z_viz"
                )

            if st.button("👁 Показать выбросы", key="show_out_viz"):
                masks = (detect_outliers_iqr(df, cols_viz, q_low, q_high)
                         if method_viz == "IQR-метод"
                         else detect_outliers_zscore(df, cols_viz, z_thresh))
                fig = plot_outliers_distribution(df, masks, cols_viz)
                st.plotly_chart(fig, use_container_width=True)

                summary = outliers_summary(df, masks)
                st.table(summary.set_index("column"))

        st.markdown("---")

        # Автоочистка выбросов
        st.subheader("🤖 Автообработка выбросов")
        with st.expander("📌 Правила автообработки выбросов"):
            render_outlier_rules_table()

        if st.button("🚀 Запустить автоочистку выбросов", key="auto_out"):
            before, log, cleaned_df = run_auto_outlier_removal(df)
            st.session_state["df"] = cleaned_df

            total_removed = sum(item["removed_count"] for item in log)
            if total_removed == 0:
                st.info("Автоматически выбросы не найдены", icon="✅")
            else:
                report = (
                    pd.DataFrame(log)
                      .rename(columns={
                          "column": "Столбец",
                          "method": "Метод",
                          "removed_count": "Удалено выбросов"
                      })
                      .set_index("Столбец")
                )
                st.markdown("**Отчет автоочистки выбросов**")
                st.table(report)
                st.success(f"Удалено выбросов: {total_removed}")

                st.markdown("### Сравнение распределений до и после автоочистки")
                fig_cmp = plot_outlier_removal_comparison(df, cleaned_df, numeric_cols)
                st.plotly_chart(fig_cmp, use_container_width=True)

        st.markdown("---")

        st.subheader("🔧 Ручная очистка выбросов")
        with st.expander("✍️ Панель ручной очистки выбросов", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                cols_manual = st.multiselect(
                    "Столбцы для обработки",
                    numeric_cols,
                    key="out_manual_cols"
                )
            with col2:
                method_manual = st.selectbox(
                    "Метод обработки",
                    [
                        "Удалить выбросы (IQR)",
                        "Каппинг (IQR-границы)",
                        "Удаление по Z-score",
                        "Удаление по процентилям"
                    ],
                    key="out_manual_method"
                )

            # Параметры для каждого метода
            if method_manual in ("Удалить выбросы (IQR)", "Каппинг (IQR-границы)"):
                low_q, high_q = st.slider(
                    "Квантили для IQR",
                    0.0, 0.5, (0.25, 0.75),
                    step=0.05,
                    key="iqr_manual"
                )
            elif method_manual == "Удаление по Z-score":
                z_manual = st.number_input(
                    "Порог Z-score",
                    min_value=1.0, max_value=5.0,
                    value=3.0, step=0.1,
                    key="z_manual"
                )
            else:  # Удаление по процентилям
                p_low, p_high = st.slider(
                    "Процентили для удаления",
                    0, 100, (5, 95),
                    step=1,
                    key="percentile_manual"
                )

            if st.button("✅ Применить ручную очистку"):
                before_manual = df.copy()
                cleaned_manual = df.copy()

                for col in cols_manual:
                    if method_manual == "Удалить выбросы (IQR)":
                        cleaned_manual = remove_outliers_iqr(cleaned_manual, [col], low_q, high_q)
                    elif method_manual == "Каппинг (IQR-границы)":
                        cleaned_manual = cap_outliers(cleaned_manual, [col], low_q, high_q)
                    elif method_manual == "Удаление по Z-score":
                        cleaned_manual = remove_outliers_zscore(cleaned_manual, [col], z_manual)
                    else:
                        cleaned_manual = remove_outliers_percentile(cleaned_manual, [col], p_low, p_high)

                st.session_state["df"] = cleaned_manual
                st.success("✅ Ручная очистка выбросов завершена")

                # show_outlier_summary(before_manual, cleaned_manual, cols_manual)

                st.markdown("### Сравнение распределений до и после ручной очистки")
                fig_cmp_manual = plot_outlier_removal_comparison(
                    before_manual, cleaned_manual, cols_manual
                )
                st.plotly_chart(fig_cmp_manual, use_container_width=True)

        # === 📥 Кнопка скачивания, если были изменения ===
        if st.session_state.get("data_changed", False) and not st.session_state["df"].empty:
            st.markdown("---")
            st.subheader("📥 Скачать обработанные данные")

            base_name = "data"
            if "original_filename" in st.session_state:
                base_name = os.path.splitext(st.session_state["original_filename"])[0]
            file_name = f"{base_name}_cleaned.csv"

            csv_buffer = io.BytesIO()
            st.session_state["df"].to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            st.success("✅ Файл готов к скачиванию")
            st.download_button(
                label=f"💾 Скачать {file_name}",
                data=csv_buffer,
                file_name=file_name,
                mime="text/csv"
            )


# === Визуализация ===
elif st.session_state["page"] == "Визуальный анализ (EDA)":
    st.title("📊 Визуальный анализ (EDA)")

    if "df" not in st.session_state:
        st.warning("📥 Сначала загрузите данные.", icon="⚠️")
    else:
        df = st.session_state["df"]
        st.markdown("---")

        # === 🔖 Вкладки ===
        tabs = st.tabs(["📈 Графики", "❄️ Корреляции", "📊 Сводные таблицы"])

        # === 📈 Вкладка: Графики ===
        with tabs[0]:
            st.subheader("🧭 Выбор переменных")
            st.markdown("Выберите переменные, которые вы хотите визуализировать по осям X и Y")

            x = st.selectbox(
                "🟥 Ось X",
                df.columns,
                index=st.session_state.get("eda_x_index", 0),
                key="eda_x"
            )
            y_options = [""] + list(df.columns)
            y = st.selectbox(
                "🟦 Ось Y (необязательно)",
                y_options,
                index=st.session_state.get("eda_y_index", 0),
                key="eda_y"
            ) or None

            st.session_state["eda_x_index"] = list(df.columns).index(x)
            st.session_state["eda_y_index"] = y_options.index(y if y else "")

            if x == y and y is not None:
                st.warning("Переменные X и Y не должны совпадать.")
                y = None

            # === Кнопка построения графика
            build_chart = False
            if x:
                build_chart = st.button("📊 Построить график", key="build_chart")
                st.info("📉 График появится ниже после нажатия кнопки.")
            else:
                st.info("Пожалуйста, выберите хотя бы переменную для оси X.")

            st.markdown("---")

            # === AI-подсказки
            with st.expander("💡 Получить советы для визуализации от ИИ по X и Y"):
                if st.button("✨ Предложи комбинации", key="suggest_combinations"):
                    df_info = f"Переменные: {', '.join(df.columns)}"
                    with st.spinner("Генерируем рекомендации..."):
                        time.sleep(2)
                        suggestion = suggest_visualization_combinations(df_info)
                    st.markdown("**📝 Рекомендации от ИИ:**")
                    st.info(suggestion, icon="🤖")

            st.markdown("---")

            # === Тип графика
            st.subheader("🎨 Тип графика")
            chart_options = [
                "Автоматически", "Гистограмма", "Круговая диаграмма",
                "Точечный график", "Boxplot", "Bar-график", "Лайнплот"
            ]
            chart_type = st.selectbox(
                "Выберите график",
                chart_options,
                index=st.session_state.get("eda_chart_index", 0),
                key="eda_chart"
            )
            st.session_state["eda_chart_index"] = chart_options.index(chart_type)

            st.markdown("---")

            # === Фильтры
            with st.expander("🔍 Фильтры по числовым переменным"):
                filters = {}
                cols_to_filter = [x] + ([y] if y else [])
                for col in dict.fromkeys(cols_to_filter):
                    if pd.api.types.is_numeric_dtype(df[col]):
                        lo, hi = float(df[col].min()), float(df[col].max())
                        if lo == hi:
                            st.info(f"⚠️ Для «{col}» фильтр не применён: значения одинаковы ({lo})")
                            continue
                        sel = st.slider(
                            f"Фильтр по {col}",
                            min_value=lo,
                            max_value=hi,
                            value=st.session_state.get(f"slider_{col}", (lo, hi)),
                            key=f"slider_{col}"
                        )
                        filters[col] = sel

            with st.expander("📌 Показать только top-N категорий"):
                top_n = None
                limit_topn = st.checkbox(
                    "Ограничить top-N",
                    value=st.session_state.get("limit_topn", False),
                    key="limit_topn"
                )
                if limit_topn:
                    top_n = st.slider(
                        "N категорий",
                        3, 30,
                        st.session_state.get("top_n_slider", 10),
                        key="top_n_slider"
                    )

            st.markdown("---")
            st.subheader("📈 График")

            if build_chart:
                with st.spinner("Построение графика..."):
                    time.sleep(2.5)
                    fig = plot_data_visualizations(
                        df=df,
                        x=x,
                        y=y,
                        top_n=top_n,
                        numeric_filters=filters,
                        chart_type=chart_type
                    ) 

                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("🎯 Чтобы увидеть график, сначала выберите переменные выше и нажмите кнопку «Построить график».")



        # === ❄️ Вкладка: Корреляции ===
        with tabs[1]:
            st.subheader("❄️ Тепловая карта корреляций")
            fig = plot_correlation_heatmap(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 Чем ближе значение к 1 или -1, тем сильнее линейная связь между переменными.")

                if st.button("📤 Зафиксировать корреляции в ИИ", key="fix_corr"):
                    _ = send_correlation_to_ai(df)
                    st.session_state["correlation_saved"] = True
                    st.success("✅ Корреляции зафиксированы в ИИ.")
                elif st.session_state.get("correlation_saved"):
                    st.info("✅ Корреляции уже были зафиксированы.")
            else:
                st.info("Невозможно построить тепловую карту.")

        # === 📊 Вкладка: Сводные таблицы ===
        with tabs[2]:
            st.subheader("📊 Сводные таблицы (Pivot)")

            col1, col2 = st.columns(2)
            with col1:
                index_col = st.selectbox(
                    "Группировать по",
                    df.columns,
                    index=st.session_state.get("pivot_index_index", 0),
                    key="pivot_index"
                )
                st.session_state["pivot_index_index"] = list(df.columns).index(index_col)

            with col2:
                num_cols = df.select_dtypes(include='number').columns
                value_col = st.selectbox(
                    "Агрегировать",
                    num_cols,
                    index=st.session_state.get("pivot_value_index", 0),
                    key="pivot_value"
                )
                st.session_state["pivot_value_index"] = list(num_cols).index(value_col)

            agg_options = ["mean", "sum", "count"]
            agg_func = st.radio(
                "Метод агрегации",
                agg_options,
                index=st.session_state.get("pivot_agg_index", 0),
                horizontal=True,
                key="pivot_agg"
            )
            st.session_state["pivot_agg_index"] = agg_options.index(agg_func)

            pivot_table = generate_pivot_table(df, index_col, value_col, agg_func)
            if pivot_table is not None:
                st.dataframe(pivot_table, use_container_width=True)

                if st.button("📤 Зафиксировать в ИИ", key="fix_pivot"):
                    _ = send_pivot_to_ai(pivot_table, index_col, value_col, agg_func)
                    st.session_state["pivot_saved"] = True
                    st.success("✅ Сводная таблица зафиксирована в ИИ.")
                elif st.session_state.get("pivot_saved"):
                    st.info("✅ Сводная таблица уже была зафиксирована.")
            else:
                st.info("Возможно, вы выбрали одни и те же столбцы!")



if st.session_state.get("page") == "Моделирование и предсказание":
    st.title("🤖 Моделирование и предсказание")
    st.caption("Фокус: понять, как и почему признаки влияют на целевую переменную")

    if "df" not in st.session_state:
        st.warning("📥 Сначала загрузите данные.")
        st.stop()

    df: pd.DataFrame = st.session_state["df"]

    # ====== Липкое состояние страницы ======
    ms = ensure_modeling_state(df)

    # ====== Липкий выбор целевой ======
    options = list(df.columns)
    target_col, target_changed = sticky_selectbox(
        ns="modeling_state",          # неймспейс состояния
        key="target",                 # ключ внутри неймспейса
        label="🎯 Целевая переменная (binary target)",
        options=options,
        ui_key="modeling_target_ui"   # уникальный ключ UI
    )

    unique_target = pd.Series(df[target_col].dropna().unique())
    if len(unique_target) > 2:
        st.error(f"Целевая переменная должна быть бинарной (найдено уникальных значений: {len(unique_target)})")
        st.stop()

    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        st.error("Нет признаков для обучения (все поля — это Target).")
        st.stop()

    # ====== Настройки модели ======
    st.subheader("⚙️ Настройки модели")
    c1, c2 = st.columns(2)
    with c1:
        C_value = st.number_input("Параметр регуляризации C", 0.01, 100.0, 1.0, 0.01)
        penalty = st.selectbox("Тип регуляризации", ["l1", "l2"], index=1)
        use_class_weight = st.checkbox("Сбалансировать веса классов", value=False)
    with c2:
        threshold = st.slider("Порог классификации", 0.05, 0.95, 0.5, 0.05)
        test_size = st.slider("Размер тестовой выборки (%)", 10, 50, 20, 5) / 100
        max_iter = st.number_input("Макс. итераций", 100, 5000, 1000, 100)

    # ====== Обучение модели ======
    if st.button("🚀 Обучить / переобучить модель", use_container_width=True):
        try:
            X, y_encoded, le, num_cols, cat_cols = prepare_features_and_target(df, target_col)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            class_weight = "balanced" if use_class_weight else None
            model, meta = train_logistic_regression(
                X_train, y_train, C=C_value, penalty=penalty,
                class_weight=class_weight, max_iter=max_iter,
                label_encoder=le
            )

            metrics, roc_data, pr_data = evaluate_model(model, X_test, y_test, meta, threshold)
            importance_df = compute_feature_importance(model, meta)
            short_text = interpret_feature_importance(importance_df, top_n=3)

            st.session_state["modeling"] = {
                "model": model, "meta": meta,
                "threshold": threshold, "metrics": metrics,
                "roc": roc_data, "pr": pr_data,
                "importance_df": importance_df, "short_text": short_text,
                "target_col": target_col, "feature_cols": feature_cols,
                "params": {
                    "C": C_value, "penalty": penalty,
                    "class_weight": class_weight, "max_iter": max_iter,
                    "test_size": test_size
                }
            }

            # Сброс флага dirty — модель соответствует текущему таргету
            mark_model_trained()

            st.success("✅ Модель обучена и сохранена")
        except Exception as e:
            st.error(f"Не удалось обучить модель: {e}")

    # ====== Показ результатов, если модель есть ======
    if "modeling" in st.session_state:
        data = st.session_state["modeling"]

        st.subheader("📊 Результаты и анализ")
        with st.expander("Показать метрики и кривые", expanded=False):
            m_df = pd.DataFrame({
                "Метрика": list(data["metrics"].keys()),
                "Значение": [round(v, 4) for v in data["metrics"].values()]
            })
            st.dataframe(m_df, use_container_width=True, hide_index=True)
            fpr, tpr = data["roc"]
            precision, recall = data["pr"]
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(make_roc_fig(fpr, tpr), use_container_width=True)
            with c2:
                st.plotly_chart(make_pr_fig(precision, recall), use_container_width=True)

        st.subheader("📌 Важность признаков")
        st.plotly_chart(plot_feature_importance(data["importance_df"]), use_container_width=True)
        st.info(data["short_text"])

        with st.expander("📤 Зафиксировать метрики в ИИ", expanded=False):
            # Placeholder для вывода статуса/спиннера именно тут
            status_placeholder = st.empty()

            if st.button("Отправить", use_container_width=True):
                try:
                    status_placeholder.info("⏳ Отправляем данные в ИИ...")

                    summary = summarize_dataset_for_ai(
                        target_col=data["target_col"],
                        metrics=data["metrics"],
                        importance_df=data["importance_df"]
                    )

                    # Запоминаем в чатовой истории, чтобы ИИ знал эти данные в будущем диалоге
                    _ = chat_only(f"[FIX-MODEL-SUMMARY] {summary}")

                    status_placeholder.success("✅ Данные зафиксированы в ИИ")
                except Exception as e:
                    status_placeholder.error(f"Ошибка отправки: {e}")



        # 🔍 Прогноз
        st.subheader("🔍 Прогноз для одного объекта")
        num_cols, cat_cols = split_features_by_type(df, data["feature_cols"])

        # UI: строим словарь значений
        user_input = {}
        cols = st.columns(3) if len(data["feature_cols"]) >= 9 else (st.columns(2) if len(data["feature_cols"]) >= 4 else st.columns(1))

        # Порядок: числовые, затем категориальные, чтобы было удобнее
        all_feats = num_cols + cat_cols

        for i, feat in enumerate(all_feats):
            with cols[i % len(cols)]:
                series = df[feat]
                if pd.api.types.is_numeric_dtype(series):
                    vmin = float(series.min())
                    vmax = float(series.max())
                    vdefault = float(series.median())
                    user_input[feat] = st.number_input(
                        f"{feat}",
                        min_value=vmin if np.isfinite(vmin) else None,
                        max_value=vmax if np.isfinite(vmax) else None,
                        value=vdefault if np.isfinite(vdefault) else 0.0
                    )
                else:
                    options = pd.Series(series.dropna().unique()).astype(str).tolist()
                    options = sorted(options)[:300]  # safety
                    if not options:
                        options = ["(пусто)"]
                    user_input[feat] = st.selectbox(f"{feat}", options, index=0)

        if st.button("Сделать прогноз"):
            X_input_df, errors = validate_and_prepare_single_input(df, data["feature_cols"], user_input)
            if errors:
                for k, msg in errors.items():
                    st.warning(f"{k}: {msg}")
            else:
                try:
                    result = predict_with_explanation(
                        model=data["model"],
                        meta=data["meta"],
                        X_input_df=X_input_df,
                        threshold=data["threshold"],
                        top_k=3
                    )
                    st.success(f"Предсказанный класс: {result['pred_class']} (вероятность {result['proba']:.2f})")
                    st.write(result["explanation"])
                except Exception as e:
                    st.error(f"Не удалось получить прогноз: {e}")

        # ======================
        # Экспорт
        # ======================
        st.subheader("📦 Экспорт")
        cdl1, cdl2, cdl3, cdl4 = st.columns(4)

        with cdl1:
            try:
                model_bytes = serialize_model(data["model"])
                st.download_button(
                    "Скачать модель (.pkl)",
                    data=model_bytes,
                    file_name="logreg_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Экспорт модели: {e}")

        with cdl2:
            imp_csv = data["importance_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Скачать важности (CSV)",
                data=imp_csv,
                file_name="feature_importance.csv",
                mime="text/csv",
                use_container_width=True
            )

        with cdl3:
            metrics_df = pd.DataFrame([data["metrics"]])
            metr_csv = metrics_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Скачать метрики (CSV)",
                data=metr_csv,
                file_name="metrics.csv",
                mime="text/csv",
                use_container_width=True
            )

        with cdl4:
            md = generate_markdown_report(
                target_col=data["target_col"],
                metrics=data["metrics"],
                importance_df=data["importance_df"],
                threshold=data["threshold"],
                model_params=data["params"],
                top_n=10
            )
            st.download_button(
                "Скачать отчёт (MD)",
                data=md.encode("utf-8"),
                file_name="model_report.md",
                mime="text/markdown",
                use_container_width=True
            )


if st.session_state.get("page") == "Разъяснение результатов (с ИИ)":
    st.title("💬 Поговорим о ваших данных?")
    st.markdown("---")

    # Кнопка очистки чата
    if st.button("🗑 Очистить чат"):
        reset_chat_history()
        st.success("Чат очищен.")
        st.stop()  # чтобы сразу перерисовать пустой чат

    # Инициализируем историю чата
    st.session_state.setdefault("chat_history", [])

    # Рендерим предыдущие сообщения
    for msg in st.session_state.chat_history:
        render_message(msg["text"], msg["sender"])

    # Ввод нового сообщения
    question = st.chat_input("Напишите свой вопрос…")

    if question:
        # Отображаем вопрос пользователя
        st.session_state.chat_history.append({"text": question, "sender": "user"})
        render_message(question, "user")

        # Получаем и показываем ответ ИИ
        with st.spinner("ИИ обрабатывает…"):
            answer = chat_only(question)

        st.session_state.chat_history.append({"text": answer, "sender": "ai"})
        render_message(answer, "ai")



elif st.session_state['page'] == "Руководство пользователя":
    st.title("Руководство пользователя ClaryData")
    
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(content)
    except FileNotFoundError:
        st.warning("Файл README.md не найден, проверь путь или название файла.")


# Футер внизу страницы (автор)
# Постоянная надпись внизу лево, вне зависимости от содержимого
st.markdown("""
    <style>
        .bottom-right {
            position: fixed;
            right: 15px;
            bottom: 10px;
            font-size: 0.75em;
            color: #333333;
            z-index: 9999;
        }
    </style>
    <div class="bottom-right">© Created by Rahimov M.A. TTU 2025</div>
""", unsafe_allow_html=True)

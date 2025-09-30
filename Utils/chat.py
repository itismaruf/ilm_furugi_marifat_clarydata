import streamlit as st

from AI_helper import chat_with_context

def continue_chat(user_message):
    """Обрабатывает сообщение пользователя с учётом контекста проекта."""
    if not user_message or not isinstance(user_message, str):
        return "❌ Пустой или некорректный запрос."

    return chat_with_context(user_message.strip())


def render_message(text: str, sender: str):
    if sender == "user":
        cols = st.columns([1, 3])
        with cols[1]:
            st.markdown(
                f"<div style='background:#E0F7FA; padding:8px; border-radius:8px; "
                f"text-align:right; margin:5px 0;'>"
                f"🧑‍💻 {text}</div>",
                unsafe_allow_html=True,
            )
    else:
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(
                f"<div style='background:#F1F8E9; padding:8px; border-radius:8px; "
                f"text-align:left; margin:5px 0;'>"
                f"🤖 {text}</div>",
                unsafe_allow_html=True,
            )


def reset_chat_history():
    """
    Очищает историю чата в session_state.
    """
    st.session_state["chat_history"] = []
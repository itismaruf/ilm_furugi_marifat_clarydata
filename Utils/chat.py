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
                f"""
                <div style='
                    background: rgba(0, 123, 255, 0.1);
                    color: var(--text-color);
                    padding: 10px 14px;
                    border-radius: 12px;
                    text-align: right;
                    margin: 6px 0;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.15);
                '>
                    🧑‍💻 {text}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(
                f"""
                <div style='
                    background: rgba(40, 167, 69, 0.1);
                    color: var(--text-color);
                    padding: 10px 14px;
                    border-radius: 12px;
                    text-align: left;
                    margin: 6px 0;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.15);
                '>
                    🤖 {text}
                </div>
                """,
                unsafe_allow_html=True,
            )



def reset_chat_history():
    """
    Очищает историю чата в session_state.
    """
    st.session_state["chat_history"] = []
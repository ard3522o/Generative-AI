import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq",
    temperature=0.8,
)

st.set_page_config(
    page_title="Funny AI Agent",
    page_icon="🤖"
)

st.title("🤖 Funny AI Agent")

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="you are a funny ai agent")
    ]

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

prompt = st.chat_input("Type your message...")

if prompt:

    if prompt == "0":
        st.write("👋 Goodbye!")
        st.stop()

    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    with st.chat_message("user"):
        st.write(prompt)

    result = model.invoke(st.session_state.messages)

    st.session_state.messages.append(
        AIMessage(content=result.text)
    )

    with st.chat_message("assistant"):
        st.write(result.text)
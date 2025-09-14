import sys
import os
from dotenv import load_dotenv

# Force Python to use pysqlite3 instead of default sqlite3
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from datetime import datetime

load_dotenv()

# 🔑 Your API Key
groq_api_key = os.getenv("GROQ_API_KEY")


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------
# Session State Initialization
# -------------------------
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = 0

    # Initialize ChromaDB once
    if "collection" not in st.session_state:
        try:
            persist_dir = "./chroma_db"
            os.makedirs(persist_dir, exist_ok=True)
            st.session_state.collection = Chroma(
                collection_name="chats", persist_directory=persist_dir
            )
        except Exception as e:
            st.error(f"Chroma init error: {e}")
            st.session_state.collection = None


# -------------------------
# Custom Persona Prompts
# -------------------------
def get_custom_prompt():
    persona = st.session_state.get("selected_persona", "Default")
    personas = {
        "Default": """You are a friendly and helpful AI assistant, providing clear, concise, and accurate responses based only on Wikipedia.
        Always provide the Wikipedia link as source. If content is unavailable, say: 'The content is not available on Wikipedia.'
        """,
        "Expert": """You are a highly knowledgeable expert. Give precise, technical answers based on Wikipedia.
        Always provide the Wikipedia link as source. If content is unavailable, say: 'The content is not available on Wikipedia.'
        """,
        "Creative": """You are an imaginative and creative AI, using only Wikipedia as source. Use vivid descriptions and metaphors.
        Always provide the Wikipedia link as source. If content is unavailable, say: 'The content is not available on Wikipedia.'
        """,
    }

    return ChatPromptTemplate.from_messages(
        [
            ("system", personas[persona]),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )


# -------------------------
# Store Messages in Chroma
# -------------------------
def store_message(human_message, ai_message):
    if st.session_state.collection:
        st.session_state.collection.add_texts(
            texts=[f"Human: {human_message}\nAI: {ai_message}"],
            metadatas=[{"chat_id": st.session_state.chat_id}],
            ids=[f"message_{len(st.session_state.chat_history)}"],
        )


# -------------------------
# Retrieve Similar Messages
# -------------------------
def get_similar_messages(query, n=3):
    if not st.session_state.collection:
        return []

    try:
        results = st.session_state.collection.similarity_search(query, k=n)
        return [doc.page_content for doc in results]
    except Exception as e:
        st.error(f"Chroma query error: {e}")
        return []


# -------------------------
# Main App
# -------------------------
def main():
    initialize_session_state()

    with st.sidebar:
        st.title("⚙️ Configure Chat")
        model = st.selectbox(
            "Select Model:",
            ["llama-3.3-70b-versatile", "gemma2-9b-it", "llama-3.1-8b-instant", 'qwen/qwen3-32b'],
        )
        st.session_state.selected_persona = st.selectbox(
            "Choose Personality 🎭", ["Default", "Expert", "Creative"]
        )

        if st.session_state.start_time:
            st.subheader("📊 Session Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Messages", len(st.session_state.chat_history))
            with col2:
                duration = datetime.now() - st.session_state.start_time
                st.metric(
                    "Session Duration",
                    f"{duration.seconds // 60}m {duration.seconds % 60}s",
                )

        if st.button("🗑️ Reset Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.start_time = None
            st.session_state.chat_id += 1
            st.rerun()

    st.title("📖 Wikipedia AI Chatbot")

    # LLM Setup
    groq_chat = ChatGroq(api_key=groq_api_key, model_name=model)

    # Prompt
    prompt = get_custom_prompt()

    # Chain
    def run_chain(user_input):
        # Build conversation history in LangChain format
        history = []
        for msg in st.session_state.chat_history:
            history.append(HumanMessage(content=msg["human"]))
            history.append(AIMessage(content=msg["AI"]))

        final_prompt = prompt.invoke({"history": history, "input": user_input})
        return groq_chat.invoke(final_prompt.to_messages())

    # Render previous messages
    for message in st.session_state.chat_history:
        with st.container():
            st.info(f"🗣️ You: {message['human']}")
            st.success(f"🤖 AI: {message['AI']}")

    # User input
    user_question = st.text_area(
        "💬 Ask something:",
        height=100,
        placeholder="Ask me anything...",
        key="user_input",
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        send_button = st.button("🚀 Send", use_container_width=True)

    if send_button and user_question:
        if not st.session_state.start_time:
            st.session_state.start_time = datetime.now()

        with st.spinner("Thinking..."):
            try:
                similar_context = get_similar_messages(user_question)
                context = "\n".join(similar_context)
                if context:
                    user_question = f"(Related past context: {context})\n\n{user_question}"

                response = run_chain(user_question)
                ai_response = response.content

                st.session_state.chat_history.append(
                    {"human": user_question, "AI": ai_response}
                )
                store_message(user_question, ai_response)

                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown(
        f"✅ Chat is running in **{st.session_state.selected_persona} mode** using **{model}**"
    )


if __name__ == "__main__":
    main()

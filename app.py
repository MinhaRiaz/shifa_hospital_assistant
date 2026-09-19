import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Page setup
st.set_page_config(
    page_title="Hospital Knowledge Assistant",
    page_icon="🏥",
    layout="centered"
)

# Custom CSS to match screenshot UI
st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
    }
    .title-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 5px;
    }
    .title-icon {
        font-size: 2.2rem;
        background: #F3F4F6;
        padding: 8px 12px;
        border-radius: 10px;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #111827;
        margin: 0;
    }
    .sub-caption {
        color: #6B7280;
        font-size: 1rem;
        margin-bottom: 30px;
    }
    .user-box {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 1.05rem;
        font-weight: 500;
        color: #1E293B;
        margin-bottom: 15px;
        border: 1px solid #F1F5F9;
    }
    .user-icon {
        background-color: #FF5252;
        color: white;
        border-radius: 8px;
        padding: 6px;
        font-size: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
    }
    .assistant-header {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 10px;
    }
    .assistant-icon {
        background-color: #FFB000;
        color: white;
        border-radius: 8px;
        padding: 6px;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
    }
    </style>
""", unsafe_allow_html=True)

DB_FAISS_PATH = "faiss_index"

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    if not os.path.exists(DB_FAISS_PATH):
        st.error(f"Directory '{DB_FAISS_PATH}' not found! Please run ingest.py first.")
        st.stop()
        
    vectorstore = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

st.markdown('''
    <div class="title-container">
        <span class="title-icon">🏥</span>
        <h1 class="main-title">Hospital Knowledge Assistant</h1>
    </div>
    <div class="sub-caption">
        Ask questions about hospital policies and get answers grounded in the hospital knowledge base.
    </div>
''', unsafe_allow_html=True)

if not groq_api_key:
    st.warning("⚠️ `GROQ_API_KEY` not found in `.streamlit/secrets.toml`. Please configure your secret key.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'''
            <div class="user-box">
                <div class="user-icon">👤</div>
                <div>{msg["content"]}</div>
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
            <div class="assistant-header">
                <div class="assistant-icon">🤖</div>
                <div>Response</div>
            </div>
        ''', unsafe_allow_html=True)
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📚 Source Documents"):
                for src in msg["sources"]:
                    st.write(f"• **Department:** {src['department']} | **File:** `{src['file']}`")

if prompt := st.chat_input("Ask a question about hospital policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'''
        <div class="user-box">
            <div class="user-icon">👤</div>
            <div>{prompt}</div>
        </div>
    ''', unsafe_allow_html=True)

    with st.spinner("Searching knowledge base..."):
        vectorstore = load_vectorstore()
        docs = vectorstore.similarity_search(prompt, k=3)

        context_blocks = []
        sources_info = []

        for doc in docs:
            dept = doc.metadata.get('department', 'Unknown')
            fname = doc.metadata.get('file_name', doc.metadata.get('source', 'Unknown'))
            context_blocks.append(f"[Department: {dept} | File: {fname}]\n{doc.page_content}")
            sources_info.append({"department": dept, "file": fname})

        context_text = "\n\n".join(context_blocks)

        if groq_api_key:
            try:
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name="openai/gpt-oss-120b",
                    temperature=0.2
                )

                chat_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a professional assistant for hospital staff. "
                               "Answer the user's question accurately based ONLY on the provided context. "
                               "Use clean markdown formatting, bold section headers, and numbered steps. "
                               "If the context does not contain the answer, state that clearly."),
                    ("human", "Context:\n{context}\n\nQuestion: {question}")
                ])

                chain = chat_prompt | llm
                response = chain.invoke({"context": context_text, "question": prompt})
                answer = response.content
            except Exception as e:
                answer = f"Error generating response from Groq API: {e}"
        else:
            answer = "Here are the top retrieved sources from the knowledge base (Configure `GROQ_API_KEY` to see generated answers):"

        st.markdown(f'''
            <div class="assistant-header">
                <div class="assistant-icon">🤖</div>
                <div>Response</div>
            </div>
        ''', unsafe_allow_html=True)
        st.markdown(answer)

        with st.expander("📚 Source Documents"):
            for src in sources_info:
                st.write(f"• **Department:** {src['department']} | **File:** `{src['file']}`")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources_info
        })

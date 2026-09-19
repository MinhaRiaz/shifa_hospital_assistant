import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Page setup
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🏥",
    layout="wide"
)

DB_FAISS_PATH = "faiss_index"

# 1. Load Embeddings & FAISS Vectorstore with Caching
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

# 2. Get API Key from Secrets
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

st.title("🏥 Enterprise Knowledge Assistant")
st.caption("Ask questions about hospital policies, procedures, and department guidelines.")

if not groq_api_key:
    st.warning("⚠️ `GROQ_API_KEY` not found in `.streamlit/secrets.toml`. Please configure your secret key to enable LLM answers.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📚 Source Documents"):
                for src in msg["sources"]:
                    st.write(f"• **Department:** {src['department']} | **File:** `{src['file']}`")

# Handle User Input
if prompt := st.chat_input("Ask a question about hospital procedures..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            vectorstore = load_vectorstore()
            docs = vectorstore.similarity_search(prompt, k=3)

            # Build context & track sources
            context_blocks = []
            sources_info = []

            for doc in docs:
                dept = doc.metadata.get('department', 'Unknown')
                fname = doc.metadata.get('file_name', doc.metadata.get('source', 'Unknown'))
                context_blocks.append(f"[Department: {dept} | File: {fname}]
{doc.page_content}")
                sources_info.append({"department": dept, "file": fname})

            context_text = "

".join(context_blocks)

            if groq_api_key:
                try:
                    # Query Groq LLM
                    llm = ChatGroq(
                        groq_api_key=groq_api_key,
                        model_name="openai/gpt-oss-120b",
                        temperature=0.2
                    )

                    chat_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a professional assistant for hospital staff. "
                                   "Answer the user's question accurately based ONLY on the provided context. "
                                   "If the context does not contain the answer, state that clearly."),
                        ("human", "Context:
{context}

Question: {question}")
                    ])

                    chain = chat_prompt | llm
                    response = chain.invoke({"context": context_text, "question": prompt})
                    answer = response.content
                except Exception as e:
                    answer = f"Error generating response from Groq API: {e}"
            else:
                answer = "Here are the top retrieved sources from the knowledge base (Configure `GROQ_API_KEY` to see generated answers):"

            st.markdown(answer)

            # Display source documents expander
            with st.expander("📚 Source Documents"):
                for src in sources_info:
                    st.write(f"• **Department:** {src['department']} | **File:** `{src['file']}`")

            # Store assistant message in history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources_info
            })

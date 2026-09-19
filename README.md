# 🏥 Hospital Knowledge Assistant

An AI-powered **Hospital Knowledge Assistant** built with **Streamlit, FAISS, Hugging Face Embeddings, LangChain, and Groq**.

The application allows hospital staff to ask questions about **hospital policies, procedures, department guidelines, and other internal documents**. It retrieves relevant information from the hospital knowledge base and uses an LLM to generate an answer based only on the retrieved context.

## 🚀 Live Demo

**Streamlit App:**
https://shifa-hospital-assistant-app.streamlit.app/

---

## ✨ Features

* 🏥 **Hospital Knowledge Assistant**

  * Ask questions about hospital policies, procedures, and department guidelines.

* 🔎 **Semantic Search**

  * Uses FAISS to find the most relevant documents from the knowledge base.

* 🤖 **AI-Powered Answers**

  * Uses Groq's `openai/gpt-oss-120b` model to generate responses.

* 📚 **Source Documents**

  * Displays the department and file associated with the retrieved information.

* 💬 **Chat Interface**

  * Maintains conversation history using Streamlit session state.

* ⚡ **Cached Resources**

  * Uses Streamlit caching to avoid repeatedly loading the embedding model and FAISS database.

* 🔐 **API Key Security**

  * Groq API key is loaded through Streamlit Secrets or environment variables.

---

## 🧠 How It Works

The application follows a simple **Retrieval-Augmented Generation (RAG)** workflow:

```text
User Question
      ↓
FAISS Semantic Search
      ↓
Retrieve Top 3 Relevant Documents
      ↓
Build Context
      ↓
Groq LLM
      ↓
Generate Answer
      ↓
Display Answer + Sources
```

### 1. User asks a question

The user enters a question through the Streamlit chat interface.

### 2. Document Retrieval

The question is searched against the FAISS vector database using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The application retrieves the **top 3 most relevant documents**.

### 3. Context Creation

The retrieved documents are combined into a context containing:

* Department
* File name
* Document content

### 4. AI Response

The retrieved context is passed to the Groq LLM:

```text
openai/gpt-oss-120b
```

The model is instructed to answer **only from the provided context**.

### 5. Sources

The application displays the retrieved department and file information under **Source Documents**.

---

## 🛠️ Technologies Used

| Technology            | Purpose                      |
| --------------------- | ---------------------------- |
| Python                | Programming language         |
| Streamlit             | Web application interface    |
| LangChain             | LLM and RAG orchestration    |
| FAISS                 | Vector similarity search     |
| Hugging Face          | Text embeddings              |
| Sentence Transformers | Document/question embeddings |
| Groq                  | LLM inference                |
| `openai/gpt-oss-120b` | Language model               |

---

## 📦 Project Structure

```text
hospital-knowledge-assistant/
│
├── app.py
├── ingest.py
├── requirements.txt
├── README.md
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
└── .streamlit/
    └── secrets.toml
```

> The exact files in the project may vary depending on how the knowledge base is created.

---

## 🔑 API Key Configuration

The application expects a Groq API key.

For local development, create:

```text
.streamlit/secrets.toml
```

and add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

For Streamlit Cloud, add the same secret through the application's **Secrets** settings.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd hospital-knowledge-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

### 5. Create the FAISS index

Run the ingestion script:

```bash
python ingest.py
```

This should generate the:

```text
faiss_index/
```

directory.

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔍 Retrieval System

The application uses:

```text
HuggingFaceEmbeddings
```

with:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The generated embeddings are stored in a **FAISS vector database**.

When a user asks a question, the application performs semantic similarity search:

```python
docs = vectorstore.similarity_search(prompt, k=3)
```

This retrieves the three most relevant document chunks.

---

## 🤖 LLM Configuration

The application uses Groq through LangChain:

```python
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.2
)
```

A low temperature is used to encourage more consistent responses.

The system prompt instructs the model to:

* Act as a professional hospital assistant.
* Answer based only on the retrieved context.
* Clearly state when the available context does not contain the answer.

---

## 📚 Source Tracking

Each retrieved document contains metadata such as:

```text
Department
File Name
```

The application displays these sources under:

**📚 Source Documents**

This helps users identify which department and document contributed to the retrieved information.

---

## 🔐 Security

The Groq API key should **never be hard-coded directly into the source code**.

Instead, use:

```text
.streamlit/secrets.toml
```

or Streamlit Cloud Secrets.

Also make sure that sensitive hospital documents and credentials are not accidentally committed to a public GitHub repository.

---

## ⚠️ Limitations

* The quality of answers depends on the documents available in the FAISS knowledge base.
* The assistant can only provide information contained in the retrieved context.
* Incorrect or incomplete source documents can lead to incomplete answers.
* The system is intended as a **knowledge-assistance tool**, not a replacement for official hospital policies or professional judgment.
* Access to confidential hospital information should be appropriately controlled.

---

## 🌐 Live Application

Try the application here:

**https://shifa-hospital-assistant-app.streamlit.app/**

---

## 👩‍💻 Author

Developed as an AI-powered **Hospital Knowledge Assistant** using Retrieval-Augmented Generation (RAG), semantic search, and large language models.

---

## 📄 License

This project is intended for educational and demonstration purposes. Add an appropriate open-source license if the project is distributed publicly.

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with Navya's Portfolio",
    page_icon="🤖",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

.main-header {
    text-align: center;
    padding: 2rem 0 1rem;
}

.main-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    color: #7ee8a2;
    letter-spacing: -1px;
    margin-bottom: 0.3rem;
}

.main-header p {
    color: #888;
    font-size: 0.9rem;
}

.tag-row {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin: 1rem 0 2rem;
}

.tag {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    color: #7ee8a2;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
}

.suggestion-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 1rem 0;
}

.suggestion-card {
    background: #111120;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #bbb;
    cursor: pointer;
    transition: border-color 0.2s;
}

.suggestion-card:hover {
    border-color: #7ee8a2;
    color: #e8e8f0;
}

.stChatMessage {
    background: transparent !important;
}

[data-testid="stChatMessageContent"] {
    background: #111120 !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
}

.stChatInputContainer {
    border-top: 1px solid #2a2a4a;
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⚡ Chat with Navya's Portfolio</h1>
    <p>Ask me anything about Navya's skills, projects, and experience</p>
</div>
<div class="tag-row">
    <span class="tag">AI/ML</span>
    <span class="tag">RAG</span>
    <span class="tag">LangChain</span>
    <span class="tag">GenAI</span>
    <span class="tag">Python</span>
</div>
""", unsafe_allow_html=True)

# ── Load API key ──────────────────────────────────────────────────────────────
groq_api_key = os.environ.get("GROQ_API_KEY", "")

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password")
    if not groq_api_key:
        st.sidebar.warning("Add your Groq API key to start chatting!")
        st.stop()

# ── Load & index resume ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📄 Reading resume...")
def load_vectorstore():
    loader = PyPDFLoader("resume_navya.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = load_vectorstore()

# ── Build chain ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Setting up AI...")
def build_chain(_vectorstore, api_key):
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-8b-8192",
        temperature=0.3
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=False,
        verbose=False
    )
    return chain

chain = build_chain(vectorstore, groq_api_key)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Suggested questions ───────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="suggestion-grid">
        <div class="suggestion-card">💼 What projects has Navya built?</div>
        <div class="suggestion-card">🛠️ What tech stack does she know?</div>
        <div class="suggestion-card">🎓 What is her educational background?</div>
        <div class="suggestion-card">🤖 Tell me about her GenAI experience</div>
    </div>
    """, unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about Navya's portfolio..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            system_context = """You are an AI assistant representing Navya Kapoor's portfolio. 
            Answer questions about her skills, projects, education, and experience based on her resume.
            Be concise, friendly, and professional. If asked something not in the resume, say so politely.
            Always speak positively about Navya's work."""

            full_prompt = f"{system_context}\n\nQuestion: {prompt}"
            result = chain.invoke({"question": full_prompt})
            answer = result["answer"]

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👩‍💻 Navya Kapoor")
    st.markdown("AI/ML Engineer · Bennett University")
    st.markdown("---")
    st.markdown("🔗 [GitHub](https://github.com/navyakapoor004)")
    st.markdown("🔗 [LinkedIn](https://linkedin.com/in/navya-kapoor826632aa)")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

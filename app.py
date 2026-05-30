import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

st.set_page_config(page_title="Chat with Navya's Portfolio", page_icon="🤖", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }
.main-header { text-align: center; padding: 2rem 0 1rem; }
.main-header h1 { font-family: 'Space Mono', monospace; font-size: 1.8rem; color: #7ee8a2; letter-spacing: -1px; margin-bottom: 0.3rem; }
.main-header p { color: #888; font-size: 0.9rem; }
.tag-row { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin: 1rem 0 2rem; }
.tag { background: #1a1a2e; border: 1px solid #2a2a4a; color: #7ee8a2; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-family: 'Space Mono', monospace; }
</style>
""", unsafe_allow_html=True)

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

# ── API Key ──
groq_api_key = os.environ.get("GROQ_API_KEY", "")
if not groq_api_key:
    groq_api_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password")
    if not groq_api_key:
        st.sidebar.warning("Add your Groq API key to start chatting!")
        st.stop()

# ── Load vectorstore ──
@st.cache_resource(show_spinner="📄 Reading resume...")
def load_vectorstore():
    loader = PyPDFLoader("resume_navya.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ── Build RAG chain ──
@st.cache_resource(show_spinner="🤖 Setting up AI...")
def build_chain(api_key):
    llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192", temperature=0.3)
    
    prompt = PromptTemplate.from_template("""You are an AI assistant for Navya Kapoor's portfolio.
Answer questions about her skills, projects, education, and experience based on the context below.
Be concise, friendly, and professional. If something isn't in the context, say so politely.

Context: {context}

Question: {question}

Answer:""")
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

chain = build_chain(groq_api_key)

# ── Chat UI ──
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.info("💡 Try asking: *What projects has Navya built?* or *What are her technical skills?*")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Navya's portfolio..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chain.invoke(prompt)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Sidebar ──
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

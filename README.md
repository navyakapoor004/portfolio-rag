# 🤖 Chat with Navya's Portfolio — RAG Chatbot

An AI-powered portfolio chatbot built with **RAG (Retrieval-Augmented Generation)** that lets anyone have a conversation with Navya Kapoor's resume and portfolio.

## 🚀 Live Demo
> Deploy on Streamlit Cloud (free) — instructions below

## 🛠️ Tech Stack
- **LangChain** — RAG pipeline orchestration
- **Groq (Llama 3)** — Fast LLM inference (free tier)
- **FAISS** — Local vector store for semantic search
- **HuggingFace Embeddings** — `all-MiniLM-L6-v2` for text embeddings
- **Streamlit** — Interactive web UI
- **PyPDF** — Resume PDF parsing

## 📁 Project Structure
```
portfolio-rag/
├── app.py                 # Main Streamlit app
├── resume_navya.pdf       # Resume (source knowledge base)
├── requirements.txt       # Python dependencies
└── README.md
```

## ⚙️ Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/navyakapoor004/portfolio-rag
cd portfolio-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key
Get a free key at [console.groq.com](https://console.groq.com)

```bash
export GROQ_API_KEY=your_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add `GROQ_API_KEY` in **Secrets** settings
5. Deploy!

## 💡 How It Works

```
Resume PDF → PyPDF Loader → Text Chunks → FAISS Vector Store
                                                    ↓
User Question → Embedding → Semantic Search → Relevant Chunks
                                                    ↓
                              Groq Llama 3 → Contextual Answer
```

## 🔮 Features
- 💬 Conversational memory (remembers context)
- 🎯 Semantic search over resume content
- ⚡ Fast responses via Groq's free API
- 🎨 Clean dark UI with suggested questions
- 📱 Mobile friendly

---
Built by [Navya Kapoor](https://github.com/navyakapoor004) | AI/ML Engineer

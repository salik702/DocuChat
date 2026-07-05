<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=220&section=header&text=DOCUCHAT%20AI&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=RAG%20Chatbot%20%E2%80%A2%20LangChain%20%2B%20Mistral%20%2B%20ChromaDB&descAlignY=55&descSize=18&descColor=93c5fd" width="100%"/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=20&duration=2600&pause=700&color=818CF8&center=true&vCenter=true&repeat=true&width=750&height=50&lines=Chat+with+Any+PDF+Document;Mistral+Embeddings+%2B+ChromaDB;Retrieval-Augmented+Generation;Zero+Hallucination%2C+Source-Grounded" alt="Typing SVG" />
</a>

<br/>

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Mistral%20AI-FA520F?style=for-the-badge&logo=mistralai&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-6E56CF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready-22c55e?style=for-the-badge" />
</p>

</div>

<br/>

## 📌 Overview

**DocuChat AI** is a Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF — a textbook, research paper, or manual — and ask questions answered strictly from its content. It combines Mistral's embedding and chat models with a local ChromaDB vector store, wrapped in a polished Streamlit interface with live-tunable retrieval settings and full source-chunk transparency.

> Built to demonstrate a complete, production-style RAG pipeline: document ingestion, chunking, vector embedding, MMR-based retrieval, and grounded generation — with no external database or vendor lock-in beyond the Mistral API.

<br/>

## ✨ Features

<table>
<tr>
<td width="50%" valign="top">

### 📄 Document Ingestion
- Upload any PDF directly from the browser
- Automatic chunking with configurable size/overlap
- Per-document vector store (hash-based caching)
- Re-uploads skip re-embedding automatically

</td>
<td width="50%" valign="top">

### 🧠 Retrieval-Augmented Generation
- Mistral `mistral-embed` for semantic embeddings
- MMR search for diverse, non-redundant context
- Strict "context-only" answering — no hallucinated facts
- Configurable top-k, fetch-k, and diversity (λ)

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 💬 Chat Experience
- Persistent multi-turn conversation history
- Source chunks + page numbers shown per answer
- Model + temperature selection on the fly
- One-click chat reset or document swap

</td>
<td width="50%" valign="top">

### 🎨 Interface
- Custom dark, glassmorphism-inspired theme
- Real-time chunk/message metrics
- Responsive layout, no default Streamlit look
- Clean upload-first onboarding flow

</td>
</tr>
</table>

<br/>

## 🧱 System Architecture

```mermaid
flowchart LR
    A["Upload PDF"] --> B["PyPDFLoader"]
    B --> C["Recursive Text Splitter"]
    C --> D["Mistral Embeddings"]
    D --> E[("ChromaDB")]
    F["User Question"] --> G["MMR Retriever"]
    E --> G
    G --> H["Context + Prompt Template"]
    H --> I["ChatMistralAI"]
    I --> J["Grounded Answer + Sources"]
```

<br/>

## 🛠️ Tech Stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/Mistral%20AI-FA520F?style=flat-square&logo=mistralai&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-6E56CF?style=flat-square" />
  <img src="https://img.shields.io/badge/PyPDF-000000?style=flat-square" />
</p>

**Core techniques applied:** Document chunking · Vector embeddings · Maximal Marginal Relevance (MMR) retrieval · Prompt engineering · Context-grounded generation

<br/>

## 📁 Project Structure

```
DocuChat-AI/
├── app.py               # Streamlit application (upload + chat UI)
├── chroma_stores/        # Auto-generated per-document vector stores
├── requirements.txt
└── README.md
```

<br/>

## 🔧 Installation

```bash
git clone https://github.com/SalikAhmad702/DocuChat-AI.git
cd DocuChat-AI

python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file with your Mistral API key:
```
MISTRAL_API_KEY=your_api_key_here
```

Then run the app:
```bash
streamlit run app.py
```

<br/>

## 📖 Usage

| Action | Steps |
|---|---|
| **Upload a document** | Home screen → choose a PDF → click Process document |
| **Ask a question** | Type in the chat box once the document is indexed |
| **View sources** | Expand "Sources" under any answer to see the retrieved chunks + page numbers |
| **Tune retrieval** | Sidebar → adjust chunk size, top-k, fetch-k, or MMR diversity |
| **Switch model** | Sidebar → Model → pick a Mistral model + temperature |
| **Start over** | Sidebar → Load a different document |

<br/>

## 📊 Core Components

| Component | Responsibility |
|---|---|
| `build_vectorstore` | Loads PDF, splits into chunks, embeds via Mistral, persists to ChromaDB |
| `get_retriever` | Configures MMR-based retriever with tunable k / fetch_k / λ |
| `answer_question` | Retrieves context, fills prompt template, queries `ChatMistralAI` |
| `file_hash` | Deduplicates uploads so identical PDFs reuse existing embeddings |

**System prompt used for generation**
```
You are a helpful assistant. Use only the provided context to answer 
the question. If the context does not contain the answer, respond 
with "I don't know."
```

<br/>

## 🗺️ Roadmap

- [x] PDF upload + chunking + embedding pipeline
- [x] MMR retrieval with tunable parameters
- [x] Source-chunk transparency in UI
- [ ] Multi-document / multi-PDF sessions
- [ ] Conversation memory across follow-up questions
- [ ] Support for DOCX and TXT uploads
- [ ] Streaming token-by-token responses
- [ ] Deployment guide (Streamlit Cloud / Docker)

<br/>

## 🤝 Contributing

Contributions are welcome — especially multi-document support, streaming responses, and evaluation metrics for retrieval quality.

```bash
git checkout -b feature/your-improvement
git commit -m "feat: describe your change"
git push origin feature/your-improvement
```
Then open a pull request.

<br/>

## 📄 License

This project is **free and open source** — no license restrictions.

- ✅ Free to use, modify, and distribute
- ✅ Free for personal, academic, and commercial use
- ✅ No attribution required
- ⚠️ Provided "as-is" without warranty

---

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 📧 Let's Connect

<div align="center">

<h3>Built with obsession by <b>Salik Ahmad</b> 🤖</h3>

<p>
  <a href="https://salikahmad.vercel.app/" target="_blank">
    <img src="https://img.shields.io/badge/Website-salikahmad.vercel.app-818CF8?style=for-the-badge&logo=vercel&logoColor=white&labelColor=0d0d0d" />
  </a>
  <a href="https://www.linkedin.com/in/salik-ahmad-programmer/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-Salik%20Ahmad-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white&labelColor=0d0d0d" />
  </a>
  <a href="https://www.kaggle.com/salikahmad702" target="_blank">
    <img src="https://img.shields.io/badge/Kaggle-salikahmad702-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white&labelColor=0d0d0d" />
  </a>
  <a href="https://github.com/SalikAhmad702" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-SalikAhmad702-181717?style=for-the-badge&logo=github&logoColor=white&labelColor=0d0d0d" />
  </a>
</p>

<br/>

<a href="https://salikahmad.vercel.app/">
  <img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&size=14&duration=4000&pause=1000&color=818CF8&center=true&vCenter=true&width=700&lines=AI%2FML+Engineer;Copyright+©+2026+Salik+Ahmad.+All+rights+reserved." alt="Footer Typing" />
</a>

<br/><br/>

<img src="https://capsule-render.vercel.app/api?type=shark&color=0:0f0c29,50:302b63,100:24243e&height=140&section=footer" width="100%"/>

</div>

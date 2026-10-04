# Local Doc Chat 📄💬

**Chat with your own documents** — attach PDFs, text files, or Markdown, ask
questions, and get answers with cited sources. A from-scratch RAG
(Retrieval-Augmented Generation) pipeline running 100% on your machine.

- 📎 Attach documents in the chat — they're chunked, embedded, and indexed
- 🔍 Questions retrieve the most relevant chunks before answering
- 📚 Every answer cites its source documents
- 🙅 Honest "I don't know" when the answer isn't in your docs — no hallucinating
- 🔒 No API keys, no cloud — Ollama + ChromaDB, all local

Built as project #2 of my AI engineering journey, following the
[AI Engineering Hub](https://github.com/patchy631/ai-engineering-hub) roadmap.

## Quickstart

**1. Pull the models** (~2.5 GB total)
```bash
ollama pull nomic-embed-text   # embeddings
ollama pull llama3.2           # chat (3B — fast and decent)
```

**2. Install dependencies** (Python 3.11+)
```bash
pip install -r requirements.txt
```

**3. Run**
```bash
chainlit run app.py -w
```
Open http://localhost:8000, attach a document with the paperclip, then ask away.
Type `clear docs` anytime to wipe the index.

## How it works

```
You attach report.pdf
        │
        ▼
┌───────────────┐   ┌──────────────┐   ┌─────────────┐
│ Chunk (800    │──▶│ Embed        │──▶│ ChromaDB    │
│ chars, 120    │   │ nomic-embed  │   │ (local,      │
│ overlap)      │   │ -text        │   │ persistent) │
└───────────────┘   └──────────────┘   └─────────────┘

You ask "What was Q3 revenue?"
        │
        ▼
┌──────────────┐   ┌──────────────┐   ┌─────────────┐
│ Embed the    │──▶│ Top-4 chunks │──▶│ llama3.2      │
│ question     │   │ retrieved    │   │ answers with  │
└──────────────┘   └──────────────┘   │ citations     │
                                      └─────────────┘
```

- **Chunking**: overlapping 800-char windows, cut at paragraph/sentence boundaries
- **Retrieval**: cosine similarity over `nomic-embed-text` vectors, top 4 chunks
- **Honesty**: the prompt template forbids answering from outside the context
- **Memory**: raw Q&A pairs kept in session history; retrieval prompts are not

## Configuration

| Variable      | Default             | What it does                          |
|---------------|---------------------|---------------------------------------|
| `LLM_MODEL`   | `llama3.2`          | Ollama chat model                     |
| `EMBED_MODEL` | `nomic-embed-text`  | Ollama embedding model                |
| `CHROMA_DIR`  | `.chroma`           | Where the vector store lives on disk  |

Example — a bigger brain for harder documents:
```bash
ollama pull llama3.1:8b
LLM_MODEL=llama3.1:8b chainlit run app.py -w
```

## Project structure

```
├── app.py            # Chainlit UI: uploads, chat loop, streaming, citations
├── rag.py            # RAG engine: chunking, Ollama embeddings, ChromaDB
├── requirements.txt
└── README.md
```

## Roadmap

Project #2 in the series — next up: AI agents that can use tools.

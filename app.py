"""
Local Doc Chat — ask questions about your own documents.

Attach PDF / TXT / Markdown files in the chat to index them, then ask
anything. Answers stream token-by-token, cite their sources, and the model
says "I don't know" when the answer isn't in your documents.

Everything runs on your machine: Ollama for the LLM + embeddings,
ChromaDB for the vector store. No API keys, no cloud.

Setup:
    ollama pull nomic-embed-text
    ollama pull llama3.2
    pip install -r requirements.txt
    chainlit run app.py -w

Environment:
    LLM_MODEL    Chat model (default: llama3.2)
    EMBED_MODEL  Embedding model (default: nomic-embed-text)
    CHROMA_DIR   Vector store location (default: .chroma)
"""

import os

import chainlit as cl
import ollama

from rag import (
    EMBED_MODEL,
    clear_all,
    indexed_count,
    ingest_file,
    retrieve,
)

LLM = os.getenv("LLM_MODEL", "llama3.2")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant answering questions about the user's documents.",
)

DOC_EXTS = (".pdf", ".txt", ".md", ".markdown")

WELCOME = (
    "Welcome to your local document chat! Here's how it works:\n\n"
    "1. **Attach** a PDF, TXT, or Markdown file with the paperclip — I'll index it\n"
    "2. **Ask** anything about your documents\n\n"
    f"Powered by `{LLM}` + `{EMBED_MODEL}`, 100% on your machine.\n"
    'Type "clear docs" anytime to forget everything I\'ve indexed.'
)

QA_TEMPLATE = """Answer the question using ONLY the context below, which comes from the user's own documents.
If the answer is not in the context, say "I don't know based on the documents provided." Do not make things up.

Context:
{context}

Question: {question}

Answer (end with "Sources:" followed by the document names you used):"""

OLLAMA_DOWN_HINT = (
    "I couldn't reach Ollama. Make sure it's running:\n\n```bash\nollama serve\n```"
)

EMBED_MISSING_HINT = (
    f"The embedding model `{EMBED_MODEL}` isn't pulled yet. Run:\n\n"
    f"```bash\nollama pull {EMBED_MODEL}\n```\n\n"
    "then attach your document again."
)


def _is_model_missing_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "model" in msg and "not found" in msg


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [{"role": "system", "content": SYSTEM_PROMPT}])
    msg = cl.Message(content="")
    for token in WELCOME:
        await msg.stream_token(token)
    await msg.send()


@cl.on_message
async def main(message: cl.Message):
    elements = message.elements or []
    doc_files = [f for f in elements if f.path.lower().endswith(DOC_EXTS)]

    # 1) Index any attached documents first.
    if doc_files:
        try:
            lines, total = [], 0
            for f in doc_files:
                n = ingest_file(f.path, f.name)
                total += n
                lines.append(f"- {f.name} — {n} chunks")
            await cl.Message(
                content=f"Indexed **{total}** chunks:\n"
                + "\n".join(lines)
                + "\n\nAsk me anything about them!"
            ).send()
        except Exception as e:
            print(f"[local-doc-rag] ingest failed: {type(e).__name__}: {e}")
            hint = EMBED_MISSING_HINT if _is_model_missing_error(e) else OLLAMA_DOWN_HINT
            await cl.Message(content=hint).send()
            return
        if not message.content.strip():
            return

    text = message.content.strip()

    if text.lower() in ("clear docs", "clear", "forget docs", "forget"):
        clear_all()
        await cl.Message(
            content="Done — I've forgotten all indexed documents. Attach new ones anytime."
        ).send()
        return

    # 2) Answer from the indexed documents.
    try:
        count = indexed_count()
    except Exception as e:
        print(f"[local-doc-rag] store failed: {type(e).__name__}: {e}")
        await cl.Message(content=OLLAMA_DOWN_HINT).send()
        return

    if count == 0:
        await cl.Message(
            content="I don't have any documents yet. Attach a PDF, TXT, or Markdown "
            "file with the paperclip and I'll index it — then ask away."
        ).send()
        return

    try:
        hits = retrieve(text)
    except Exception as e:
        print(f"[local-doc-rag] retrieve failed: {type(e).__name__}: {e}")
        hint = EMBED_MISSING_HINT if _is_model_missing_error(e) else OLLAMA_DOWN_HINT
        await cl.Message(content=hint).send()
        return

    context = "\n\n---\n\n".join(
        f"[Source: {h['source']}]\n{h['text']}" for h in hits
    )
    prompt = QA_TEMPLATE.format(context=context, question=text)

    history = cl.user_session.get("history")
    messages = history + [{"role": "user", "content": prompt}]

    msg = cl.Message(content="")
    try:
        # Errors surface while iterating a stream, so the loop is inside try.
        stream = ollama.chat(model=LLM, messages=messages, stream=True)
        parts: list[str] = []
        for chunk in stream:
            token = chunk.message.content or ""
            parts.append(token)
            await msg.stream_token(token)
    except Exception as e:
        print(f"[local-doc-rag] chat failed: {type(e).__name__}: {e}")
        hint = EMBED_MISSING_HINT if _is_model_missing_error(e) else OLLAMA_DOWN_HINT
        await msg.stream_token(hint)
        await msg.send()
        return

    reply = "".join(parts)
    # Keep the raw Q&A in history, not the retrieval-stuffed prompt.
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": reply})
    await msg.send()

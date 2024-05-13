"""
This is a simple chatbot app that uses the Streamlit Chat component.
It is based on the use of a simple RAG simple orchestrated with some
components build at src folder.

Usage:
```bash
streamlit run app.py
```
"""

import streamlit as st

from src.helpers.config import Config
from src.processing.readers import PDFReader
from src.processing.chunking import SymbolChunker
from src.database.chromadb import ChromaDB
from src.embeddings.ollama import OllamaEmbedding
from src.database.types import CollectionItem
from src.llm.ollama import OllamaLLM

###########################################################################################
##################################### Configurations ######################################
###########################################################################################
CONFIG_PATH = "configs/settings-ollama.yaml"
CONFIG = Config(CONFIG_PATH)


# https://docs.streamlit.io/develop/concepts/architecture/caching
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    pdf_reader = PDFReader(enable_ocr=CONFIG.readers["enable_ocr"])
    chunker = SymbolChunker(
        chars_limit=CONFIG.readers["chunk_size"],
        overlap=CONFIG.readers["chunk_overlap"],
    )
    database = ChromaDB(
        path=CONFIG.vectorstore["path"], name=CONFIG.vectorstore["name"]
    )
    embedder = OllamaEmbedding(
        model=CONFIG.embedding["model"], base_url=CONFIG.embedding["api_url"]
    )
    llm = OllamaLLM(model=CONFIG.llm["model"], base_url=CONFIG.llm["api_url"])
    return pdf_reader, chunker, database, embedder, llm


pdf_reader, chunker, database, embedder, llm = load_models()

###########################################################################################
############################## Side Graphical Interface ###################################
###########################################################################################
st.sidebar.title("📁 Upload your data")
st.sidebar.write("Upload your PDFs files to populate the database.")
num_chunks = database.num_documents
st_num_chunks = st.sidebar.empty()
st_num_chunks.write(f"Number of chunks in the database: {num_chunks}")
# if the number of chunks is greater than 0, show a button to clear the database
if num_chunks > 0:
    if st.sidebar.button("Clear database"):
        database.remove()
        num_chunks = database.num_documents
        st_num_chunks.empty()
        st_num_chunks.write(f"Number of chunks in the database: {num_chunks}")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    pdf_path = "tmp/uploaded_file.pdf"
    # write the uploaded file to disk
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # read the uploaded file
    reader_placeholder = st.sidebar.empty()
    reader_placeholder.write("📖 Reading the uploaded file...")
    pdf_info = pdf_reader.get_text(pdf_path)
    reader_placeholder.empty()

    # create a list of chunks
    chunks = chunker.get_chunks(pdf_info)

    # Iterate over the chunks, extract the embeddings and insert them into the database
    chunks_bar = st.sidebar.progress(0, text="Processing chunks...")
    for chunk_index, chunk in enumerate(chunks):
        embedding = embedder.get_embedding(chunk.text)
        document = CollectionItem(
            embedding=embedding,
            document_path=pdf_path,
            location=chunk.location,
            text=chunk.text,
        )
        database.insert(document)
        chunks_bar.progress(
            (chunk_index + 1) / len(chunks),
            text=f"Processing chunk {chunk_index + 1}/{len(chunks)}",
        )

    chunks_bar.empty()

    st.sidebar.write("✅ File uploaded successfully!")
    st.sidebar.write("Uploaded file has been processed and added to the database.")
    num_chunks = database.num_documents
    st_num_chunks.empty()
    st_num_chunks.write(f"Number of chunks in the database: {num_chunks}")


###########################################################################################
############################## Main Graphical Interface ###################################
###########################################################################################
st.write("# 💬 Ask Me Anything *with RAG*")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input():
    query_embedding = embedder.get_embedding(query)
    query_context = database.search(query_embedding, top_k=CONFIG.embedding["top_k"])

    # print the query context
    with st.expander("Query context"):
        for item in query_context:
            st.write(f"**{item.document_path}: {item.location}**")
            st.write(item.text)

    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    msg = llm.ask(st.session_state.messages, query_context=query_context)
    st.session_state.messages.append({"role": "assistant", "content": msg})

    st.chat_message("assistant").write(msg)

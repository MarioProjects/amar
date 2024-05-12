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

from src.llm.ollama import OllamaLLM

###########################################################################################
##################################### Configurations ######################################
###########################################################################################
VERBOSE = True
EMBEDDING_MODEL = "qwen:0.5b"
OLLAMA_SERVER = "http://localhost:11434"

llm = OllamaLLM(model=EMBEDDING_MODEL, base_url=OLLAMA_SERVER)

###########################################################################################
################################## Graphical Interface ####################################
###########################################################################################
st.title("💬 Ask Me Anything")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = llm.ask(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

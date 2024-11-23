import os
from dotenv import find_dotenv, load_dotenv
import openai


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message
from utils import *


chat_history = []

qa_chain = qa()

st.title("Docs QA Bot using Langchain")
st.header("Ask anything about your documents... 🤖")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_query():
    """
    Get user input for querying the QA bot.

    Returns:
    str: User input query.
    """
    input_text = st.chat_input("Ask a question about your documents...")
    return input_text


user_input = get_query()
if user_input:
    result = qa_chain({"question": user_input, "chat_history": chat_history})
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result["answer"])


if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i], key=str(i))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st


st.set_page_config(
    page_title="Aroma Alchemist - AA",
    page_icon="🧴",
)

st.session_state["messages"] = []


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


# @st.cache_data(show_spinner="Embedding file...")
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = Chroma.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    if save:
        # if "messages" not in st.session_state:
        #    st.session_state["messages"] = []
        save_message(message, role)
    with st.chat_message(role):
        st.markdown(message)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are personal perfume adviser. You recommend perfumes based on the context ONLY.
            If you don't have the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


# UI starts from here
st.title("Aroma Alchemist")

st.markdown(
    """
Welcome!

I'm your personal perfume advisor.

You can tell me what type of perfume you're looking for, or simply select your preferences on the sidebar for a quick recommendation!
"""
)


with st.sidebar:
    st.title("Preferences")
    gender = st.radio("Gender:", ["Male", "Female", "Not specified"])
    age_group = st.radio("Age group:", ["10s-20s", "30s-40s", "50s+", "Not specified"])
    season = st.radio(
        "Season:", ["Spring", "Summer", "Fall", "Winter", "Not specified"]
    )
    submit_button = st.button("Submit Preferences")


file = st.file_uploader(
    "Upload reviews for embedding (.txt .pdf or .docx file)",
    type=["pdf", "txt", "docx"],
)

if file:
    retriever = embed_file(file)


send_message(
    "Hello! I'm here to help you find the perfect scent. What specific qualities or preferences are you looking for in a perfume?",
    "ai",
    save=False,
)
paint_history()
message = st.chat_input("Enter your fragrance preferences...")
if message:
    send_message(message, "human")
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    with st.chat_message("ai"):
        chain.invoke(message)

if submit_button:
    preferences_msg = f"Gender: {gender}, Age Group: {age_group}, Season: {season}"
    send_message(preferences_msg, "human")
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    with st.chat_message("ai"):
        chain.invoke(preferences_msg)
    # message = st.text_input(
    #    "Enter your fragrance preferences...", key="fragrance_prefs"
    # )

    # if message:
    #    send_message(message, "human")
    #    context = f"{preferences_msg}. Additional preferences: {message}"
    #    chain = (
    #        {
    #            "context": context,
    #            "question": message,
    #        }
    #        | prompt
    #        | llm
    #    )
    #    response = chain.invoke(message)
    #    send_message(response, "ai")

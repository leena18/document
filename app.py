import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to initialize the Google Gemini model
def initialize_gemini_model():
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])
    return chat

# Initialize Streamlit app
st.set_page_config(page_title="Chat with Multiple PDFs", layout="wide")

st.header("Chat with Multiple PDF Documents")

# Upload PDF files
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Load and split documents
    documents = []
    for uploaded_file in uploaded_files:
        loader = PyPDFLoader(uploaded_file)
        documents.extend(loader.load())

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and store in FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Initialize the Gemini model
    chat = initialize_gemini_model()

    # Initialize conversation chain with retrieval
    conversation_chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        llm=chat
    )

    # Chat history management
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Input field and submit button
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Ask a question based on the PDFs")
        submit = st.form_submit_button("Send")

    if submit and user_input:
        # Generate response
        response = conversation_chain({"question": user_input, "chat_history": st.session_state['chat_history']})

        # Update chat history
        st.session_state['chat_history'].append((user_input, response["answer"]))

    # Display chat history
    for question, answer in st.session_state['chat_history']:
        st.markdown(f"**You:** {question}")
        st.markdown(f"**Gemini Pro:** {answer}")
else:
    st.warning("Please upload PDF files to begin.")

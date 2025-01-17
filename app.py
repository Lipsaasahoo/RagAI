
import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from groq import Groq  # Import Groq for using Groq API

# Load environment variables from the .env file
load_dotenv()

# Fetch the API key from the environment variable (if applicable)
api_key = os.getenv("GROQ_API_KEY")

# Initialize directories if they don't exist
if not os.path.exists('pdfFiles'):
    os.makedirs('pdfFiles')

if not os.path.exists('vectorDB'):
    os.makedirs('vectorDB')

# Initialize state variables for session state
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

   Context: {context}
   History: {history}

   User: {question}
   Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question",
    )

# Initialize Groq client for LLM usage
if 'groq_client' not in st.session_state:
    st.session_state.groq_client = Groq(api_key=api_key)

# Initialize Chroma vector store and Groq model
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(persist_directory='vectorDB',
                                           embedding_function=OllamaEmbeddings(base_url='http://localhost:8502', model="llama2"))

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(
    page_title="Standards.io",
    page_icon="🗎",
    layout="centered"
)

# App layout
st.title("Find the Codes & Standards")

# Upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# Handle file upload and processing
if uploaded_file is not None:
    st.text("File uploaded successfully")
    file_path = 'pdfFiles/' + uploaded_file.name
    if not os.path.exists(file_path):
        with st.status("Saving file..."):
            bytes_data = uploaded_file.read()
            with open(file_path, 'wb') as f:
                f.write(bytes_data)

            # Load PDF and split into chunks
            loader = PyPDFLoader(file_path)
            data = loader.load()

            # Split text into chunks for embedding
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )
            all_splits = text_splitter.split_documents(data)

            # Create and persist vector store
            st.session_state.vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=OllamaEmbeddings(model="llama2")
            )
            st.session_state.vectorstore.persist()

    # Initialize retriever
    st.session_state.retriever = st.session_state.vectorstore.as_retriever()

    # Handle user input and response
    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)

        # Create a variable to accumulate the response
        full_response = ""

        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                # Use Groq for chat completion
                completion = st.session_state.groq_client.chat.completions.create(
                    model="llama3-8b-8192",  # Groq model
                    messages=[{"role": "user", "content": user_input}],
                    temperature=1,
                    max_tokens=1024,
                    top_p=1,
                    stream=True,
                    stop=None
                )
                
                # Handle the streaming response
                for chunk in completion:
                    full_response += chunk.choices[0].delta.content or ""
                    time.sleep(0.05)  # Simulate typing delay

                # Display the full response after completion
                st.markdown(full_response)

        # Append assistant's response to chat history
        chatbot_message = {"role": "assistant", "message": full_response}
        st.session_state.chat_history.append(chatbot_message)

else:
    st.write("Please upload a PDF file to start the chatbot")

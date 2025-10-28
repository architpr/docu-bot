import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
from gtts import gTTS
import io

# Import the LangChain components
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# --- Helper Functions ---

def text_to_speech(text):
    """Converts text to speech and returns the audio bytes."""
    try:
        tts = gTTS(text=text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.read()
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {e}")
        return None

def setup_rag_chain(source_docs, model_name):
    """
    Sets up the RAG chain based on a list of loaded documents and a model name.
    """
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(source_docs)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create Chroma vector store
    vector_store = Chroma.from_documents(texts, embeddings)
    
    # Create the retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Initialize the LLM
    llm = ChatGroq(model_name=model_name, temperature=0.7)
    
    # Create the RAG chain
    # --- NEW FEATURE: Show Sources ---
    # We add `return_source_documents=True` to get the source chunks.
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True
    )

# --- Main App Logic ---

def main():
    st.set_page_config(page_title="Chat With Anything", layout="wide")
    st.title("📄 Chat With Anything: Docs, PDFs, or Websites")
    st.write("Upload your documents or provide a website URL to start chatting with your knowledge base.")

    # --- NEW FEATURE: Let Users Choose the AI Model ---
    # Initialize session state for the model if it doesn't exist
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "llama-3.1-8b-instant"

    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Model selection dropdown
        st.session_state.selected_model = st.selectbox(
            "Choose your AI model:",
            ("llama-3.1-8b-instant", "gemma2-9b-it", "llama-3.1-70b-versatile"),
            index=0 # Default to the first model
        )
        
        st.markdown("---")
        
        # --- NEW FEATURE: Chat with a Website ---
        st.header("🌐 Chat with a Website")
        url_input = st.text_input("Enter a website URL:")
        if st.button("Process URL"):
            if url_input:
                with st.spinner(f"Fetching and processing content from {url_input}..."):
                    try:
                        loader = WebBaseLoader(url_input)
                        docs = loader.load()
                        st.session_state.qa_chain = setup_rag_chain(docs, st.session_state.selected_model)
                        st.session_state.chat_history = [] # Reset chat history
                        st.success("Website processed! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Failed to process URL: {e}")
            else:
                st.warning("Please enter a URL.")
        
        st.markdown("---")
        
        # File uploader for documents
        st.header("📂 Chat with Documents")
        uploaded_files = st.file_uploader(
            "Upload files (.pdf or .txt)", 
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing your documents..."):
                    all_docs = []
                    temp_dir = tempfile.TemporaryDirectory()
                    for uploaded_file in uploaded_files:
                        temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
                        with open(temp_filepath, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        if temp_filepath.endswith(".pdf"):
                            loader = PyPDFLoader(temp_filepath)
                        else: # .txt
                            loader = TextLoader(temp_filepath)
                        all_docs.extend(loader.load())
                        
                    st.session_state.qa_chain = setup_rag_chain(all_docs, st.session_state.selected_model)
                    st.session_state.chat_history = [] # Reset chat history
                    temp_dir.cleanup()
                    st.success("Documents processed! You can now ask questions.")
            else:
                st.warning("Please upload at least one document.")
    
    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if "qa_chain" in st.session_state:
        if user_question := st.chat_input("Ask a question about your content..."):
            # Add user question to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # Get the answer
            with st.spinner("Finding the answer..."):
                try:
                    result = st.session_state.qa_chain.invoke(user_question)
                    answer = result['result']
                    
                    # Add AI answer to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        
                        # --- NEW FEATURE: Text-to-Speech for Answers ---
                        audio_bytes = text_to_speech(answer)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3", autoplay=True)

                    # --- NEW FEATURE: Show Sources ---
                    with st.expander("View Sources"):
                        for doc in result['source_documents']:
                            st.info(f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}")
                            st.markdown(f"> {doc.page_content}")
                            
                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
    else:
        st.info("Please process a website or documents to start the chat.")


if __name__ == "__main__":
    main()
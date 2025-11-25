# 📄 Chat With Anything AI 🤖

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green?logo=langchain)](https://www.langchain.com/)

An intelligent chatbot that can read and answer questions from multiple documents, PDFs, or even live websites. This app serves as a powerful, interactive knowledge base for any set of documents.
deployed site : [https://chat-doc-bot-u2.streamlit.app/]
---


## 🚀 Key Features

* **Multi-Source Chat:** Upload multiple `.pdf` and `.txt` files or scrape a live website to build a knowledge base.
* **Show Your Sources:** The AI cites its sources, showing you the exact text chunks from the documents used to generate an answer.
* **Text-to-Speech:** Hear the chatbot's answers spoken out loud automatically.
* **Selectable AI Models:** Choose from different powerful language models (like Llama 3.1 and Gemma 2) to customize your chat experience.
* **Interactive UI:** A clean and user-friendly interface built with Streamlit.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **AI/LLM Orchestration:** [LangChain](https://www.langchain.com/)
* **LLM Provider:** [Groq](https://groq.com/)
* **Embedding & Vector Store:** Hugging Face `sentence-transformers` & `ChromaDB`
* **Core Language:** Python

---

## ⚙️ Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

* Python 3.9+
* A Groq API Key (get one for free at [groq.com](https://groq.com/))

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)https://github.com/architpr/docu-bot
    cd docu-bot-streamlit
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API key:**
    * Create a file named `.env` in the root of the project.
    * Add your Groq API key to it like this:
        ```
        GROQ_API_KEY="your-secret-api-key-goes-here"
        ```

### Running the Application

1.  **Launch the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
2.  Your web browser will automatically open with the application running.

---

## 📖 Usage

1.  **Choose an AI Model** from the dropdown in the sidebar.
2.  **Provide Content:**
    * **To chat with files:** Use the file uploader to select one or more `.pdf` or `.txt` files and click "Process Documents."
    * **To chat with a website:** Paste a URL into the text box and click "Process URL."
3.  **Ask Questions:** Once the content is processed, the chat interface will appear. Type your questions and get instant, source-backed answers!


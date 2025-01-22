# Free Palestine Project

This project implements a **question-answering system** that utilizes **Ollama's LLMs**, **Streamlit**, and **Langchain** to provide detailed answers based on a dataset of text documents. The system retrieves relevant information from a local Chroma database and generates responses using AI-powered models.

## Libraries Used

1. **`os`**:
   - Provides a way to interact with the file system (e.g., check if directories exist).

2. **`ollama`**:
   - A library for interacting with Ollama's language models, such as `Llama3`, to generate AI-driven responses to questions.

3. **`streamlit`**:
   - A framework for building interactive web applications. In this project, it's used to create the user interface (UI) for asking questions and displaying answers.

4. **`PIL` (Pillow)**:
   - A library for working with images. It's used in this project to display a flag image in the sidebar of the Streamlit app.

5. **`langchain.text_splitter`**:
   - Provides utilities to split long text into smaller chunks for processing. The `RecursiveCharacterTextSplitter` class is used to break the text into manageable pieces.

6. **`langchain_community.vectorstores`**:
   - The `Chroma` library is used to create a **vector store** that holds the embeddings of the documents. The vector store helps find similar documents to a given query.

7. **`langchain_community.embeddings`**:
   - **`OllamaEmbeddings`**: This class generates text embeddings using the Ollama language model (`nomic-embed-text`). These embeddings help represent text in a high-dimensional space, allowing for efficient similarity searches.

8. **`langchain.schema`**:
   - The `Document` class represents a chunk of text. It’s used to structure the text into objects that can be embedded and indexed in the vector store.

## Workflow

### 1. **Loading and Processing Text**
- The system loads a text file (`dataset.txt`), splits it into smaller chunks (using `RecursiveCharacterTextSplitter`), and stores these chunks in a list of `Document` objects. These chunks are then converted into embeddings using the `OllamaEmbeddings` model.

### 2. **Setting Up Vectorstore**
- If a **Chroma database** (`chroma_db`) already exists, the system loads the database and uses it. If the database doesn’t exist, it processes the text, creates embeddings, stores them in the vector store, and persists the database for future use.

### 3. **Retriever and RAG (Retrieval-Augmented Generation)**
- A **retriever** finds relevant documents from the vector store based on a user's query. The retrieved documents are formatted and used in a prompt template to generate a response using the `ollama.chat` API (powered by the **Llama3** model).

### 4. **Streamlit Interface**
- A simple web interface is built using **Streamlit**. It allows users to input questions. Once a question is submitted, the app uses the **RAG chain** to retrieve relevant information and generate an answer. The app also displays an image of the Palestine flag in the sidebar.

## Key Functions

- **`load_text_file(file_path)`**: Loads the content of a text file into a string.
- **`format_docs(docs)`**: Formats the list of retrieved documents into a string to pass to the AI model.
- **`generate_message_template(question, context)`**: Generates a structured prompt for the AI model, combining the user’s question with the retrieved context.
- **`ollama_llm(question, context)`**: Sends a request to the Ollama LLM model and generates a response based on the question and context.
- **`rag_chain(question)`**: Retrieves relevant documents and generates a response using the RAG process.
- **Streamlit Widgets**: The interface includes a question input field, a submit button, and a space to display the answer, as well as an image of the Palestine flag.

## Installation

To set up and run this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mohamed-Ali-Farhat/Palestine_Will_Be_Free


2. **Run the Streamlit app**:
To start the app, run the following command:
    ```bash
    streamlit run app.py
   


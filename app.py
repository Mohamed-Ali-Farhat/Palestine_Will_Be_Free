import os
import ollama
import streamlit as st
from PIL import Image  # Import for handling images
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document

# Function to load content from a text file
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# Path to the text file and Chroma database directory
text_file_path = "dataset.txt"
chroma_db_dir = "chroma_db"

# Check if Chroma database already exists
if os.path.exists(chroma_db_dir):
    vectorstore = Chroma(persist_directory=chroma_db_dir, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
else:
    text_content = load_text_file(text_file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text_content)
    documents = [Document(page_content=split) for split in splits]
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=chroma_db_dir)
    vectorstore.persist()

retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_message_template(question, context):
    return f"""
    Please respond in English and provide a detailed answer. Avoid saying "I don't know" or starting your response with phrases like "Based on the provided context" or any similar expressions.
    
    Question:
    {question}
    
    Context:
    {context}
    """



def ollama_llm(question, context):
    formatted_prompt = generate_message_template(question, context)
    messages = [{'role': 'user', 'content': formatted_prompt}]
    response = ollama.chat(
        model='llama3:latest',
        messages=messages
    )
    return response['message']['content']

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    response = ollama_llm(question, formatted_context)
    return response

# Streamlit Interface
st.title("From the river to the sea , Palestine will be free")



# Add image to the front left
image_path = "Flag_Palestine.png"  # Replace with your image file path
image = Image.open(image_path)  # Load image
st.sidebar.image(image, use_column_width=True)  # Display image in the sidebar

question = st.text_input("Enter your question:")
if st.button("Submit"):
    if question.strip():
        with st.spinner("Retrieving and generating answer..."):
            result = rag_chain(question)
        st.subheader("Answer:")
        st.write(result)
    else:
        st.warning("Please enter a question.")

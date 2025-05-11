import os

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from lagnchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = "figtures/"
figures_directory = "pdfs/"

embeddings = OllamaEmbeddings(model = "llama3.2")
vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model = "gemma3:4b")

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.get.buffer())
    
def load_pdf(file_path):
    elements = partition_pdf(
        file_path,
        strategy = PartitionStrategy.HI_RES,
        extract_image_block_types = ['Image', 'Table'],
        extract_image_block_output_dir = figures_directory
    )
    
    text_elements = [element.text for element in elements if element.category not in ['Image', 'Table']]
    
    for file in os.listdir(figures_directory):
        extracted_text = extract_text(figures_directory + file)
        text_elements.append(extracted_text)
    
    return "\n\n".join()

def extract_text(file_path):
    model_with_image_context = model.bind(images = [file_path])
    return model_with_image_context.invoke("Tell me what do you see in this picture")

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    
    return text_splitter.split_text(text)

def index_docs(texts):
    vector_store.add_texts(texts)
    
def retrieve_docs(texts):
    return vector_store.similarity_search(texts)

def answer_question(question, documents):
    contexts = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    return chain.invoke({"question": question, "context": contexts})

uploaded_file = st.file_uploader(
    "Upload a PDF file", 
    type = "pdf",
    accept_multiple_files = False,    
)

if uploaded_file:
    upload_pdf(uploaded_file)
    text = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_texts = split_text(text)
    index_docs(chunked_texts)
    
    question = st.chat_input()
    
    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_docs(question)
        answer = answer_question(question, related_documents)
        
        st.chat_message("assistant").write(answer)
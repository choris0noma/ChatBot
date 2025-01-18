import streamlit as st
import ollama
import chromadb
from chromadb.config import Settings
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama
from io import StringIO
import uuid
import pdfplumber
import os
import re

logging.basicConfig(level=logging.INFO)


client = chromadb.PersistentClient(
    path='database/',
    settings=Settings()
)

history = client.get_or_create_collection(name="history")
contextCollection = client.get_or_create_collection(name="context")
constitutionCollection = client.get_or_create_collection(name="constitution")

if st.sidebar.button("Clear Files"):
    contextCollection.delete(contextCollection.get()['ids'])
    st.success("files cleared successfully.")
    
if st.sidebar.button("Clear History"):
    history.delete(history.get()['ids'])
    st.success("history cleared successfully.")

uploaded_files = st.file_uploader("Choose a txt file", accept_multiple_files=True)
docs = []
ids = []

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    doc_content = stringio.getvalue()
    docs.append(doc_content)
    unique_id = str(uuid.uuid4())
    ids.append(unique_id)
    st.write("Filename:", uploaded_file.name)
    st.write(bytes_data)

if len(docs) > 0:
    contextCollection.add(
        documents=docs,
        ids=ids
    )
    st.success("Documents added to context collection.")


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_articles(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    
    articles = re.split(r"(Article \d+)", text)
    
    parsed_articles = {}
    for i in range(1, len(articles), 2):
        article_number = articles[i].strip()
        article_content = articles[i + 1].strip()  
        if article_content:
            full_article = f"{article_number} {article_content}"
            parsed_articles[article_number] = full_article
    
    if not parsed_articles:
        logging.warning("No articles were extracted. Please check the constitution text format.")
    
    return parsed_articles



def store_constitution(articles):
    constitutionCollection.add(
        ids=list(articles.keys()), 
        documents=list(articles.values()) 
    )
    
pdf_path = "constitution.pdf"
txt_path = "constitution.txt"

if not os.path.exists(txt_path) or os.stat(txt_path).st_size == 0 or len(constitutionCollection.get()['ids']) <=0:
    constitution_text = extract_text_from_pdf(pdf_path)
    if constitution_text.strip():
        with open(txt_path, "w", encoding="utf-8") as file:
            file.write(constitution_text)
        articles = extract_articles(txt_path)
        store_constitution(articles)
    else:
        logging.error("Failed to extract text from the PDF. Please check the file.")

def get_chat_history(history):
    try:
        all_documents = history.get()
        ids = all_documents['ids']
        documents = all_documents['documents']
        sorted_data = sorted(zip(ids, documents), key=lambda x: int(x[0].split('-')[1]))
        
        chat_history = []
        for id_, doc in sorted_data:
            if id_.startswith("user-"):
                chat_history.append({"role": "user", "content": doc})
            elif id_.startswith("assistant-"):
                chat_history.append({"role": "assistant", "content": doc})
        
        return chat_history
    except Exception as e:
        logging.error(f"Error retrieving chat history: {str(e)}")
        return []


def query_context(prompt, n_results=1, targetCollection = contextCollection):
    try:
        results = targetCollection.query(
            query_texts=[prompt],
            n_results=n_results
        )
        return results["documents"] if results["documents"] else ["No relevant documents found."]
    except Exception as e:
        logging.error(f"Error querying context: {str(e)}")
        return ["No relevant documents found."]


def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        prompt = messages[-1].content
        
        user_embedding_response = ollama.embeddings(
            prompt=prompt,
            model="mxbai-embed-large"
        )
        user_embedding = user_embedding_response["embedding"]
        retrieved_docs = []
        if model == "constitModel":
            retrieved_docs = query_context(prompt, 10, constitutionCollection)
            print(retrieved_docs[0])
        else:
            length = len(contextCollection.get()['ids'])
            if length > 0:
                retrieved_docs = query_context(prompt, length)
        
        context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."
        messages[-1].content = f"Context: {context}\n\nQuestion: {prompt}\nAnswer:"
        
        resp = llm.stream_chat(messages)
        assistant_response = ""
        response_placeholder = st.empty()
        for r in resp:
            assistant_response += r.delta
            response_placeholder.write(assistant_response)
            
        assistant_embedding_response = ollama.embeddings(
            prompt=assistant_response,
            model="mxbai-embed-large"
        )
        assistant_embedding = assistant_embedding_response["embedding"]
        
        existing_count = len(history.get()['ids']) // 2
        history.add(
            ids=[f"user-{existing_count+1}", f"assistant-{existing_count+1}"],
            embeddings=[user_embedding, assistant_embedding],
            documents=[prompt, assistant_response]
        )

        logging.info(f"Model: {model}, Messages: {messages}, Response: {assistant_response}")
        return assistant_response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e


def main():
    st.title("The ChatBot")
    logging.info("App started")

    model = st.sidebar.selectbox("Choose a model", ["llama3.2", "constitModel"])
    logging.info(f"Model selected: {model}")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    chat_history = get_chat_history(history)
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                with st.spinner("Writing..."):
                    try:
                        messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
                        response_message = stream_chat(model, messages)
                        duration = time.time() - start_time
                        response_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
                        st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("An error occurred while generating the response.")

if __name__ == "__main__":
    main()

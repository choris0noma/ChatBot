import streamlit as st
import ollama
import chromadb
from chromadb.config import Settings
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama

logging.basicConfig(level=logging.INFO)

client = chromadb.PersistentClient(
    path='database/',
    settings=Settings()
)
collection = client.get_or_create_collection(name="history")

def get_chat_history(collection):
    try:
        all_documents = collection.get()
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

def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
       
        user_message = messages[-1].content
        user_embedding_response = ollama.embeddings(
            prompt=user_message,
            model="mxbai-embed-large"
        )
        user_embedding = user_embedding_response["embedding"]
        
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
        
        existing_count = len(collection.get()['ids']) // 2
        collection.add(
            ids=[f"user-{existing_count+1}", f"assistant-{existing_count+1}"],
            embeddings=[user_embedding, assistant_embedding],
            documents=[user_message, assistant_response]
        )

        logging.info(f"Model: {model}, Messages: {messages}, Response: {assistant_response}")
        return assistant_response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

def main():
    st.title("The ChatBot")
    logging.info("App started")

    model = st.sidebar.selectbox("Choose a model", ["llama3.2"])
    logging.info(f"Model selected: {model}")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    chat_history = get_chat_history(collection)
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

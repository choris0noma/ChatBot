import streamlit as st
import ollama
import chromadb
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama

logging.basicConfig(level=logging.INFO)
documents = [
  "student's name is Danial",
  "student's age is 18",
  "student's major is Computer Science",
  "student's hobby is game developing"
]

client = chromadb.PersistentClient(
    path='database/',
    settings = Settings()
)
collection = client.get_or_create_collection(name="subaru")

for i, d in enumerate(documents):
  response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
  embedding = response["embedding"]
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[d]
  )

if 'messages' not in st.session_state:
    st.session_state.messages = []

def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        
        response = ollama.embeddings(
            prompt=messages[-1].content,
            model="mxbai-embed-large"
        )
        queryResults = collection.query(
            query_embeddings=[response["embedding"]],
            n_results=1
        )
        data = queryResults['documents'][0][0]
        
        messages.append(ChatMessage(role="assistant", content=f"Using this data: {data}"))
        
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

def main():
    st.title("The ChatBot")
    logging.info("App started")

    model = st.sidebar.selectbox("Choose a model", ["llama3.2"])
    logging.info(f"Model selected: {model}")

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

import streamlit as st
import chromadb
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama


configuration = {
    "client": "PersistentClient",
    "path": "/tmp/.chroma"
}
collection_name = "documents_collection"
embedding_function_name = "DefaultEmbedding"
documents = [
    "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels.",
    "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands.",
    "Llamas can grow as much as 6 feet tall though the average llama is between 5 feet 6 inches and 5 feet 9 inches tall.",
    "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight.",
    "Llamas are vegetarians and have very efficient digestive systems.",
    "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old."
]

conn = st.connection("chromadb", type=ChromadbConnection, **configuration)
conn.create_collection(
    collection_name=collection_name,
    embedding_function_name=embedding_function_name
)
conn.upload_documents(
    collection_name=collection_name,
    documents=documents,
    embedding_function_name=embedding_function_name
)

if 'messages' not in st.session_state:
    st.session_state.messages = []

def stream_chat(model: str, messages: list[ChatMessage]):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        collection_data = conn.get_collection_data(
            collection_name=collection_name,
            attributes=["documents", "embeddings"]
        )
        context = "\n".join(collection_data["documents"])
        prompt = f"Using this data: {context}. Respond to this prompt: {messages[-1].content}"
        response = ""
        response_placeholder = st.empty()
        for r in llm.stream_chat(prompt):
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

    model = st.sidebar.selectbox("Choose a model", ["mymodel", "llama3.2", "phi3", "mistral"])
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

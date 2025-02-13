import streamlit as st
import ollama
import chromadb
from chromadb.config import Settings
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama
import os
from langchain_community.vectorstores import Chroma
#from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'
pdf_path = "constitution.pdf"
constitution_db_path = 'datbabse/constitution/'
history_db_path = 'datbabse/history/'
embedding=OllamaEmbeddings(model="mxbai-embed-large",)
text_splitter = SemanticChunker(embedding)

logging.basicConfig(level=logging.INFO)

uploaded_files = st.file_uploader("Choose a txt file", accept_multiple_files=True, type=['pdf', 'txt'])
ragFusionOn = st.sidebar.toggle("rag fusion")
import pdfplumber

def load_files():
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    documents = []
    
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension == ".pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        else:
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

        documents.append(Document(page_content=text))

    try:
        return Chroma.from_documents(documents=documents, embedding=embedding)
    except Exception as e:
        logging.error(f"Error creating Chroma vectorstore: {str(e)}")


def load_constitution():
    if os.path.exists(constitution_db_path) and os.listdir(constitution_db_path):
        return Chroma(persist_directory=constitution_db_path, embedding_function=embedding)
    
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    documents = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embedding,
        persist_directory=constitution_db_path
    )
    return vectorstore

def load_history():
    return Chroma(persist_directory=history_db_path, embedding_function=embedding)

def history_to_chat(vectorstore):
    try:
        results = vectorstore.get()
        if results and "documents" in results:
            messages = [Document(page_content=doc) for doc in results["documents"]]
            
            for message in messages:
                cleaned_content = message.page_content.lstrip("user: ").lstrip("assistant: ")
                with st.chat_message("user" if "user" in message.page_content else "assistant"):
                    st.markdown(cleaned_content)

            st.session_state.messages = [{"role": "user" if "user" in msg.page_content else "assistant", 
                                          "content": msg.page_content.lstrip("user: ").lstrip("assistant: ")} 
                                         for msg in messages]
    except Exception as e:
        logging.error(f"Error retrieving history: {str(e)}")

files_vectorstore = load_files()
constitution_vectorstore = load_constitution()
history_vectorstore = load_history()

def generate_queries(question,model, n=5):
    llm = Ollama(model=model, request_timeout=60.0)
    
    prompt = f"""You are a helpful assistant. Generate {n} search queries related to: {question}.
    Output {n} variations, each on a new line."""
    
    response = llm.complete(prompt)
    queries = response.text.split("\n")

    
    return [q.strip() for q in queries if q.strip()]  


def reciprocal_rank_fusion(results, k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [loads(doc[0]) for doc in reranked_results]

def query_context(prompt,model, n_results=5, vectorstore=None):
    if vectorstore is None:
        return ["No context found or provided."]
    
    try:
        if ragFusionOn:
            queries = generate_queries(prompt,model)
            st.sidebar.write(queries)
            retrieved_results = [vectorstore.as_retriever(search_kwargs={"k": n_results}).invoke(q) for q in queries]
            reranked_docs = reciprocal_rank_fusion(retrieved_results)
        else:
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": n_results})
            reranked_docs = retriever.invoke(prompt)

        if not isinstance(reranked_docs, list) or not all(hasattr(doc, "page_content") for doc in reranked_docs):
            logging.error("Invalid retrieved documents format.")
            return ["No relevant documents found."]
        
        return reranked_docs
    except Exception as e:
        logging.error(f"Error querying context: {str(e)}")
        return ["No relevant documents found."]


def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        prompt = messages[-1].content

        if model == "constitModel":
            retrieved_docs = query_context(prompt,model, 10, constitution_vectorstore)
        else:
            retrieved_docs = query_context(prompt,model, 10, files_vectorstore)

        context = " ".join(doc.page_content for doc in retrieved_docs if hasattr(doc, "page_content"))

        messages[-1].content = f"Context: {context}\n\nQuestion: {prompt}\nAnswer:"
        resp = llm.stream_chat(messages)
        
        assistant_response = ""
        response_placeholder = st.empty()
        for r in resp:
            assistant_response += r.delta
            response_placeholder.write(assistant_response)

        logging.info(f"Model: {model}, Messages: {messages}, Response: {assistant_response}")

        history_vectorstore.add_documents([
            Document(page_content=f"user: {prompt}"),
            Document(page_content=f"assistant: {assistant_response}")
        ])

        return assistant_response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        return "An error occurred."




def main():
    st.title("The ChatBot")
    logging.info("App started")

    model = st.sidebar.selectbox("Choose a model", ["llama3.2", "constitModel"])
    logging.info(f"Model selected: {model}")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    history_to_chat(history_vectorstore)
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
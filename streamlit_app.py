import streamlit as st
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

st.title('🎈 Ask the doc app')

f = st.file_uploader(label="Upload a document", type=".txt")


question = st.text_input("Enter your question:")


def generate_response(file, question):
    if file is not None: 
        text = [file.read().decode()]
        text_splitter = TokenTextSplitter(chunk_size= 1000, chunk_overlap=0)
        chunks = text_splitter.create_documents(text)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma.from_documents(chunks, embeddings)
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        return qa.run(question)

with st.form("my_form"):
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    submitted = st.form_submit_button("Submit")

    if (submitted & (not openai_api_key.startswith('sk-'))):
        st.warning("Please enter a valid OpenAI API key")
    elif (submitted & openai_api_key.startswith("sk-")):
        answer = generate_response(f, question)
        st.info(answer)


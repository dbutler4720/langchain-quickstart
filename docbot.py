# this version was created by OpenAI GPT 3.5 instruct
# several iterations due to errors and didn't get it to run yet.

# To Run
#   streamlit run docbot2.py
# no longer needed
#   streamlit run docbot2.py --server.enableCORS false --server.enableXsrfProtection false
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate 
#from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv('.env', override=True)

st.write("Hello world")
st.set_page_config(page_title="DocBot", page_icon="🤖")

st.sidebar.title("Upload Document")
doc_text = st.sidebar.text_area("Paste or upload document text")

qa_llm = OpenAI(temperature=0)

qa_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Answer the question based on the context below:\n\n{context}\n\nQuestion: {question}\nAnswer:",
)

st.title("Ask DocBot a Question")
question = st.text_input("Enter your question:")

if question:
    response = qa_llm.generate(prompt=qa_template, context=doc_text, question=question)
    st.write("### DocBot:")
    st.write(response.output)

    st.write("### Human:")
    st.write(question)
    
    st.write("### DocBot:")
    st.write(response.output)
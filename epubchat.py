# This version was pulled from
# https://discuss.streamlit.io/t/langchain-tutorial-5-build-an-ask-the-data-app/47672
# and modified to work with the new langchain package.
# Used Cody to fix a few bugs and updated to latest langchain package
# Not working yet
import os
import tempfile
import uuid
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI


import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

#from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
#from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
import pypandoc

DB_DIR = './data'

def main():
  load_dotenv('.env', override=True)
  pypandoc.download_pandoc()
  streamlit_app()

# Generate LLM response
import langchain

def chat_with_epub(epub_path, question:str):
  # Load epub data
  loader = UnstructuredEPubLoader(epub_path)
  data = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  documents = text_splitter.split_documents(data)

  # we are specifying that OpenAI is the LLM that we want to use in our chain
  chain = load_qa_chain(llm=OpenAI())
  query = 'What is this book about?'
  response = chain.invoke({"input_documents": documents, "question": query})
  return st.success(response["output_text"])

  # create the open-source embedding function
  # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

  # load it into Chroma
  # db = Chroma.from_documents(documents, embedding_function)

  # query it
  # query = "What is this book about"
  # docs = db.similarity_search(query)

#   # print results
  # return st.success((docs[0].page_content))

#   embeddings = OpenAIEmbeddings()
  
#   # we create our vectorDB, using the OpenAIEmbeddings tranformer to create
#   # embeddings from our text chunks. We set all the db information to be stored
#   # inside the ./data directory, so it doesn't clutter up our source files
#   vectorstore = chromadb.PersistentClient()
#   collection = vectorstore.create_collection(name="my_collection")
#   collection.add(documents=documents, embedding=embeddings)
#   vectorstore.persist()
#   # vectordb = Chroma.from_documents(
#   #   collection_id = str(uuid.uuid4()),
#   #   documents=documents,
#   #   embedding=OpenAIEmbeddings(),
#   #   persist_directory='./data'
#   # )
#   # vectordb.persist()
  
#   qa_chain = RetrievalQA.from_chain_type(
#     llm=OpenAI(),
#     retriever=vectorstore.as_retriever(search_kwargs={'k': 7}),
#     return_source_documents=True
#   )

#   # we can now execute queries against our Q&A chain
#   result = qa_chain.invoke({'query': question})
# #   print(result['result'])
# #   messages = [
# #     SystemMessage(
# #         content="You are an assistant that answers questions about the context you provide."
# #     ),
# #     HumanMessage(
# #         content=question
# #     )
# #   ]

# #   # Create chat agent
# #   chat_agent = ChatOpenAI(temperature=0)

# #   # Get chat response using epub context
# # #   response = chat_agent(messages)
# #   response = chat_agent.generate(context=context, prompt=question)
# #   #os.remove(epub_path)

  return st.success(result)


def streamlit_app():
  # Page title
  st.set_page_config()
  st.title('🦜🔗 Chat with an e-book')

  # Input widgets
  uploaded_file = st.file_uploader('Upload an EPUB file', type=['epub'])
  if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    epub_path = tfile.name

  question_list = [
    'What is this book about?',
    'Other']
  query_text = st.selectbox('Select an example query:', question_list, disabled=not uploaded_file)

  # App logic
  if query_text is 'Other':
    query_text = st.text_input('Enter your query:', placeholder = 'Enter query here ...', disabled=not uploaded_file)
  if uploaded_file is not None:
    st.header('Output')
    chat_with_epub(epub_path, query_text)

if __name__ == "__main__":
  main()
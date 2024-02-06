# This version was pulled from
# https://discuss.streamlit.io/t/langchain-tutorial-5-build-an-ask-the-data-app/47672
# and modified to work with the new langchain package.
# Used Cody to fix a few bugs and updated to latest langchain package
# Not working yet
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv

load_dotenv('.env', override=True)

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Data App')
st.title('🦜🔗 Ask the Data App')

# Load CSV file
def load_csv(input_csv):
  df = pd.read_csv(input_csv)
  with st.expander('See DataFrame'):
    st.write(df)
  return df

# Generate LLM response
def generate_response(csv_file, input_query):
  df = load_csv(csv_file)
  
  # Create Pandas DataFrame Agent
  agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors="Check your output and make sure it conforms!",
  )
  
  response = agent.run(input_query)
  return st.success(response)

# Input widgets
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
question_list = [
  'How many rows are there?',
  'How many columns are there?',
  'What is the average value for MolWt?',
  'What is the range of values for MolWt with logS greater than 0?',
  'How many rows have MolLogP value greater than 0.',
  'Other']
query_text = st.selectbox('Select an example query:', question_list, disabled=not uploaded_file)

# App logic
if query_text is 'Other':
  query_text = st.text_input('Enter your query:', placeholder = 'Enter query here ...', disabled=not uploaded_file)
if uploaded_file is not None:
  st.header('Output')
  generate_response(uploaded_file, query_text)
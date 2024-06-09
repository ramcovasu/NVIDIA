import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI  # Assuming OpenAI is your local LLM integration
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from langchain.tools import tool
from groq import Groq
from langchain_groq import ChatGroq
#from tools.custom_tools import CustomTools
import pandas as pd
import time
import datetime
import subprocess
import json


load_dotenv()

# Point to the local server (replace with your actual details)
client = OpenAI(base_url="http://localhost:1234/v1", api_key="Not required")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

functions= [  
    {
        "name" : "send_email",
        "api_name": "send_email",
        "description": "template to have an email sent.",
        "parameters": [
             {
                "name": "to_address",
                "description":  "To address for email"
             },   
             {
                "name": "subject",
                "description": "the subject of the email"
             }
        ]
    }
]
myllm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-70b-8192"
        )
def get_gorilla_response(prompt):
  #print("I am inside get gorilla")
  #client = OpenAI(base_url="http://localhost:1234/v1", api_key="Not required")
  input_prompt = f'{prompt} <<function>> {json.dumps(functions)}'
  completion = client.chat.completions.create(
        model="gorilla-llm/gorilla-openfunctions-v2-gguf/gorilla-openfunctions-v2-q6_K.gguf:2",  # Replace with your local LLM model name
        messages=[
            {"role": "system", "content": "You need to be able to create the right function call with right parameters based on the content provided."},
            {"role": "user", "content": f"{input_prompt}"}
        ],
        temperature=0,
  )
  #print(completion.choices[0].message.content)
  return completion.choices[0].message.content

def run_generated_code(file_path):

    # Command to run the generated code using Python interpreter
    command = ["python", file_path]
    
    try:
        # Execute the command as a subprocess and capture the output and error streams
        # result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result = 0
        # Check if the subprocess ran successfully
        if result.returncode == 0:
            st.success("Generated code executed successfully.")
            # Display the output of the generated code
            st.code(result.stdout, language="python")
        else:
            st.error("Generated code execution failed with the following error:")
            # Display the error message
            st.code(result.stderr, language="bash")
    
    except Exception as e:
        st.error("Error occurred while running the generated code:", e)      

def get_pdf_text(pdf_docs):
  text = ""
  for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()
  return text

def get_text_chunks(text):
  chunks = ""
  issues = text.split("Issue: ")
  issues = [issue.strip() for issue in issues if issue.strip()]
  return issues

def create_faiss_index(text_chunks):
  # Assuming you have a pre-trained embedding model available locally
  # Replace 'your_embedding_model' with the actual model path
  #embeddings = FAISS.read_DenseEmbedding(data=text_chunks, dtype=np.float32, path='embedding_model')
  vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
  vector_store.save_local("faiss_index")


# Anomaly detection done by this agent
class DeviceAgent():
  def __init__(self):
      """ test"""
                    
  def Anomaly_agent(self,df):
    #anomaly_dataset = f"Device dataset: {', '.join(df.columns.tolist())}"
    
    return Agent(
                        role='Anomaly Detection Agent',
                        goal='You are a Data Expert and you need to find the Anomaly in the dataset provided, pay attention and you cannot make a mistake here.Let us think step by step. Take your time.',
                        backstory="""You are an expert in detecting Anomaly in IOT Devices data, you have to find any anomalies and provide a one line summary of the anomaly""",
                        verbose=False,
                        llm=myllm,
                        memory=True,
                        allow_delegation=False,
                        tools=[]
                    )
 
  def RCA_agent(self,anomaly_data,resolution):
    
    return Agent(
                        role='Root cause Analysis Agent',
                        goal='You have to create a good summary of the issue and the resolution based on the input provided to you',
                        backstory=f"""You are a Root cause Analsis agent, you have to provide a great summary of the issue {anomaly_data} and the resolution {resolution}. Like Step 1: details , Step 2: details""",
                        verbose=True,
                        llm=myllm,
                        allow_delegation=True,
                        memory=True,
                        tools=[]
                    )
  
  
class DeviceTask():
  
 
  def Anomaly_Task(self, agent,df,df_anomaly):
    
    #anomaly_dataset = f"Device dataset: {', '.join(df.columns.tolist())}"
    return Task(
    agent=agent,
    description=f"""You are an expert in detecting anomalies in IoT device data.  Follow these steps to detect if there is anomaly in record.
              Step 1: Go through the sample Anomaly data record by record which is provided in this file {df_anomaly}, understand each column and its value. Each record is an example of how to identify an anomaly record. The column model_output gives you the reason why this record is an anomaly, you need to learn from this data set and then apply the same in the real data set" 
              Let us think step by step. Take your time. Start with each row and try to check each field. Repeating values is not Anomaly.
              You need to be 100% accurate.
              Here's the real dataset to analyze: {df}. """,
    expected_output="Anomaly Found in the dataset and provide the anomaly data with the attribute name and value",
    context=[]
)
            
  def RCA_Task(self, agent, df,data_anomaly,resolution):
    
    return Task(
            agent=agent,
            description=f"You are a Root cause Analsis agent, you have to use both the anomaly data and the resolution steps and provide a great summary. The anomaly dataset is {data_anomaly}, the resolution steps are  {resolution}. The output should be like Like Step 1: details , Step 2: details",
            expected_output="Great summary in the form of Step 1: details , Step 2: details",
            context=[]
            )

  

class RCACrew:
  def __init__(self, df,df_anomaly,counter):
    self.df = df
    self.df_anomaly = df_anomaly
    self.counter = counter

  def run(self):
    deviceagent = DeviceAgent()
    anomaly_agent = deviceagent.Anomaly_agent(self.df)
    devicetask = DeviceTask()
    
    anomaly_task = devicetask.Anomaly_Task(anomaly_agent,self.df,self.df_anomaly)

    mycrew = Crew(
      agents=[
        anomaly_agent
      ],
      tasks=[
        anomaly_task
      ],
      #process=Process.sequential,
      #memory=True,
      #manager_llm=myllm,
      verbose=2
    )
    result1 = mycrew.kickoff()
    #embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    anomaly_value = anomaly_task.output.exported_output
    st.write(f"Data set number: {self.counter}")
    if "No Anomalies Found" not in anomaly_value:
      resolution = new_db.similarity_search(anomaly_value, 1)
      
      rca_agent = deviceagent.RCA_agent(anomaly_value,resolution) 
      rca_task = devicetask.RCA_Task(rca_agent,self.df,anomaly_value,resolution)
      mycrew2 = Crew(
        agents=[
          rca_agent
        ],
        tasks=[
          rca_task
        ],
        #process=Process.sequential,
        memory=True,
        #manager_llm=myllm,
        output_log_file=True,
        verbose=2,
        embedder={"provider": "google",
                  "config":{
                          "model": 'models/embedding-001',
                          "task_type": "retrieval_document",
                          "title": "Embeddings for Embedchain"  
                          }
                  }
        
                  )
      result2 = mycrew2.kickoff()
      # Under As part of the last action...
      # extract the first line from previous result and that becomes the input_prompt
      # Split the text by line breaks
      lines = result2.splitlines()
      # Find the line that starts with "1."
      step1_line = [line for line in lines if "Send" in line]
      #print(step1_line)
      # If step 1 line is found, extract the text after the colon (:)
      if step1_line:
        #print("The action to take")
        #print(step1_line)
        step1_text = step1_line[0].strip()
        first_action = step1_text
        input_prompt = first_action + ":" + "Subject" + ":" + anomaly_value
        last_result = get_gorilla_response(prompt=input_prompt)
               
        #if "<<function>>" in last_result:
        #code_result = last_result.split("<<function>>")[1].strip()
        st.code(last_result, language='python')
        file_path = "gen_code.py"
        with open(file_path, 'w') as file:
          file.write(last_result)
          #run_generated_code(file_path)
        st.success("All necessary Actions related to Device Anomaly completed successfuly for this dataset")
        return
      else:
        st.write("No Anomalies found in this dataset")
        return
        
       
def read_csv_in_chunks(file_path):
    chunk_size = 10
    chunk_counter = 1
    
    try:
        reader = pd.read_csv(file_path, chunksize=chunk_size)
        anomaly_chunk = pd.read_csv("device_data_anomaly.csv")
        for chunk in reader:
            # Assuming RCACrew class and run() method handle the data
            rcacrew = RCACrew(chunk,anomaly_chunk,chunk_counter)
            result = rcacrew.run()
            chunk_counter=chunk_counter+1
            #st.write(result)
            
    except Exception as e:
        #print(f"Error processing CSV: {e}")
        st.error(f"An error has occured: {e}")


def main():
  # Call set_page_config only once at the beginning
  # st.set_page_config("Local LLM RAG", page_icon=":scroll:")
  st.header("Local Edge - Assistant - Upload PDF and csv")

  with st.expander("Uploaded PDFs (for Issure Resolution Guide)"):
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    #csv = st.file_uploader("Upload your CSV File", accept_multiple_files=False)
    st.sidebar.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
        <a href="https://lmstudio.io" target="_blank">Local Edge Assistant</a> 
        </div>
        """,
        unsafe_allow_html=True,
    )

  if pdf_docs:
    with st.spinner("Processing..."):  # User-friendly message
      raw_text = get_pdf_text(pdf_docs)
      text_chunks = get_text_chunks(raw_text)

      # Create FAISS index only if PDFs are uploaded
      if text_chunks:
        create_faiss_index(text_chunks)
      #if csv is not None:
        # Get the temporary file path for uploaded data
        file_path = "device_data.csv"
        read_csv_in_chunks(file_path)  # Pass the temporary file path to the function
      #else:
            #st.warning("No device data shared")                              
  else:
      st.warning("No PDFs.")
if __name__ == "__main__":
  main()
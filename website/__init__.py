import json
from flask import Flask, request, session, redirect, url_for, jsonify, flash, Blueprint
# from flask import redirect
# from flask import url_for
# from flask import Flask, request, session, redirect, url_for, jsonify
from flask import render_template
from flask import Response
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from requests import post, get
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt

import json
import requests
import tensorflow as tf
import cv2 
import numpy as np
import os
import base64
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import wget
import zipfile
# nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
import uuid
import joblib 
import time

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import azure.cognitiveservices.speech as speechsdk

import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
from openai import OpenAI
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error

from fer import FER
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from langchain.chains import ConversationalRetrievalChain

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
# from langchain_openai import ChatOpenAI
# import constants
from langchain.indexes import VectorstoreIndexCreator
import csv

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from flask_socketio import SocketIO
from diffusers import StableDiffusionPipeline
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import torch
#from torch import autocast
from PIL import Image
from datetime import datetime
import io
#starting point of the program execution
# bp = Blueprint('respite', __name__,template_folder='templates')
app = Flask(__name__)
# app.config['APPLICATION_ROOT'] = '/simpshoi'
# app.register_blueprint(bp, url_prefix='/simpshoi')

bcrypt = Bcrypt(app)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'fallback_secret_key')
# api_key = os.getenv('api_key')
# client = OpenAI(api_key=api_key)
dalle_api_key = os.getenv('DALLE_OPENAI_SECRET')
client = OpenAI(api_key = dalle_api_key)
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)
# class ReverseProxied:
#     def __init__(self, app, script_name):
#         self.app = app
#         self.script_name = script_name

#     def __call__(self, environ, start_responses):
#         script_name = self.script_name
#         if script_name:
#             environ['SCRIPT_NAME'] = script_name
#             path_info = environ['PATH_INFO']
#             if path_info.startswith(script_name):
#                 environ['PATH_INFO'] = path_info[len(script_name):]
#                 print(f"SCRIPT_NAME: {environ.get('SCRIPT_NAME')}, PATH_INFO: {environ.get('PATH_INFO')}")
#         return self.app(environ, start_responses)
# app.wsgi_app = ReverseProxied(app.wsgi_app, script_name='/simpshoi')
socketio = SocketIO(app)
import getpass
import os

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvapi_key = getpass.getpass("Enter your NVIDIA API key: ")
    assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key

NVIDIA_API_KEY="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"
nvapi_key= os.getenv('NVIDIA_API_KEY')


db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'auth_plugin': 'mysql_native_password'
}

def db_connect():
    """Create a database connection."""
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL Database: {e}")
        return None

#index
@app.route("/")
def index():
    return render_template('index.html')
@app.route("/simpshoi")
def simpshoi():
    return render_template('index.html')
#aboutus
@app.route("/aboutus")
def aboutus():
    return render_template('aboutus.html')


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
def generate_unique_conversation_id():
    return str(uuid.uuid4())
def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    # Tokenize and keep only alphanumeric words
    text = ' '.join([word for word in word_tokenize(text) if word.isalnum()])
    # Convert to lowercase
    text = ' '.join([word.lower() for word in text.split()])
    # Remove English stopwords
    text = ' '.join([word for word in text.split() if word not in set(stopwords.words('english'))])
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text
nRowsRead = 1000
data = pd.read_csv('datas/intent.csv', delimiter=',', nrows = nRowsRead)
data['Context'] = data['Context'].apply(clean_text)

model = joblib.load('models/linear_svc_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
import random
import re

# def generate_user_id():
#     return str(uuid.uuid4())

# def get_active_chats_for_user(user_id):
#     # In a real application, you would filter chats by user_id.
#     # Here, we return all chats in the session for simplicity.
#     user_id = session['user_id']
#     return [{'id': chat_id, 'start_time': 'N/A'} for chat_id in session.get('user_chats', {})]

def get_active_chats_for_user(user_id):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        sql_query = "SELECT chat_id, conversation_history FROM memory_journal.chatbot WHERE user_id = %s"
        cursor.execute(sql_query, (user_id,))
        result = cursor.fetchall()
        active_chats = [{'chat_id': row[0], 'conversation_history': json.loads(row[1])} for row in result]
        print("these are the active ones:", active_chats)
        cursor.close()
        connection.close()
        return active_chats
    except Exception as e:
        print("Error:", e)
        return []

def load_conversation_history(chat_id):
    return session.get('user_chats', {}).get(chat_id, [])

# user_id = generate_user_id()
# print(user_id)
@app.route('/start_chat', methods=['GET', 'POST'])
def start_chat():
    if 'loggedin' in session and session['loggedin']:
        # user_id = generate_user_id()
        user_id = session['user_id']
        session['user_id'] = user_id
        # if request.method == 'POST':
            # Assuming you have a form with user_id and conversation_history fields
        # user_id = request.form['user_id']
        # conversation_history = request.form['conversation_history']

            # Generate a unique user ID
        if 'user_chats' not in session:
            session['user_chats'] = {}

        new_chat_id = generate_unique_conversation_id()
        session['current_chat_id'] = new_chat_id
        # session['user_chats'][new_chat_id] = []
        conversation_history = [] 
        # Insert chat details into MySQL database
        # Convert conversation history list to a string using JSON serialization
        conversation_history_json = json.dumps(conversation_history)

        # Insert chat details into MySQL database
        insert_chat_details2(new_chat_id, user_id, conversation_history_json)

        return redirect(url_for('load_chat', chat_id=new_chat_id, user_id=user_id))
    else:
        # User is not logged in, prompt to log in
        flash('Please log in to access your Chatbot.', 'info')
        return redirect(url_for('login'))


def insert_chat_details2(chat_id, user_id, conversation_history_json):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        sql_query = "INSERT INTO memory_journal.chatbot (chat_id, user_id, conversation_history) VALUES (%s, %s, %s)"
        values = (chat_id, user_id, conversation_history_json)
        cursor.execute(sql_query, values)

        connection.commit()
        cursor.close()
        connection.close()
    except Exception as e:
        print("Error:", e)

    return

#     return redirect(url_for('load_chat', chat_id=new_chat_id, user_id = user_id))
@app.route('/list_chats/<user_id>', methods=['GET', 'POST'])
def list_chats(user_id):
    if request.method == 'POST':
        chat_id = request.form['chat_id']
        conversation_history = request.form['conversation_history']
        
        # Insert chat details into MySQL database
        insert_chat_details(chat_id, user_id, conversation_history)
        
        # You can redirect or render a response after inserting chat details
        return redirect(url_for('list_chats', user_id=user_id))
    
    # Retrieving active chats for the current user
    active_chats = get_active_chats_for_user(user_id)
    # Store active chats in the session
    session['user_chats'] = {chat['chat_id']: chat for chat in active_chats}
    return render_template('list_chats.html', active_chats=active_chats, user_id=user_id)

def insert_chat_details(chat_id, user_id, conversation_history):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        conversation_history_json = json.dumps(conversation_history)
        print("Conversation history JSON:", conversation_history_json)
        sql_query = "INSERT INTO memory_journal.chatbot (chat_id, user_id, conversation_history) VALUES (%s, %s, %s)"
        values = (chat_id, user_id, conversation_history_json)
        cursor.execute(sql_query, values)

        connection.commit()
        cursor.close()
        connection.close()
    except Exception as e:
        print("Error:", e)

    return

# # Function to load conversation history from MySQL database for a given chat_id
# def load_conversation_history_from_database(chat_id):
#     try:
#         connection = mysql.connector.connect(**db_config)
#         cursor = connection.cursor()

#         # SQL query to retrieve conversation history for the specified chat_id
#         sql_query = "SELECT conversation_history FROM memory_journal.chatbot WHERE chat_id = %s"
        
#         # Execute the query
#         cursor.execute(sql_query, (chat_id,))
        
#         # Fetch the conversation history as a JSON string
#         result = cursor.fetchone()
#         if result:
#             conversation_history_json = result[0]
#             print("Conversation history JSON from database:", conversation_history_json)  # Debugging statement

#             # Parse JSON string to Python object
#             conversation_history = json.loads(conversation_history_json)
#             return conversation_history

#     except mysql.connector.Error as error:
#         print(f"Error loading conversation history from MySQL database: {error}")
#         return None

#     finally:
#         # Close connection
#         if connection.is_connected():
#             cursor.close()
#             connection.close()
#             print("MySQL connection is closed")

def load_conversation_history_from_database(chat_id):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # SQL query to retrieve conversation history for the specified chat_id
        sql_query = "SELECT conversation_history FROM memory_journal.chatbot WHERE chat_id = %s"
        
        # Execute the query
        cursor.execute(sql_query, (chat_id,))
        
        # Fetch the conversation history as a JSON string
        result = cursor.fetchone()
        if result:
            conversation_history_json = result[0]
            print("Conversation history JSON from database:", conversation_history_json)  # Debugging statement

            # Parse JSON string to Python object
            conversation_history = json.loads(conversation_history_json)
            return conversation_history
        else:
            return None

    except mysql.connector.Error as error:
        print(f"Error loading conversation history from MySQL database: {error}")
        return None

    finally:
        # Close connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

@app.route('/load_chat/<chat_id>', methods=['GET', 'POST'])
def load_chat(chat_id):
    if request.method == 'POST':
        f = request.files.get('file')
    
            # Extracting uploaded file name
        data_filename = secure_filename(f.filename)

        f.save(os.path.join(app.config['UPLOAD_FOLDER'],
                            data_filename))

        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
        print("Success")
        # return render_template('chat.html', conversation_history=conversation_history, chat_id=chat_id, user_id=user_id)
    # Load conversation history for the specified chat_id from the MySQL database
    user_id = session['user_id']
    if not user_id:
        return redirect(url_for('login'))  # Redirect to login if user_id is not set in session
    conversation_history = load_conversation_history_from_database(chat_id) or []
    print("Conversation history loaded:", conversation_history) 
    return render_template('chat.html', conversation_history=conversation_history, chat_id=chat_id, user_id=user_id)

def delete_chat_from_database(chat_id):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # SQL query to delete the chat with the specified chat_id
        sql_query = "DELETE FROM memory_journal.chatbot WHERE chat_id = %s"
        
        # Execute the query
        cursor.execute(sql_query, (chat_id,))
        
        # Commit the transaction
        connection.commit()
        print("Chat deleted successfully")

    except mysql.connector.Error as error:
        print(f"Error deleting chat from MySQL database: {error}")

    finally:
        # Close connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
@app.route('/delete_chat/<chat_id>')
def delete_chat(chat_id):
    # Check if the chat_id exists in the user_chats session data
    user_chats = session.get('user_chats', {})
    user_id = session['user_id']
    if chat_id in user_chats:
        # Remove the chat_id from the user_chats session data
        del user_chats[chat_id]
        session['user_chats'] = user_chats
        
        # Delete the chat from the MySQL database
        delete_chat_from_database(chat_id)
        
        return redirect(url_for('list_chats', user_id=user_id))
    else:
        # Handle the case where the chat_id does not exist
        return "Chat not found", 404
    

def update_chat_details(chat_id, conversation_history_json):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # SQL query to update conversation history
        sql_query = "UPDATE memory_journal.chatbot SET conversation_history = %s WHERE chat_id = %s"
        values = (conversation_history_json, chat_id)
        cursor.execute(sql_query, values)

        connection.commit()
        cursor.close()
        connection.close()
        print("Chat details updated successfully")

    except Exception as e:
        print("Error:", e)

def retrieve_conversation_history(chat_id):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)
        print(chat_id)
        
        # SQL query to retrieve conversation history for the specified user_id and chat_id
        sql_query = "SELECT conversation_history FROM memory_journal.chatbot WHERE chat_id = %s"
        
        # Execute the query with parameters
        cursor.execute(sql_query, (chat_id,))
        
        # Fetch the conversation history JSON string 
        result = cursor.fetchone()
        print("Result:", result)
        # Check if conversation history is retrieved
        if result is not None:
            # Convert JSON string to Python dictionary/list
            conversation_history_json = result['conversation_history']
            conversation_history = json.loads(conversation_history_json)
            return conversation_history
        else:
            print("Conversation history not found for the specified chat_id.")
            return []

    except mysql.connector.Error as error:
        print(f"Error retrieving conversation history from MySQL database: {error}")
        return []

    finally:
        # Close connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    # Tokenize and keep only alphanumeric words
    text = ' '.join([word for word in word_tokenize(text) if word.isalnum()])
    # Convert to lowercase
    text = ' '.join([word.lower() for word in text.split()])
    # Remove English stopwords
    text = ' '.join([word for word in text.split() if word not in set(stopwords.words('english'))])
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text
nRowsRead = 1000

from typing import List, Union
import os
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
# UPLOAD_FOLDER = 'pdf'
# ALLOWED_EXTENSIONS = {'pdf'}
from langchain.text_splitter import CharacterTextSplitter

class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content  # Assuming 'content' maps to the 'page_content' attribute
        self.metadata = metadata if metadata is not None else {}

r_splitter = CharacterTextSplitter(
    separator = "\\n\\n",  # Split character (default \\n\\n)
    chunk_size=240,
    chunk_overlap=16,
)
# from langchain_community.embeddings.openai import OpenAIEmbeddings
from llama_index.embeddings.nvidia import NVIDIAEmbedding
# embeddings_model = OpenAIEmbeddings()
embeddings_model = NVIDIAEmbeddings(model="NV-Embed-QA", NVIDIA_API_KEY= nvapi_key, truncate="END")
client = OpenAI(
  base_url="https://ai.api.nvidia.com/v1/retrieval/nvidia"
)
# def index_docs(splitter, documents: List[Document], dest_embed_dir: str) -> None:
#     embeddings_model = NVIDIAEmbeddings(model="NV-Embed-QA", NVIDIA_API_KEY= nvapi_key)
#     # embeddings_model = NVIDIAEmbeddings(model="ai-embed-qa-4", model_type="query")
#     # embeddings_model = OpenAIEmbeddings()
#     texts = []
#     metadatas = []
#     for document in documents:
#         doc_texts = splitter.split_text(document.page_content)
#         texts.extend(doc_texts)
#         metadatas.extend([document.metadata] * len(doc_texts))
    
#     print("Documents processed:")
#     print(documents)
#     print(f"Number of texts: {len(texts)}")
#     print(f"Sample text: {texts[:1]}")
#     print(f"Sample metadata: {metadatas[:1]}")

#     if texts:
#         # embedding_dim = embeddings_model.get_embedding_dimension()
#         # print(f"Embedding dimension: {embedding_dim}")
#         if os.path.exists(dest_embed_dir):
#             update = FAISS.load_local(folder_path=dest_embed_dir, embeddings=embeddings_model)
#             update.add_texts(texts, metadatas=metadatas)
#             update.save_local(folder_path=dest_embed_dir)
#             # print("@shape:",update.shape)
#         else:
#             docsearch = FAISS.from_texts(texts, embeddings_model, metadatas)
#             docsearch.save_local(folder_path=dest_embed_dir)
#             # print("shaping:",docsearch.shape)
#     else:
#         print("No texts to embed, skipping FAISS indexing.")

# def load_and_transform_data_from_csv(file_path: str, embedding_path: str = "./embedding") -> None:
#     embedding_path = "./embedding"
#     print(f"Storing embeddings to {embedding_path}")
#     documents = []
#     with open(file_path, mode='r', encoding='utf-8') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             metadata = {'source': 'CSV Import'}
#             documents.append(Document(row['answer'], metadata))
#     # Extract text from Document objects
#     # Extract text from Document objects
#     # texts = [doc.page_content for doc in documents]
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0, length_function=len)
#     # documentss = text_splitter.create_documents(texts)
#     index_docs(splitter=text_splitter, documents=documents, dest_embed_dir=embedding_path)
#     print("Generated embeddings successfully")

# def split_documents(documents, chunk_size=240, chunk_overlap=16):
#     r_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
#     all_splits = []
#     for document in documents:
#         splits = r_splitter.split_text(document.page_content)
#         for split in splits:
#             all_splits.append(Document(split, document.metadata))

#     return all_splits

# Path to your CSV file
# file_path = 'data/output_file.csv'
file_path = 'data/'

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.nvidia import NVIDIA

# Here we are using mixtral-8x7b-instruct-v0.1 model from API Catalog
Settings.llm = NVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
Settings.text_splitter = SentenceSplitter(chunk_size=400)
documents = SimpleDirectoryReader(file_path).load_data()

from llama_index.core import VectorStoreIndex
# When you use from_documents, your Documents are split into chunks and parsed into Node objects
# By default, VectorStoreIndex stores everything in memory
index = VectorStoreIndex.from_documents(documents)
# # Load the data
# data = load_and_transform_data_from_csv(file_path)
# embedding_path = "embedding/"
# docsearch = FAISS.load_local(folder_path=embedding_path, embeddings=embeddings_model)
# retriever = docsearch.as_retriever()

# from langchain.tools import BaseTool

# class DataRetriever(BaseTool):
#     name = "about dementia and mental health"
#     description = "Useful for when you need to answer questions about dementia patient's mental health needs, how to help them and so on."

#     def _run(self, query):
#         out = retriever.invoke(query)
#         o = out[0]
#         item=o.page_content.split('|')
#         output = '\n'.join(item)
#         return output

#     def _arun(self, query: str):
#         raise NotImplementedError("This tool does not support async")
# sv=DataRetriever()

# from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper

# wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# from langchain.tools import Tool

# ## Make sure you give it a proper name and a good description on how to use the tools
# wiki_tool = Tool.from_function(
#     func=wikipedia.run,
#     name="Wiki",
#     description="useful for when you need to search certain topic on Wikipedia, aka wiki")
# retriever_tool=Tool.from_function(
#     func=sv.invoke,
#     name="Aboutdementiapatients",
#     description="useful for when you need to find information about dementia patients mental health needs")

# tools = [wiki_tool, retriever_tool]

# from langchain_core.agents import AgentFinish
# from langgraph.prebuilt.tool_executor import ToolExecutor
# import os
# from langchain.agents import AgentExecutor
# from langchain.agents import initialize_agent
# from langchain.prompts import MessagesPlaceholder
# from langchain.memory import ConversationBufferMemory
# from langchain.agents import AgentType, Agent, ConversationalAgent
# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# # This a helper class we have that is useful for running tools
# # It takes in an agent action and calls that tool and returns the result
# tool_executor = ToolExecutor(tools)





# llm = ChatNVIDIA(model="llama2_70b")

# memory = ConversationBufferMemory(memory_key="chat_history", input_key='input', output_key="output", return_messages=True)

# # question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)


# prompt_template = """
# ### [INST]

# Assistant is a large language model trained by Mistral.

# Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

# Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

# Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

# Context:
# ------

# Assistant has access to the following tools:

# {tools}

# To use a tool, please use the following format:

# '''
# Thought: Do I need to use a tool? Yes
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# '''

# When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

# '''
# Thought: Do I need to use a tool? No
# Final Answer: [your response here]
# '''

# Begin!

# Previous conversation history:
# {chat_history}

# New input: {input}

# Current Scratchpad:
# {agent_scratchpad}

# [/INST]
#  """

# # Create prompt from prompt template
# prompt = PromptTemplate(
#     input_variables=['agent_scratchpad', 'chat_history', 'input', 'tool_names', 'tools'],
#     template=prompt_template,
# )

# prompt = prompt.partial(
#     tools=[t.name for t in tools],
#     tool_names=", ".join([t.name for t in tools]),
# )
# print("prompt ---> \n", prompt)
# from typing import Any, Optional, Sequence

# from langchain_core._api import deprecated
# from langchain_core.callbacks import BaseCallbackManager
# from langchain_core.language_models import BaseLanguageModel
# from langchain_core.tools import BaseTool

# from langchain.agents.agent import AgentExecutor
# from langchain.agents.agent_types import AgentType
# from langchain.agents.loading import AGENT_TO_CLASS, load_agent

# agent_cls = AGENT_TO_CLASS[AgentType.CONVERSATIONAL_REACT_DESCRIPTION]
# agent_kwargs = {}
# agent_obj = agent_cls.from_llm_and_tools(
#     llm, tools, callback_manager=None, **agent_kwargs)

# agent_execute=AgentExecutor.from_agent_and_tools(
#         agent=agent_obj,
#         tools=tools,
#         callback_manager=None,
#         handle_parsing_errors=True,
#         verbose=True,
#         output_key = "output",
#         max_iterations=3,
#         return_intermediate_steps=True,
#         early_stopping_method="generate", # or use **force**
#         memory=ConversationBufferMemory(memory_key="chat_history", input_key='input', output_key="output")
# )
# from typing import TypedDict, Annotated, List, Union
# from langchain_core.agents import AgentAction, AgentFinish
# from langchain_core.messages import BaseMessage
# import operator


# class AgentState(TypedDict):
#     # The input string
#     input: str
#     # The list of previous messages in the conversation
#     chat_history: list[BaseMessage]
#     # The outcome of a given call to the agent
#     # Needs `None` as a valid type, since this is what this will start as
#     agent_outcome: Union[AgentAction, AgentFinish, None]
#     # List of actions and corresponding observations
#     # Here we annotate this with `operator.add` to indicate that operations to
#     # this state should be ADDED to the existing values (not overwrite it)
#     intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

# # Define the agent
# from langchain_core.agents import AgentActionMessageLog

# def run_agent(data):
#     inputs = data.copy()
#     print(inputs)
#     text = inputs['input']
#     agent_outcome = agent_execute.invoke({"input":text})
#     return {"agent_outcome": agent_outcome}

# # Define the function to execute tools
# def execute_tools(data):
#     # Get the most recent agent_outcome - this is the key added in the `agent` above
#     agent_output = data["agent_outcome"]
#     if len(agent_output['intermediate_steps'])>=1 :
#         agent_action = agent_output['intermediate_steps'][0][0]
#         output = tool_executor.invoke(agent_action)
#         return {"intermediate_steps": [(agent_action, str(output))]}
#     else:
#         return {"intermediate_steps":[]}

# # Define logic that is used to determine which conditional edge to go down
# def should_continue(data):
#     # If the agent outcome is an AgentFinish, then we return `exit` string
#     # This will be used when setting up the graph to define the flow
#     if data["agent_outcome"]["output"] is not None:
#         print(" **AgentFinish** " )
#         return "end"
#     # Otherwise, an AgentAction is returned
#     # Here we return `continue` string
#     # This will be used when setting up the graph to define the flow
#     else:
#         print(" **continue** " )
#         return "continue"

# from langgraph.graph import END, StateGraph

# # Define a new graph
# workflow = StateGraph(AgentState)

# # Define the two nodes we will cycle between
# workflow.add_node("agent", run_agent)
# workflow.add_node("action", execute_tools)

# # Set the entrypoint as `agent`
# # This means that this node is the first one called
# workflow.set_entry_point("agent")

# # We now add a conditional edge
# workflow.add_conditional_edges(
#     # First, we define the start node. We use `agent`.
#     # This means these are the edges taken after the `agent` node is called.
#     "agent",
#     # Next, we pass in the function that will determine which node is called next.
#     should_continue,
#     # Finally we pass in a mapping.
#     # The keys are strings, and the values are other nodes.
#     # END is a special node marking that the graph should finish.
#     # What will happen is we will call `should_continue`, and then the output of that
#     # will be matched against the keys in this mapping.
#     # Based on which one it matches, that node will then be called.
#     {
#         # If `tools`, then we call the tool node.
#         "continue": "action",
#         # Otherwise we finish.
#         "end": END,
#     },
# )

# # We now add a normal edge from `tools` to `agent`.
# # This means that after `tools` is called, `agent` node is called next.
# workflow.add_edge("action", "agent")

# # Finally, we compile it!
# # This compiles it into a LangChain Runnable,
# # meaning you can use it as you would any other runnable
# appss = workflow.compile()



# chat = ChatNVIDIA(model="mixtral_8x7b", temperature=0.1, max_tokens=1000, top_p=1.0)

# doc_chain = load_qa_chain(chat , chain_type="stuff", prompt=QA_PROMPT)


# chain = ConversationalRetrievalChain(
#     retriever=docsearch.as_retriever(),
#     combine_docs_chain=doc_chain,
#     memory=memory,
#     question_generator=question_generator,
# )
# return chain
from langchain_core.language_models.base import LanguageModelInput
query_engine = index.as_query_engine(similarity_top_k=10)
@app.route('/predict2/<chat_id>', methods=['GET', 'POST'])
def predict2(chat_id):
    user_id = session['user_id']
    casual_inquiries = ['hello', 'hi', 'hey', 'what are you doing', 'how are you']

    def should_fallback(generated_response, input_text):
    # Ensure generated_response is a string (or get the text from the response object)
        generated_text = generated_response.get('text', '') if isinstance(generated_response, dict) else str(generated_response)

        is_casual = any(inquiry in input_text.lower() for inquiry in casual_inquiries)
        
        # Check if generated_response is uncertain
        is_uncertain = "I'm not sure" in generated_text
        is_too_short = len(generated_text.split()) < 3  # Example threshold

        # Implement the fallback logic
        if is_casual and (is_uncertain or is_too_short):
            return True
        elif is_casual:
            return is_uncertain or is_too_short
        elif is_uncertain or is_too_short:
            return True
        else:
            return False

    # Load or initialize conversation history
    user_chats = session.get('user_chats', {})
    # conversation_history = user_chats.get(chat_id, [])
    print(user_chats)
    print(chat_id)
    conversation_history = retrieve_conversation_history(chat_id) or [] 
    print(f"Conversation history: ", conversation_history)
    if request.method == 'POST':
        # Get user input from the form
        # input_text = request.form.get('input_text').strip()
        input_text = request.form.get('input_text')
        # input_text = input_text.strip()
        input_text = input_text.replace(r'\S*@\S*\s?', '')
        input_text = input_text.replace(r'[^\w\s]', '')
        input_text = input_text.replace(r'â€™', '')
        input_text = input_text.replace(r'Â', '')
        input_text = input_text.replace(r'\S*@\S*\s?', '')
        input_text = input_text.replace(r'[^\w\s]', '')
        input_text = input_text.replace(r'â€™', '')
        input_text = input_text.replace(r'Â', '')
        input_text = input_text.replace(r'â', '')
        prompt = "You are a helpful assistant for patients that are having dementia."
        for chat in conversation_history:
            role = "user" if chat['sender'] == 'User' else "assistant"
            prompt += f"\n{chat['message']}"


         # Initialize the messages list with a system message (if needed)
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        
        # Append conversation history to messages
        for chat in conversation_history:
            role = "user" if chat['sender'] == 'User' else "assistant"
            messages.append({"role": role, "content": chat['message']})
            messages.append({"role": "user", "content": input_text})
        # Add the new user message
        prompt += f"\n{input_text}"
        # Load or initialize conversation history
        # Process the input text with your conversational model
        # simple_chat_history = [f"{chat['sender']}: {chat['message']}" for chat in conversation_history]
        # Assuming you need to convert to a list of tuples format

        

        jgenerated_response = query_engine.query(input_text)
        print(type(jgenerated_response))  # Check the type of the response
        print(jgenerated_response)
        generated_response = str(jgenerated_response)
        print(generated_response)

        # chat_history_for_chain = [(chat['sender'], chat['message']) for chat in conversation_history]

        # # Then, pass this to the chain function
        # result = ({'input': input_text, 'chat_history': chat_history_for_chain})
        # result = appss.invoke({'input': input_text, 'chat_history': chat_history_for_chain})
        # # result = appss.invoke({'input': input_text, 'chat_history': conversation_history})
        # print(result)
        # generated_response = result.get('agent_outcome', {}).get('output', None)





        # Append the new input and response to the conversation history
        # conversation_history.append({"user": input_text, "bot": result['answer']})
        # session[f'chat_history_{chat_id}'] = conversation_history

        # Extract the generated response
        # generated_response = response.choices[0].text.strip()



        if should_fallback(generated_response, input_text):
            # Check if it is a casual inquiry
            if any(inquiry in input_text.lower() for inquiry in casual_inquiries):
                generated_response = "Hello! I'm here to assist you. What can I do for you today?"
            # Generic fallback for other cases
            else:
                generated_response = "I'm not quite sure I have the answer to that. Can you tell me more or try asking something else?"
            
        
        # if not generated_response.endswith('.'):
        #     generated_response += '.'

        if should_fallback(generated_response, input_text):  # Implement this function based on your criteria
            generated_response = "I'm not sure how to answer that. Can you ask something else?"




        # Update conversation history
        conversation_history.append({'sender': 'User', 'message': input_text, 'class': 'user-message'})
        conversation_history.append({'sender': 'ChatGPT2.0', 'message': generated_response, 'class': 'bot-message'})

        # Update the session
        user_chats[chat_id] = conversation_history
        session['user_chats'] = user_chats
        # Convert updated conversation history list to a string using JSON serialization
        updated_conversation_history_json = json.dumps(user_chats[chat_id])

        # Update chat details in MySQL database
        update_chat_details(chat_id, updated_conversation_history_json)

        return render_template('chatthird.html', conversation_history=conversation_history, chat_id=chat_id, user_id=user_id)
    
    elif request.method == 'GET':
        # Render the template with the conversation history for GET requests
        return render_template('chatthird.html', conversation_history=conversation_history, chat_id=chat_id, user_id=user_id)

    else:
        # Handle other HTTP methods if needed
        return 'Method Not Allowed', 405


import logging

logging.basicConfig(level=logging.DEBUG)

def _post(self, invoke_url, payload):
    try:
        logging.debug(f"Payload: {payload}")
        response, session = self._post(invoke_url, payload)
        self._try_raise(response)
        return response, session
    except Exception as e:
        logging.error(f"Error during request: {e}")
        raise



def db_connect():
    """Create a database connection."""
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL Database: {e}")
        return None
    
    import secrets

app.secret_key = os.urandom(24)




@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
       

        username = request.form['username']

        password_candidate = request.form['password']
        print(f"Username: {username}, Password: {password_candidate}")
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()
        if user and bcrypt.check_password_hash(user[2], password_candidate):  # Assuming password is in the 3rd column
            session['loggedin'] = True
            session['username'] = username
            session['user_id'] = user[0]
            print(session['user_id'])
            return redirect(url_for('index'))  # Redirect to the success page
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (username, password) VALUES (%s, %s)', (username, password))
        conn.commit()
        cursor.close()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/success')
def success():
    if 'loggedin' in session:
        return render_template('success.html', username=session['username'])
    return redirect(url_for('login'))  # Redirect to login if not logged in




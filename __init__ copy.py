import json
from flask import Flask, request, session, redirect, url_for, jsonify, flash
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
# nltk.download('stopwords')
# nltk.download('wordnet')
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
from langchain_openai import ChatOpenAI
# import constants
from langchain.indexes import VectorstoreIndexCreator
import csv

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from flask_socketio import SocketIO
from diffusers import StableDiffusionPipeline

import torch
#from torch import autocast
from PIL import Image
from datetime import datetime
import io
#starting point of the program execution


app = Flask(__name__)
bcrypt = Bcrypt(app)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'fallback_secret_key')
# api_key = os.getenv('api_key')
# client = OpenAI(api_key=api_key)
dalle_api_key = os.getenv('DALLE_OPENAI_SECRET')
client = OpenAI(api_key = dalle_api_key)
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)
class ReverseProxied:
    def __init__(self, app, script_name):
        self.app = app
        self.script_name = script_name

    def __call__(self, environ, start_response):
        script_name = self.script_name
        if script_name:
            environ['SCRIPT_NAME'] = script_name
            path_info = environ['PATH_INFO']
            if path_info.startswith(script_name):
                environ['PATH_INFO'] = path_info[len(script_name):]
        return self.app(environ, start_response)
app.wsgi_app = ReverseProxied(app.wsgi_app, script_name='/simpshoi')
socketio = SocketIO(app)
import getpass
import os

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvapi_key = getpass.getpass("Enter your NVIDIA API key: ")
    assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key
## Core LC Chat Interface
from langchain_nvidia_ai_endpoints import ChatNVIDIA




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


#OPENAI MODEL
@app.route('/predict/<chat_id>', methods=['GET', 'POST'])
def predict(chat_id):
    casual_inquiries = ['hello', 'hi', 'hey', 'what are you doing', 'how are you']
    user_id = session['user_id']
    def should_fallback(generated_response, input_text):
        is_casual = any(inquiry in input_text.lower() for inquiry in casual_inquiries)
        # Criteria for when the bot's response is uncertain or not helpful
        is_uncertain = "I'm not sure" in generated_response
        is_too_short = len(generated_response.split()) < 3  # Example threshold

        # Implement the fallback logic
        if is_casual and (is_uncertain or is_too_short):
            # Allow the bot to attempt small talk, but fallback if uncertain or response is too short
            return True
        elif is_casual:
            # No fallback needed for casual conversation, unless the response is uncertain or too short
            return is_uncertain or is_too_short
        elif is_uncertain or is_too_short:
            # Generic fallback for other uncertain or short responses
            return True
        else:
            # No fallback needed if none of the above conditions are met
            return False
    # Load or initialize conversation history
    user_chats = session.get('user_chats', {})
    # conversation_history = user_chats.get(chat_id, [])
    print(user_chats)
    print(chat_id)
    conversation_history = retrieve_conversation_history(chat_id) or [] 
    if request.method == 'POST':
        # Get user input from the form
        input_text = request.form.get('input_text').strip()
        input_text = input_text.replace(r'\S*@\S*\s?', '')
        input_text = input_text.replace(r'[^\w\s]', '')
        input_text = input_text.replace(r'â€™', '')
        input_text = input_text.replace(r'Â', '')
        input_text = input_text.replace(r'\S*@\S*\s?', '')
        input_text = input_text.replace(r'[^\w\s]', '')
        input_text = input_text.replace(r'â€™', '')
        input_text = input_text.replace(r'Â', '')
        input_text = input_text.replace(r'â', '')
        # Prepare the prompt with conversation history for OpenAI API
        # messages = [{"role": "system", "content": "You are a helpful assistant."}]
        # for chat in conversation_history:
        #     role = "user" if chat['sender'] == 'User' else "assistant"
        #     messages.append({"role": role, "content": chat['message']})

        # Add the new user message
        # messages.append({"role": "user", "content": input_text})
        # Prepare the prompt with conversation history for OpenAI API
        prompt = "You are a helpful assistant."
        for chat in conversation_history:
            role = "user" if chat['sender'] == 'User' else "assistant"
            prompt += f"\n{chat['message']}"


         # Initialize the messages list with a system message (if needed)
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        
        # Append conversation history to messages
        for chat in conversation_history:
            role = "user" if chat['sender'] == 'User' else "assistant"
            messages.append({"role": role, "content": chat['message']})

    # Add the new user message to the messages list
        messages.append({"role": "user", "content": input_text})
        # Add the new user message
        prompt += f"\n{input_text}"
        # Call OpenAI API
        # response = client.completions.create(
        #     model="ft:babbage-002:nanyang-polytechnic::8pfKqowc",  # Use your fine-tuned model name here if you have one
        #     prompt = input_text,
        #     max_tokens=80,
        #     temperature= 0.05,
        #     stop=["\n"]
        # )
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-1106:nanyang-polytechnic::8pyJv2vI",  # Use your fine-tuned model name here if you have one
            # model="ft:gpt-3.5-turbo-1106:nanyang-polytechnic::8pjj8tzP", 
            messages = messages,
            max_tokens=80,
            temperature= 0.9,
            top_p=0.9,
            stop=["\n"]
         )

        # Extract the generated response
        # generated_response = response.choices[0].text.strip()
        generated_response = response.choices[0].message.content
        if should_fallback(generated_response, input_text):
            # Check if it is a casual inquiry
            if any(inquiry in input_text.lower() for inquiry in casual_inquiries):
                generated_response = "Hello! I'm here to assist you. What can I do for you today?"
            # Generic fallback for other cases
            else:
                generated_response = "I'm not quite sure I have the answer to that. Can you tell me more or try asking something else?"
            
        
        if not generated_response.endswith('.'):
            generated_response += '.'

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

        return render_template('chat.html', conversation_history=conversation_history, chat_id=chat_id, user_id=user_id)
    
    elif request.method == 'GET':
        # Render the template with the conversation history for GET requests
        return render_template('chat.html', conversation_history=conversation_history, chat_id=chat_id, user_id=user_id)

    else:
        # Handle other HTTP methods if needed
        return 'Method Not Allowed', 405
#starting point of the program execution


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

m = load_model('models/my_model.h5')

# Load Tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizers = pickle.load(handle)

# Load Label Encoder
with open('label_encoder.pickle', 'rb') as handle:
    le2 = pickle.load(handle)
# Load processed data using pickle
# with open('processed_data.pickle', 'rb') as handle:
#     responses = pickle.load(handle)
with open('responses.pickle', 'rb') as handle:
    responses = pickle.load(handle)

# #specifying the URL of the GloVe pre-trained word vectors
# url = 'https://nlp.stanford.edu/data/glove.6B.zip'
# #specifying the filename for the downloaded file
# filename = 'glove.6B.zip'

# # Download the file
# wget.download(url, filename)

# # Extract the contents of the ZIP file
# with zipfile.ZipFile(filename, 'r') as zip_ref:
#     zip_ref.extractall()
vocabulary = len(tokenizers.word_index)
print("number of unique words : ",vocabulary)
output_length = le2.classes_.shape[0]
print("output length: ",output_length)
glove_dir = "glove.6B.100d.txt"
embeddings_index = {}
# Open the file with 'utf-8' encoding
with open(glove_dir, encoding='utf-8') as file_:
    for line in file_:
      # Splits each line into a list of values.
        arr = line.split()
        single_word = arr[0]
        w = np.asarray(arr[1:], dtype='float32')
        embeddings_index[single_word] = w
file_.close()
print('Found %s word vectors.' % len(embeddings_index))
max_words = vocabulary + 1
word_index = tokenizers.word_index
embedding_matrix = np.zeros((max_words,100)).astype(object)
for word , i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# # Custom function to get embedding from the matrix
# def get_glove_embedding(word):
#     index = word_index.get(word)
#     if index is not None and index < max_words:
#         return embedding_matrix[index]
#     else:
#         return np.zeros((100,))

@app.route('/predict1/<chat_id>', methods=['GET', 'POST'])
def predict1(chat_id):
    user_id = session['user_id']
    # user_chats = session.get('user_chats', {})
    conversation_history = retrieve_conversation_history(chat_id) or []
    prediction = None
    casual_inquiries = ['hello', 'hi', 'hey', 'what are you doing', 'how are you']

    def should_fallback(generated_response, input_text):
        is_casual = any(inquiry in input_text.lower() for inquiry in casual_inquiries)
        # Criteria for when the bot's response is uncertain or not helpful
        is_uncertain = "I'm not sure" in generated_response
        is_too_short = len(generated_response.split()) < 3  # Example threshold

        # Implement the fallback logic
        if is_casual and (is_uncertain or is_too_short):
            # Allow the bot to attempt small talk, but fallback if uncertain or response is too short
            return True
        elif is_casual:
            # No fallback needed for casual conversation, unless the response is uncertain or too short
            return is_uncertain or is_too_short
        elif is_uncertain or is_too_short:
            # Generic fallback for other uncertain or short responses
            return True
        else:
            # No fallback needed if none of the above conditions are met
            return False
    if request.method == 'POST':
        
    # Assuming your model takes text input
        input_text = request.form.get('input_text')
        input_text = input_text.strip()
        input_text = input_text.replace(r'\S*@\S*\s?', '')
        input_text = input_text.replace(r'[^\w\s]', '')
        input_text = input_text.replace(r'â€™', '')
        input_text = input_text.replace(r'Â', '')
        input_text = input_text.replace(r'\S*@\S*\s?', '')
        input_text = input_text.replace(r'[^\w\s]', '')
        input_text = input_text.replace(r'â€™', '')
        input_text = input_text.replace(r'Â', '')
        input_text = input_text.replace(r'â', '')
        clean_input = clean_text(input_text)
        clean_input = tokenizers.texts_to_sequences([input_text])
        # clean_input = np.array(clean_input).reshape(-1)
        clean_input = pad_sequences(clean_input, maxlen=188, truncating='pre')
        # new_data_vec = vectorizer.transform([clean_input])
    # Make predictions
        prediction = m.predict(clean_input) # Extract the first prediction
        response_index = np.argmax(prediction, axis=1)
        print(prediction)
        print(f"where is the class",le2.classes_)
        response_tag = le2.inverse_transform(response_index)[0]
        

        candidate_responses = responses.get(response_tag, "")
        print(f"prediction = ",prediction)
        # Check if the prediction is in the 'Context' values
        
        # valid_responses = data[data['Context'] == prediction]['Response'].values
        print(conversation_history)
        # Randomly choose a response from the valid ones
        response = random.choice([candidate_responses]) if candidate_responses else "No response available"
        if should_fallback(response, input_text):
            # Check if it is a casual inquiry
            if any(inquiry in input_text.lower() for inquiry in casual_inquiries):
                response = "Hello! I'm here to assist you. What can I do for you today?"
            # Generic fallback for other cases
            else:
                response = "I'm not quite sure I have the answer to that. Can you tell me more or try asking something else?"
        print(response)
        conversation_history.append({'sender': 'User', 'message': input_text, 'class': 'user-message'})
        conversation_history.append({'sender': 'Model2.0', 'message': response, 'class': 'bot-message'})

        # Update the session with the updated conversation history
        user_chats = session.get('user_chats', {})
        user_chats[chat_id] = conversation_history
        session['user_chats'] = user_chats
        updated_conversation_history_json = json.dumps(user_chats[chat_id])
        update_chat_details(chat_id, updated_conversation_history_json)
    # Render the template with the conversation history
        return render_template('chatsecond.html', prediction_text=prediction, conversation_history=conversation_history, chat_id=chat_id, user_id=user_id)
    
    elif request.method == 'GET':
        # Handle GET requests if needed
        # conversation_history = load_conversation_history(chat_id)
        return render_template('chatsecond.html', prediction_text=prediction, conversation_history=conversation_history, chat_id= chat_id, user_id=user_id)
    
    else:
        # Handle other HTTP methods if needed
        return 'Method Not Allowed', 405


# UPLOAD_FOLDER = 'pdf'
# ALLOWED_EXTENSIONS = {'pdf'}

from langchain.llms import OpenAI
PERSIST = False
if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=client.embeddings)
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    # loader = TextLoader("data/data.txt")
    # loader = DirectoryLoader("data/", glob=".pdf")
    
    class Document:
        def __init__(self, content, metadata=None):
            self.page_content = content  # Assuming 'content' maps to the 'page_content' attribute
            self.metadata = metadata if metadata is not None else {}
    def load_and_transform_data_from_csv(file_path):
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # For each row in the CSV, create a Document object with 'page_content' and optionally 'metadata'
            transformed_data = [Document(row['answer'], {'source': 'CSV Import'}) for row in reader]
        return transformed_data

    # Path to your CSV file
    file_path = 'data/output_file.csv'

# Load the data
    data = load_and_transform_data_from_csv(file_path)
    # loaders = CSVLoader(file_path='data/output_file.csv')
    # loader = loaders.load()
    print("hello")
    if PERSIST:
        index_creator = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"} if PERSIST else {})
    else:
        index_creator = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"} if PERSIST else {})
        index = index_creator.from_documents(data)

chain = ConversationalRetrievalChain.from_llm(
    llm = ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 20}),
    # max_tokens=500
)
# return chain

# chain = initialize_model()
# llm = ChatNVIDIA(model="ai-llama3-70b")
# result = llm.invoke("Write a ballad about LangChain.")
# print(result.content)

@app.route('/predict2/<chat_id>', methods=['GET', 'POST'])
def predict2(chat_id):
    user_id = session['user_id']
    casual_inquiries = ['hello', 'hi', 'hey', 'what are you doing', 'how are you']

    def should_fallback(generated_response, input_text):
        is_casual = any(inquiry in input_text.lower() for inquiry in casual_inquiries)
        # Criteria for when the bot's response is uncertain or not helpful
        is_uncertain = "I'm not sure" in generated_response
        is_too_short = len(generated_response.split()) < 3  # Example threshold

        # Implement the fallback logic
        if is_casual and (is_uncertain or is_too_short):
            # Allow the bot to attempt small talk, but fallback if uncertain or response is too short
            return True
        elif is_casual:
            # No fallback needed for casual conversation, unless the response is uncertain or too short
            return is_uncertain or is_too_short
        elif is_uncertain or is_too_short:
            # Generic fallback for other uncertain or short responses
            return True
        else:
            # No fallback needed if none of the above conditions are met
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
        chat_history_for_chain = [(chat['sender'], chat['message']) for chat in conversation_history]

        # Then, pass this to the chain function
        result = ({'question': input_text, 'chat_history': chat_history_for_chain})
        result = chain({'question': input_text, 'chat_history': chat_history_for_chain})
        result = chain({'question': input_text, 'chat_history': conversation_history})
        
        generated_response = result['answer']
        # Append the new input and response to the conversation history
        conversation_history.append({"user": input_text, "bot": result['answer']})
        session[f'chat_history_{chat_id}'] = conversation_history

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




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
#==================================================
#  import env file
#==================================================
from dotenv import load_dotenv
import os
load_dotenv()
api_key=os.getenv("GROQ_API_KEY")

#==================================================
#  import libraries
#==================================================
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import pandas as pd

#pip install langchain-groq 
#install this command library to use groq api in langchain
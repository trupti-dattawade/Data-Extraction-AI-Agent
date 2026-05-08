#==================================================
#  import env file
#==================================================
from dotenv import load_dotenv
import os
load_dotenv()
api_key=os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set in environment variables")
print("==================================================")
print("Groq API key loaded successfully")
print("==================================================")

#==================================================
#  import libraries
#==================================================
import groq
import json
import time
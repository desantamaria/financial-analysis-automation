import re
import requests
import pandas as pd
import config as cfg
from eodhd import APIClient
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
import os

api_key = 'demo'
api = APIClient(api_key)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Initialize Groq
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# A function to count the number of tokens
def count_tokens(text):
    tokens = text.split()  
    return len(tokens)

def retrieve_articles():
    resp = api.financial_news(s = "AAPL.US", from_date = '2024-01-01', to_date = '2024-01-30', limit = 100)
    df = pd.DataFrame(resp) # converting the json output into datframe
    print(df.tail())
    

    
    
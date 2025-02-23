import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel
from langgraph.checkpoint.memory import MemorySaver

import pandas as pd
from io import StringIO
import json

from tavily import TavilyClient
from typing import TypedDict, Annotated, List
import operator

from langgraph.graph import StateGraph, END

# Get API key
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Memory node
memory = MemorySaver()

# Define llm
# llm_name = "gpt-3.5-turbo"
# model = ChatOpenAI(model=llm_name)
llm_name = "qwen2.5:14b"
model = ChatOllama(model=llm_name)

# Define tavily client
tavily = TavilyClient()

class AgentState(TypedDict):
    task : str
    competitors : List[str]
    csv_file : str
    financial_data : str
    analysis : str
    competitor_data : str
    comparison : str
    comparison : str
    feedback : str
    report : str
    content : List[str]
    revsion_numner : int
    max_revision : int
    
class Queries(BaseModel):
    queries: List[str]
    
# Define the prompts for each node - IMPROVE AS NEEDED
GATHER_FINANCIALS_PROMPT = """You are an expert financial analyst. Gather the financial data for the given company. Provide detailed financial data."""
ANALYZE_DATA_PROMPT = """You are an expert financial analyst. Analyze the provided financial data and provide detailed insights and analysis."""
RESEARCH_COMPETITORS_PROMPT = """You are a researcher tasked with providing information about similar companies for performance comparison. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""
COMPETE_PERFORMANCE_PROMPT = """You are an expert financial analyst. Compare the financial performance of the given company with its competitors based on the provided data.
**MAKE SURE TO INCLUDE THE NAMES OF THE COMPETITORS IN THE COMPARISON.**"""
FEEDBACK_PROMPT = """You are a reviewer. Provide detailed feedback and critique for the provided financial comparison report. Include any additional information or revisions needed."""
WRITE_REPORT_PROMPT = """You are a financial report writer. Write a comprehensive financial report based on the analysis, competitor research, comparison, and feedback provided."""
RESEARCH_CRITIQUE_PROMPT = """You are a researcher tasked with providing information to address the provided critique. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""


    







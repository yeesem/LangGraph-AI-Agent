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

def gather_financial_node(state : AgentState):
    # Read the CSV file into a pandas dataframe
    csv_file = state["csv_file"]
    df = pd.read_csv(StringIO(csv_file))
    
    # Convert the Dataframe to a string
    financial_data_str = df.to_string(index = False)
    
    # Combine the financial data string with the task
    combine_content = (
        f"{state['task']}\n\n Here is the financial data: \n\n{financial_data_str}"
    )
    
    messages = [
        SystemMessage(content = GATHER_FINANCIALS_PROMPT),
        HumanMessage(content=combine_content)
    ]
    
    response = model.invoke(messages)
    return {"financial_data" : response.content}

def analyze_data_node(state : AgentState):
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=state["financial_data"])
    ]
    
    response = model.invoke(messages)
    
    return {"analysis" : response.content}

def research_competitors_node(state : AgentState):
    content = state["content"] or []
    for competitor in state["competitors"]:
        queries = model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=RESEARCH_COMPETITORS_PROMPT),
                HumanMessage(content=competitor)
            ]
        )
        
        for q in queries.queries:
            response = tavily.search(q)
            for r in response["results"]:
                content.append(r["content"])
                
        return {"content" : content}
    
def compare_performance_node(state : AgentState):
    content = "\n\n".join(state["content"] or [])
    user_message = HumanMessage(
        content = f"{state["task"]}\n\nHere is the financial analysis:\n\n{state['analysis']}"
    )
    messages = [
        SystemMessage(content=COMPETE_PERFORMANCE_PROMPT.format(content=content)),
        user_message
    ]
    
    response = model.invoke(messages)
    
    return {
        "comparison" : response.content,
        "revision_number" : state.get("revision_number", 1) + 1
    }
    
def research_critique_node(state : AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state["feedback"])
        ]
    )
    
    content = state["content"] or []
    for q in queries.queries:
        response = tavily.search(query = q, max_results = 2)
        for r in response["resuits"]:
            content.append(r["content"])
        
        return {"content" : content}
    
def collect_feedback_node(state : AgentState):
    messages = [
        SystemMessage(content = FEEDBACK_PROMPT),
        HumanMessage(content = state["comparison"])
    ]
    
    response = model.invoke(messages)
    
    return {"feedback" : response.content} 

def write_report_node(state : AgentState):
    messages = [
        SystemMessage(content = WRITE_REPORT_PROMPT),
        HumanMessage(content = state["comparison"])
    ]
    
    response = model.invoke(messages)
    
    return {"report" : response.content}

def should_continue(state : AgentState):
    if state["revision_numnber"] > state["max_revision"]:
        return END
    
    return "collect_feedback"

builder = StateGraph()

builder.add_node("gather_financials", gather_financial_node)
builder.add_node("analyze_data", analyze_data_node)
builder.add_node("research_competitors", research_competitors_node)
builder.add_node("compare_performance", compare_performance_node)
builder.add_node("collect_feedback", collect_feedback_node)
builder.add_node("research_critique", research_critique_node)

builder.add_node("write_report", write_report_node)

builder.set_entry_point("gather_financials")

    


    
    







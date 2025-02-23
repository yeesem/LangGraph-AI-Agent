import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel
from langgraph.checkpoint.memory import MemorySaver

import streamlit as st
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
# llm_name = "gpt-4o-mini"
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
    revision_number : int
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
    content = state.get("content", [])
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
        content = f"{state['task']}\n\nHere is the financial analysis:\n\n{state['analysis']}"
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
    
    content = state.get("content", [])
    for q in queries.queries:
        response = tavily.search(query = q, max_results = 2)
        for r in response["results"]:
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
    if state["revision_number"] > state["max_revision"]:
        return END
    
    return "collect_feedback"

builder = StateGraph(AgentState)

builder.add_node("gather_financials", gather_financial_node)
builder.add_node("analyze_data", analyze_data_node)
builder.add_node("research_competitors", research_competitors_node)
builder.add_node("compare_performance", compare_performance_node)
builder.add_node("collect_feedback", collect_feedback_node)
builder.add_node("research_critique", research_critique_node)

builder.add_node("write_report", write_report_node)

builder.set_entry_point("gather_financials")

builder.add_conditional_edges(
    "compare_performance",
    should_continue,
    {END : END, "collect_feedback" : "collect_feedback"}
)

builder.add_edge("gather_financials", "analyze_data")
builder.add_edge("analyze_data", "research_competitors")
builder.add_edge("research_competitors", "compare_performance")
builder.add_edge("collect_feedback", "research_critique")
builder.add_edge("research_critique", "compare_performance")
builder.add_edge("compare_performance", "write_report")

graph = builder.compile(checkpointer = memory)


# ==== For Console Testing ====
# def read_csv_file(file_path):
#     with open(file_path, "r") as file:
#         print("Reading CSV file...")
#         return file.read()

# if __name__ == "__main__":
#     task = "Analyze the financial performance of our (MegaAICo) company compared to competitors"
#     competitors = ["Microsoft", "Nvidia", "Google"]
#     csv_file_path = (
#         "./data/financials.csv"  # Update with the actual path to your CSV file
#     )

#     if not os.path.exists(csv_file_path):
#         print(f"CSV file not found at {csv_file_path}")
#     else:
#         print("Starting the conversation...")
#         csv_data = read_csv_file(csv_file_path)

#         initial_state = {
#             "task": task,
#             "competitors": competitors,
#             "csv_file": csv_data,
#             "max_revisions": 2,
#             "revision_number": 1,
#         }
#         thread = {"configurable": {"thread_id": "1"}}

#         for s in graph.stream(initial_state, thread):
#             print(s)
# === End Console Testing ===  

def main():
    st.title("Financial Performance Reporting Agent")

    task = st.text_input(
        "Enter the task:",
        "Analyze the financial performance of our company (MyAICo.AI) compared to competitors",
    )
    competitors = st.text_area("Enter competitor names (one per line):").split("\n")
    max_revisions = st.number_input("Max Revisions", min_value=1, value=2)
    uploaded_file = st.file_uploader(
        "Upload a CSV file with the company's financial data", type=["csv"]
    )
    
    if st.button("Start Analysis") and uploaded_file is not None:
        # Read the uploaded CSV file
        csv_data = uploaded_file.getvalue().decode("utf-8")
        
        initial_state = {
            "task" : task,
            "competitors" : [comp.strip() for comp in competitors if comp.strip()],
            "csv_file" : csv_data,
            "max_revision" : max_revisions,
            "revision_number" : 1
        }
        
        thread = {"configurable" : {"thread_id" : "1"}}
        
        final_state = None
        
        for s in graph.stream(initial_state, thread):
            st.write(s)
            final_state = s
            
            print("state -----> ", s)
            print()
            print("----------------------------------------------")
        
        if final_state and "write_report" in final_state:
            st.subheader("Final Report")
            st.write(final_state["write_report"]["report"])
            
if __name__ == "__main__":
    main()

# === Save the graph image ===
# from IPython.display import Image, display
# try:
#     img = Image(graph.get_graph().draw_mermaid_png())
    
#     # Save the image
#     with open("test.png", "wb") as f:
#         f.write(img.data)
    
# except Exception:
#   pass
# === End save the graph image ===

    


    
    







import os
import json

from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END

from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage"""
    def __init__(self, tools : list):
        self.tools_by_name = {tool.name : tool for tool in tools}
        
    def __call__(self, inputs : dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No messages in inputs")
        
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call['args']
            )
            outputs.append(
                ToolMessage(
                    content = json.dumps(tool_result),
                    name = tool_call['name'],
                    tool_call_id = tool_call['id']
                )
            )
        return {"messages" : outputs}


class State(TypedDict):
    messages : Annotated[list, add_messages]
    
def bot(state: State):
    return {"messages" : [model_with_tools.invoke(state["messages"])]}

def route_tools(state: State)-> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message contains a tool call
    Otherwise, route to the end
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError("No messages found in input state")
    
    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        return "tools"
    
    return "__end__"
    

tool = TavilySearchResults(max_results = 2)
tools = [tool]
tool_node = BasicToolNode(tools)

model = ChatOllama(model = "llama3.1")
model_with_tools = model.bind_tools(tools)

graph_builder = StateGraph(State)
graph_builder.add_node("bot", bot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
   "bot",
   route_tools,
   {"tools" : "tools", "__end__" : END} 
)

graph_builder.add_edge("tools", "bot")

graph_builder.set_entry_point("bot")
graph_builder.set_finish_point("bot")

graph = graph_builder.compile()

while True:
    user_input = input("User : ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages" : ("user", user_input)}):
        for value in event.values():
            print("Assistant : ", value["messages"][-1].content, "\n----------------------------------------------------\n")
            
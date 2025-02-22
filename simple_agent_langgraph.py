import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END

from langchain_ollama import ChatOllama

model = ChatOllama(model = "deepseek-r1:8b")

class State(TypedDict):
    messages : Annotated[list, add_messages]
    
def bot(state: State):
    return {"messages" : [model.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("bot", bot)

graph_builder.set_entry_point("bot")
graph_builder.set_finish_point("bot")

graph = graph_builder.compile()

# res = graph.invoke({"messages" : ["Hello, how are you?"]})
# print(res["messages"])

while True:
    user_input = input("User : ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages" : ("user", user_input)}):
        for value in event.values():
            print("Assistant : ", value["messages"][-1].content)
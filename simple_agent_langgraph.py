import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from openai import OpenAI

from langgraph.graph.message import add_messages

class State(TypedDict):
    messages : Annotated[list, add_messages]
    
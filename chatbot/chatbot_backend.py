from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage 
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

llm=ChatGroq(
    groq_api_key = os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    # take user query from state
    messages = state["messages"]
    # send to llm
    response = llm.invoke(messages)
    # response store to state
    return {"messages": [response]}

checkpointer = MemorySaver()

# create graph
graph = StateGraph(ChatState)
# add nodes
graph.add_node("chat_node", chat_node)

# add edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer = checkpointer)


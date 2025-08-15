import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage


thread_id = "1"
CONFIG = {"configurable": {"thread_id": thread_id}}

# loading the conversation history
if "message_history" not in st.session_state:
     st.session_state["message_history"] = []
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")
if user_input:

    # first add the user message to message history
    st.session_state["message_history"].append({"role":"user", "content": user_input})

    with st.chat_message("user"):
        st.text(user_input)

    # AI message    
    response = chatbot.invoke({"messages": [HumanMessage(content= user_input)]}, config = CONFIG)
    ai_message = response["messages"][-1].content
    
    st.session_state["message_history"].append({"role":"assistant", "content": ai_message})
    with st.chat_message("assistant"):
        st.text(ai_message)


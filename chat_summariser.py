from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Prepare summary of a chat with client. Shall include information about client, it's family, reason for purchasing flat, wshed building complex, wished financial conditions]"
    ),
    ("human", "Chat history:\n{history}")
])


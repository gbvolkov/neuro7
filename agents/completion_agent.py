from langgraph.prebuilt import create_react_agent
from utils.utils import sub_dict
from v01.retriever import search_kb
from agents.tools.tools import (get_list_of_complexes,
                   get_developer_info,
                   get_complex_info,
                   complexes
                   )
import config

from langchain_gigachat import GigaChat
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

agent_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3, frequency_penalty=0.6)
#agent_llm = GigaChat(
#            credentials=config.GIGA_CHAT_AUTH, 
#            model="GigaChat-Pro",
#            verify_ssl_certs=False,
#            temperature=0,
#            scope = config.GIGA_CHAT_SCOPE)

with open("prompts/post_call_mode_prompt.txt", encoding="utf-8") as f:
    prompt_txt = f.read()

#prompt = PromptTemplate.from_template(prompt_txt)
#completion_agent = prompt | agent_llm

completion_agent = create_react_agent(
    model=agent_llm,
    tools = [],
    prompt=(
        prompt_txt
    ),
    name="completion_agent",
    debug=config.DEBUG_WORKFLOW,
)
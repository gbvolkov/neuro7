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

complexes_names = "',".join([f"{name['name']} (aka {name['alternative_name']})" for name in sub_dict(complexes, ["name", "alternative_name"])])

agent_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
#agent_llm = GigaChat(
#            credentials=config.GIGA_CHAT_AUTH, 
#            model="GigaChat-Pro",
#            verify_ssl_certs=False,
#            temperature=0,
#            scope = config.GIGA_CHAT_SCOPE)

kb_agent = create_react_agent(
    model=agent_llm, #"openai:gpt-4.1-mini",
    tools=[get_list_of_complexes, get_developer_info, get_complex_info],# search_kb],
    prompt=(
        "You are an agent retrieving information about building complexes. "
        "You can return information about (1) building complexes available for sales; (2) developers; (3) facilities available for the complex; (4) financial conditions like loan availability, discounts and so on.\n"
        "Whatever task you got, YOU ARE PROHIBITED BY LAW to provide ANY information but information directly received from your tools!\n"
        "You always returns only information obtained from your tools.\n\n"
        "INSTRUCTIONS:\n"
        f"- Assist ONLY with tasks related to retrieval information about building complexes {complexes_names}\n"
        "- Do not answer questions related to pricing and details of flats available within building complexes\n"
        "- IMPORTANT: Always provide ONLY information returned by your tools. YOU ARE PROHIBITED to make up information!\n"
        "- If information provided by your tools does not contain information requested by user, answer that you do not hold this information.\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="kb_agent",
    debug=config.DEBUG_WORKFLOW,
)
from langgraph.prebuilt import create_react_agent
from utils import sub_dict
from retriever import search_kb
from tools import (get_list_of_complexes,
                   get_developer_info,
                   get_complex_info,
                   complexes
                   )

complexes_names = "',".join([name["name"] for name in sub_dict(complexes, ["name"])])

kb_agent = create_react_agent(
    model="openai:gpt-4.1-mini",
    tools=[get_list_of_complexes, get_developer_info, get_complex_info],# search_kb],
    prompt=(
        "You are an agent retrieving information about building complexes.\n\n"
        "INSTRUCTIONS:\n"
        f"- Assist ONLY with tasks related to retrieval information about building complexes {complexes_names}\n"
        "- Do not answer questions related to pricing and details of flats available within building complexes\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="kb_agent",
)
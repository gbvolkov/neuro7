from langgraph.prebuilt import create_react_agent
from tools import agree_call


contact_agent = create_react_agent(
    model="openai:gpt-4.1-mini",
    tools=[agree_call],
    prompt=(
        "You are an agent which returns time slots available for commercial manager to call to client.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with tasks related to defining time slots available for commercial manager to call to client\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="contact_agent",
)
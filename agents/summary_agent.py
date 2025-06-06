# agents/schedule_confirm_agent.py

from typing import Any, Dict

from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langgraph.types import Send

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

import config

def confirm_summary(summary: str) -> Command:
    """
    Инструмент, вызываемый на основе summary, переданного от Supervisor.
    inputs["summary"] содержит готовый текст-резюме разговора, который
    сформировал Supervisor. Здесь просто выставляем call_scheduled=True.
    (call_time оставляем пустым, т.к. в summary уже есть вся нужная инфа.)
    """
    update_payload = {
        "call_scheduled": True,
        "call_time": "",
        "messages": [AIMessage(content="__FIXED__365477__")]
    }
    return Command(
        update=update_payload,
        goto="post_call_next_node",
        graph=Command.PARENT,  
    )


# ───────────────────────────────────────────────────────────────────────────────
# Агент создаётся через create_react_agent:
#   - Всегда вызывает confirm_summary({'summary': <summary>})
#   - Возвращает ровно одну строку:
#       "Ваша информация обработана и сохранена в наших системах."
# ───────────────────────────────────────────────────────────────────────────────

agent_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

call_fixing_agent = create_react_agent(
    model=agent_llm,
    tools=[confirm_summary],
    prompt=(
        "Вы — вспомогательный агент. На вход получаете уже готовое summary разговора\n"
        "(summary лежит в inputs['summary']). Обязаны:\n"
        "1) Вызвать инструмент confirm_summary({'summary': <тот самый summary>}),\n"
        "   который вернёт {'call_scheduled': True, 'call_time': ''}.\n"
        "2) НЕМЕДЛЕННО вернуть ровно одно предложение:\n\n"
        "    «Ваша информация обработана и сохранена в наших системах.»\n\n"
        "Никаких других слов, пояснений или отступлений. Конец."
    ),
    name="call_fixing_agent",
    debug=config.DEBUG_WORKFLOW,
)

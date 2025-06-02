# agents/post_call_mode.py

from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

# ─────────────────────────────────────────────────────────────────────
# Здесь создаём свои LLM-инстансы для работы внутри post_call_mode:
# 1) classifier_llm — с temperature=0.0, чтобы надёжно распознавать «хочет ли клиент изменить звонок»
# 2) response_llm   — с небольшой температурой (0.7) для генерации коротких ответов
# ─────────────────────────────────────────────────────────────────────

classifier_llm = ChatOpenAI(model="gpt-4.1", temperature=0.0)
response_llm   = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)


def user_wants_to_change_call_llm(last_text: str) -> bool:
    """
    Определяем с помощью LLM, хочет ли пользователь изменить/отменить ранее согласованный звонок с менеджером.
    Модель должна ответить «да» или «нет».
    Если первый символ ответа — «д» или «y», считаем, что хочет изменить.
    """
    classifier_prompt = (
        "Вам даётся только что поступившее сообщение клиента. "
        "Определите, хочет ли клиент изменить или отменить ранее согласованный звонок с менеджером. "
        "Если хочет — ответьте «да». Если не хочет — ответьте «нет». "
        "Никаких других пояснений не давайте.\n\n"
        "Сообщение клиента:\n"
    )
    messages = [
        SystemMessage(content=classifier_prompt),
        HumanMessage(content=last_text)
    ]
    resp = classifier_llm(messages=messages)
    answer = resp.content.strip().splitlines()[0].lower()
    return answer.startswith("д") or answer.startswith("y")


def _extract_last_human_text(state: Dict[str, Any]) -> str:
    """
    Находит самое последнее сообщение пользователя (HumanMessage) в state["messages"]
    и возвращает его текст (склеивая поля "text" из списка, если content — список).
    Если не нашлось ни одного HumanMessage, возвращает пустую строку.
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            raw = msg.content
            if isinstance(raw, list):
                return " ".join(item["text"] for item in raw if item.get("type") == "text")
            else:
                return str(raw)
    return ""


def post_call_mode_condition(state: Dict[str, Any]) -> bool:
    """
    Возвращает True, если:
      1) звонок уже согласован (state['call_scheduled'] == True), и
      2) LLM НЕ обнаружило в последнем сообщении клиента желание изменить/отменить звонок.
    В противном случае — False.
    """
    if not state.get("call_scheduled", False):
        return False

    last_text = _extract_last_human_text(state)
    if not last_text:
        # Если не можем найти текст последнего сообщения от пользователя, остаёмся в текущем узле
        return False

    wants_change = user_wants_to_change_call_llm(last_text)
    return not wants_change


def cancel_post_call_condition(state: Dict[str, Any]) -> bool:
    """
    Возвращает True, если:
      1) звонок уже согласован, и
      2) LLM увидело в последнем сообщении клиента желание изменить/отменить звонок.
    Иначе — False.
    """
    if not state.get("call_scheduled", False):
        return False

    last_text = _extract_last_human_text(state)
    if not last_text:
        return False

    return user_wants_to_change_call_llm(last_text)


def post_call_mode_handler(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Генерирует ответ в режиме «после согласования звонка»:
      1) Считывает системный промпт из prompts/post_call_mode_prompt.txt.
      2) Отбирает из state["messages"]:
         а) все HumanMessage (сообщения от клиента),
         б) только те AIMessage, которые пришли от «supervisor» (без инструментов и без других агентов).
      3) Собирает цепочку: [SystemMessage(пост-звонковый-промпт), ...filtered_messages...]
      4) Вызывает response_llm и возвращает единственный AIMessage.
    """
    # 1. Считываем системный промпт
    prompt_path = Path("prompts/post_call_mode_prompt.txt")
    prompt_txt = prompt_path.read_text(encoding="utf-8")

    # 2. Отбираем только нужные сообщения: HumanMessage и AIMessage одной «команды» (supervisor)
    filtered_messages = []
    for msg in state["messages"]:
        # Берём любые сообщения пользователя
        if isinstance(msg, HumanMessage):
            filtered_messages.append(msg)
            continue

        # Если это AIMessage, проверяем, что оно от «supervisor» (нет полей "tool" и agent_name!="supervisor")
        if isinstance(msg, AIMessage):
            akw = getattr(msg, "additional_kwargs", {}) or {}
            tool_flag = akw.get("tool")
            agent_flag = akw.get("agent_name")
            # Если нет ключа "tool", и agent_name отсутствует или равно "supervisor" → включаем
            if tool_flag is None and (agent_flag is None or agent_flag == "supervisor"):
                filtered_messages.append(msg)

    # 3. Собираем цепочку для response_llm
    llm_messages = [SystemMessage(content=prompt_txt)]
    for msg in filtered_messages:
        llm_messages.append(msg)

    # 4. Запрашиваем ответ от response_llm
    response = response_llm(messages=llm_messages)

    # 5. Возвращаем один AIMessage в формате LangGraph
    return {"messages": [AIMessage(content=response.content)]}

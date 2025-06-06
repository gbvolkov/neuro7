# File: supervisor_new.py

import uuid
import os
from typing import Literal

# ── 1. ENVIRONMENT ─────────────────────────────────────────────────────────────
os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"]  = "true"

from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_handoff_tool
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.modifier import RemoveMessage

import config
from utils.utils import ModelType, sub_dict
from agents.state.state import State
from agents.user_info import user_info
from agents.kb_agent import kb_agent
from agents.contact_agent import contact_agent
from agents.pricing_agent import get_retrieval_agent
from agents.tools.supervisor_tools import create_handoff_tool_no_history
from agents.tools.tools import complexes, initiate_schedule_tool


# ── 2. LLM INSTANCES ───────────────────────────────────────────────────────────
agent_llm   = ChatOpenAI(model="gpt-4.1", temperature=1.0)  # основной супервизор
summary_llm = ChatOpenAI(model="gpt-4.1", temperature=0.7)  # для “Всё верно?”‐сообщений
detect_llm  = ChatOpenAI(model="gpt-4.1", temperature=0.7)  # для детекции изменений
confirm_llm = ChatOpenAI(model="gpt-4.1", temperature=0.7)  # для подтверждения “Да/Нет”


# ── 3. MEMORY-RESET / INTRODUCTION HELPERS ─────────────────────────────────────

def reset_memory_condition(state: State) -> str:
    """
    Если в последнем сообщении пользователя есть {"type":"reset"},
    очищаем историю. Иначе – переходим к check_introduction_needed.
    """
    if state.get("messages"):
        last_msg = state["messages"][-1]
        if isinstance(last_msg.content, list):
            first_piece = last_msg.content[0]
            if isinstance(first_piece, dict) and first_piece.get("type") == "reset":
                return "reset_memory"
    return "introduce_or_check"


def reset_memory(state: State) -> State:
    """
    Возвращаем RemoveMessage для каждого сообщения, чтобы очистить историю.
    """
    all_ids = [m.id for m in state.get("messages", [])]
    return {
        "messages": [RemoveMessage(id=mid) for mid in all_ids]
    }


def check_introduction_needed(state: State) -> Literal["introduce_and_respond", "check_scheduled"]:
    """
    Если агент ещё не представился и есть ровно одно сообщение от пользователя, отправляем приветствие.
    Иначе – сразу идём в check_scheduled.
    """
    introduced = state.get("agent_introduced", False)
    msgs = state.get("messages", [])
    user_msgs = [m for m in msgs if hasattr(m, "type") and m.type == "human"]
    if not introduced and len(user_msgs) == 1:
        return "introduce_and_respond"
    return "check_scheduled"


def introduce_and_respond(state: State) -> State:
    """
    Одноразовое приветствие. Ставит agent_introduced=True и возвращает AIMessage с приветствием.
    После – пойдём в check_scheduled.
    """
    with open("prompts/welcome_prompt.txt", encoding="utf-8") as f:
        prompt_txt = f.read()
    intro_msg = AIMessage(content=prompt_txt)
    return {
        "messages": [intro_msg],
        "agent_introduced": True
    }


# ── 4. CHECK_SCHEDULED NODE ────────────────────────────────────────────────────

def check_scheduled(state: State) -> Command:
    """
    Если is_scheduled=True → переходим в simplified_handler.
    Иначе → supervisor.
    """
    is_scheduled = state.get("is_scheduled", False)
    print(f"DEBUG: check_scheduled===>{is_scheduled}")
    if is_scheduled:
        return Command(goto="simplified_handler")
    else:
        return Command(goto="supervisor")


# ── 5. BUILD SUPERVISOR AGENT ──────────────────────────────────────────────────

def build_supervisor_agent() -> StateGraph:
    """
    Создаём «супервизора»:
      - Региструем Retrieval-агентов для БД (комплексы/цены).
      - Добавляем инструмент initiate_schedule_tool, чтобы LLM мог вызвать contact_agent.
    """
    # 5.1. Retrieval Agents (цены/БД)
    db_vesna    = get_retrieval_agent("vesna")
    db_andersen = get_retrieval_agent("andersen")
    db_7ya      = get_retrieval_agent("7ya")

    # 5.2. Handoff Tools (без истории) для каждой БД
    ho_vesna = create_handoff_tool_no_history(
        agent_name="vesna_flat_info_retriever",
        agent_purpose="дать информацию по квартирам поселка-парк «Весна»"
    )
    ho_andersen = create_handoff_tool_no_history(
        agent_name="andersen_flat_info_retriever",
        agent_purpose="дать информацию по квартирам ЖК «Андерсен»"
    )
    ho_7ya = create_handoff_tool_no_history(
        agent_name="7ya_flat_info_retriever",
        agent_purpose="дать информацию по квартирам ЖК «7Я»"
    )

    # 5.3. Собираем все Handoff Tools для супервизора:
    ho_tools = [
        create_handoff_tool_no_history(
            agent_name="kb_agent",
            agent_purpose=(
                "получать и давать информацию о жилых комплексах, "
                "застройщиках, инфраструктуре, финансовых условиях (ипотека, скидки и т. д.)"
            )
        ),
        # Инструмент для запуска согласования звонка (contact_agent)
        create_handoff_tool(agent_name="contact_agent"),
        ho_vesna,
        ho_andersen,
        ho_7ya,
        initiate_schedule_tool  # инструмент “initiate_schedule”
    ]

    # 5.4. Промпт, который говорит LLM: «если нужно договориться о звонке – вызывай initiate_schedule»
    with open("prompts/working_prompt_super.txt", encoding="utf-8") as f:
        prompt_txt = f.read()

    prompt_txt = (
        f"{prompt_txt}\n"
        f"Список жилых комплексов (контекст для LLM):\n"
        f"{sub_dict(complexes, ['id','name','alternative_name','district','ready_date','number_of_houses','comfort_level'])}\n\n"
        "Если нужно согласовать звонок – вызывай инструмент \"initiate_schedule\". "
        "contact_agent затем вернёт доступные слоты, и мы перейдём к summary_agent."
    )

    # 5.5. Собираем и компилируем supervisor_agent
    supervisor_agent = create_supervisor(
        model=agent_llm,
        agents=[kb_agent, contact_agent, db_vesna, db_andersen, db_7ya],
        prompt=prompt_txt,
        tools=ho_tools,
        add_handoff_messages=False,
        add_handoff_back_messages=False,
        output_mode="last_message",
        parallel_tool_calls=False,
        supervisor_name="neuro7"
    ).compile(name="neuro7", debug=config.DEBUG_WORKFLOW)

    return supervisor_agent


# ── 6. SUMMARY_AGENT NODE ─────────────────────────────────────────────────────

def summary_agent_node(state: State) -> Command:
    """
    После contact_agent: state['scheduled_time'] уже заполнено.
    Формируем сообщение «Вы подтвердили звонок на {slot}. Всё верно?» и помечаем awaiting_confirmation=True.
    """
    chosen_slot = state.get("scheduled_time", "(неизвестно)")
    prompt = [
        HumanMessage(content=(
            f"Пользователь выбрал слот {chosen_slot}. "
            "Сформируй сообщение:\n"
            f"\"Вы подтвердили звонок на {chosen_slot}. Всё верно?\""
        ))
    ]
    llm_out = summary_llm(prompt).content
    ai_msg = AIMessage(content=llm_out)

    print(f"DEBUG: summary_agent_node===>{llm_out}")

    return Command(
        goto="check_summary_confirmation",
        update={
            "messages": state["messages"] + [ai_msg],
            "awaiting_confirmation": True
        }
    )


# ── 7. CHECK_SUMMARY_CONFIRMATION NODE ─────────────────────────────────────────

def check_summary_confirmation_node(state: State) -> Command:
    """
    Когда пользователь отвечает «Да» или «Нет» на «Всё верно?»‐сообщение.
    Если «Да» → crm_agent. Иначе → supervisor (чтобы LLM мог предложить новый слот).
    """
    if not state.get("awaiting_confirmation", False):
        return Command(goto="END")

    last_user = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user = m.content if isinstance(m.content, str) else ""
            break

    chosen_slot = state.get("scheduled_time", "(неизвестно)")
    prompt = [
        HumanMessage(content=(
            f"Пользователь подтвердил звонок на {chosen_slot}. "
            f"Последнее сообщение: «{last_user}». "
            "Выведите JSON:\n"
            "{\"confirmed\": true} если это «Да»;\n"
            "{\"confirmed\": false} иначе."
        ))
    ]
    llm_out = confirm_llm(prompt).content
    print(f"DEBUG: check_summary_confirmation_node===>{llm_out}")

    try:
        parsed = __import__("json").loads(llm_out)
    except Exception:
        parsed = {"confirmed": False}

    if parsed.get("confirmed", False):
        return Command(goto="crm_agent")
    else:
        return Command(goto="supervisor", graph=Command.PARENT)


# ── 8. CRM_AGENT NODE ─────────────────────────────────────────────────────────

def crm_agent_node(state: State) -> Command:
    """
    Когда пользователь подтвердил (“Да”) в check_summary_confirmation_node.
    Сохраняем звонок в CRM, помечаем is_scheduled=True,
    отправляем финальное подтверждение и уходим в simplified_handler.
    """
    final_time = state.get("scheduled_time", "")
    # Здесь обычно идёт вызов CRM API:
    # resp = crm_api_create_appointment(user_name, user_phone, final_time)
    # if not resp.success: … возврат ошибки

    ai_confirm = AIMessage(content=f"Ваш звонок запланирован на {final_time}.")
    print(f"DEBUG: crm_agent_node===>{final_time}")
    return Command(
        goto="simplified_handler",
        update={
            "messages": state["messages"] + [ai_confirm],
            "is_scheduled": True
        }
    )


# ── 9. REMIND & DETECT SUBGRAPH ─────────────────────────────────────────────────

def remind_node(state: State) -> Command:
    """
    Добавляем напоминание: «Напоминаем: ваш звонок запланирован на {scheduled_time}.»
    Если звонок не был ранее запланирован, просто уходим в END.
    Затем – в detect_node.
    """
    if not state.get("is_scheduled", False) or not state.get("scheduled_time"):
        return Command(goto="END")

    scheduled_time = state.get("scheduled_time", "(неизвестно)")
    reminder = AIMessage(content=f"Напоминаем: ваш звонок запланирован на {scheduled_time}.")
    print(f"DEBUG: remind_node===>{scheduled_time}")
    return Command(
        goto="detect_node",
        update={"messages": state["messages"] + [reminder]}
    )


def detect_node(state: State) -> Command:
    """
    Спрашиваем LLM: «клиент хочет изменить время?»
    Если звонок не был ранее запланирован, просто уходим в END.
    Если да – идём обратно в supervisor (Command.PARENT).
    Иначе – END (выходим).
    """
    if not state.get("is_scheduled", False) or not state.get("scheduled_time"):
        return Command(goto="END")

    last_user = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user = m.content if isinstance(m.content, str) else ""
            break

    scheduled_time = state.get("scheduled_time", "")
    prompt = [
        HumanMessage(content=(
            f"Напоминание: у клиента звонок запланирован на {scheduled_time}. "
            f"Последнее сообщение: «{last_user}». "
            "Выведите JSON:\n"
            "{\"major_change\": true} если клиент хочет изменить звонок;\n"
            "{\"major_change\": false} иначе."
        ))
    ]
    llm_resp = detect_llm(prompt).content
    print(f"DEBUG: detect_node===>{llm_resp}")
    try:
        parsed = __import__("json").loads(llm_resp)
    except Exception:
        parsed = {"major_change": False}

    if parsed.get("major_change", False):
        return Command(goto="supervisor", graph=Command.PARENT)
    return Command(goto="END")


# ── 9.1. Собираем подграф «упрощёнки» ───────────────────────────────────────────
sub_builder = StateGraph(State)
sub_builder.add_node("remind_node", remind_node)
sub_builder.add_node("detect_node", detect_node)
sub_builder.add_edge(START, "remind_node")
sub_builder.add_edge("remind_node", "detect_node")
sub_builder.add_edge("detect_node", END)
simplified_graph = sub_builder.compile()


# ── 10. PARENT GRAPH CONSTRUCTION ──────────────────────────────────────────────

def initialize_agent(model: ModelType = ModelType.GPT):
    supervisor_agent = build_supervisor_agent()

    # ── 10.1. Разделяем “store” и “checkpointer” ──────────────────────────────
    #    - InMemoryStore (langgraph.store.memory) для кросс-thread памяти (user-level).
    #    - InMemorySaver (langgraph.checkpoint.memory) для snapshot/checkpoint каждой супер-шаги.
    memory_store  = InMemoryStore()
    checkpointer  = InMemorySaver()

    parent = (
        StateGraph(State)

        # 1) fetch_user_info — первый узел, который получает (state, config, store)
        .add_node("fetch_user_info", user_info)

        # 2) reset_memory или сразу introduce_or_check
        .add_node("reset_memory", reset_memory)

        # 3) introduction
        .add_node("introduce_and_respond", introduce_and_respond)

        # 4) check_scheduled (сам выберет goto)
        .add_node("check_scheduled", check_scheduled)

        # 5) supervisor_agent (комплексный LLM-супервизор)
        .add_node("supervisor", supervisor_agent)

        # 6) contact_agent  ← сюда LLM сам вызовет как инструмент, если потребуется
        .add_node("contact_agent", contact_agent)

        # 7) summary_agent (только “Всё верно?”)
        .add_node("summary_agent", summary_agent_node)

        # 8) Новый узел для проверки Да/Нет
        .add_node("check_summary_confirmation", check_summary_confirmation_node)

        # 9) crm_agent
        .add_node("crm_agent", crm_agent_node)

        # 10) simplified_handler (напоминание + detect)
        .add_node("simplified_handler", simplified_graph)

        # ─ Edges / Conditions ────────────────────────────────────────────────
        .add_edge(START, "fetch_user_info")

        # fetch_user_info → reset_memory или introduce_or_check
        .add_conditional_edges("fetch_user_info", reset_memory_condition)
        .add_edge("reset_memory", END)

        # Если reset_memory_condition вернуло "introduce_or_check",
        # следующий узел – check_introduction_needed
        .add_conditional_edges("fetch_user_info", check_introduction_needed)

        # Ветка «introduce → check_scheduled»
        .add_edge("introduce_and_respond", "check_scheduled")

        # check_scheduled сам отдаёт Command(goto…), поэтому
        # нужны оба target’а:
        .add_edge("check_scheduled", "supervisor")
        .add_edge("check_scheduled", "simplified_handler")

        # Если LLM в supervisor_agent НЕ вызвал initiate_schedule_tool,
        # supervisor просто отдаст Command(goto="END") → выходим
        .add_edge("supervisor", END)

        # Если внутри supervisor_agent вызвали contact_agent:
        .add_edge("contact_agent", "summary_agent")

        # summary_agent → check_summary_confirmation
        .add_edge("summary_agent", "check_summary_confirmation")

        # check_summary_confirmation → crm_agent (если “Да”)
        #                      → supervisor (если “Нет”)
        .add_edge("check_summary_confirmation", "crm_agent")
        .add_edge("check_summary_confirmation", "supervisor")

        # crm_agent → simplified_handler
        .add_edge("crm_agent", "simplified_handler")

        # ─── Собираем финальную конфигурацию ─────────────────────────────────
        .compile(checkpointer=checkpointer, store=memory_store, debug=config.DEBUG_WORKFLOW)
    )

    return parent


# ── 11. ТЕСТОВЫЙ ЗАПУСК ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = initialize_agent(model=ModelType.GPT)

    # Визуализация структуры (опционально)
    png_bytes = agent.get_graph().draw_mermaid_png()
    with open("my_graph.png", "wb") as f:
        f.write(png_bytes)

    from langchain_core.messages import HumanMessage

    tutorial_questions = [
        # 1) Простой вопрос (никаких звонков)
        "В каких ЖК вы предлагаете квартиры?",

        # 2) LLM решает запланировать звонок → вызовет contact_agent
        "Андерсен. Давай созвонимся после 17:00",

        # 3) Summary: «Всё верно?» (LLM предложил слот 18:30)
        "Да, всё верно. 18:30",

        # 4) crm_agent ставит is_scheduled=True и уходит в упрощёнку
        "Спасибо!",

        # 5) упрощёнка: напоминание, пользователь спрашивает про цены
        "Да, а в какую цену там квартиры?",

        # 6) упрощёнка: снова напоминание → пользователь меняет запрос “Можно перенести...?”
        "Можно перенести звонок на 18:00?"
    ]

    thread_id = str(uuid.uuid4())
    cfg = {
        "configurable": {
            "user_info": "34446578 34094",  
            "thread_id": thread_id,
        }
    }

    for question in tutorial_questions:
        print("USER:", question)
        response = agent.invoke(
            {"messages": [HumanMessage(content=[{"type": "text", "text": question}])]},
            cfg
        )
        answer = response["messages"][-1].content
        print("ASSISTANT:", answer)
        print("──────────────────────────────────────────")

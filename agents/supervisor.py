
import uuid
import os
import datetime

os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

from utils.utils import ModelType

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from langchain_core.messages import AIMessage

from langchain_core.messages.modifier import RemoveMessage

from agents.state.state import State
from agents.user_info import user_info

import config

from agents.kb_agent import kb_agent
from agents.schedule_call_agent import schedule_call_agent
from agents.pricing_agent import get_retrieval_agent
from agents.tools.supervisor_tools import create_handoff_tool_no_history

from agents.tools.tools import complexes
from utils.utils import sub_dict

from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_handoff_tool
from langchain_openai import ChatOpenAI


COMPLEX_LIST = sub_dict(complexes, ["id", "name", "alternative_name", "district", "ready_date", "number_of_houses", "comfort_level"])

agent_llm = ChatOpenAI(model="gpt-4.1", temperature=1)
#agent_llm = ChatMistralAI(model="mistral-large-latest", temperature=1, frequency_penalty=0.3)

#agent_llm = GigaChat(
#            credentials=config.GIGA_CHAT_AUTH, 
#            model="GigaChat-Pro",
#            verify_ssl_certs=False,
#            temperature=1,
#            scope = config.GIGA_CHAT_SCOPE)

def reset_memory_condition(state: State) -> str:
    if state["messages"][-1].content[0].get("type") == "reset":
        return "reset_memory"
    return "assistant"


def reset_memory(state: State) -> State:
    """
    Delete every message currently stored in the thread’s state.
    """
    all_msg_ids = [m.id for m in state["messages"]]
    # Returning RemoveMessage instances instructs the reducer to delete them
    return {
        "messages": [RemoveMessage(id=mid) for mid in all_msg_ids]
    }

def check_introduction_needed(state: State) -> str:
    """
    Проверяет, нужно ли представляться агенту.
    Представление нужно только при самом первом сообщении от пользователя.
    """
    need_intro = state.get("need_intro", True)
    messages = state.get("messages", [])
    
    # Считаем количество сообщений от пользователя (не системных)
    user_messages = [msg for msg in messages if hasattr(msg, 'type') and msg.type == 'human']
    
    # Представляемся только при первом сообщении пользователя и если еще не представлялись
    if need_intro and len(user_messages) == 1:
        return "introduce_and_respond"
    else:
        return "supervisor"
    
def introduce_and_respond(state: State) -> State:
    """
    Представляется и отвечает на первое сообщение пользователя.
    Этот узел вызывается только один раз - при самом первом сообщении.
    """

    with open("prompts/welcome_prompt.txt", encoding="utf-8") as f:
        prompt_txt = f.read()
    # Первичное представление + ответ на вопрос пользователя
    intro_message = AIMessage(
        content=prompt_txt
    )
    
    # Устанавливаем флаг, что представление состоялось
    # Возвращаем состояние с представлением, а supervisor обработает основной вопрос
    return {
        "messages": [intro_message],
        "need_intro": True
    }

def initialize_agent(model: ModelType = ModelType.GPT):
    db_vesna = get_retrieval_agent("vesna")
    db_andersen = get_retrieval_agent("andersen")
    db_7ya = get_retrieval_agent("7ya")


    ho_vesna = create_handoff_tool_no_history(
        agent_name = "vesna_flat_info_retriever", 
        agent_purpose="provide flats' details for building complex 'vesna' ('Весна').")
    ho_andersen = create_handoff_tool_no_history(
        agent_name = "andersen_flat_info_retriever", 
        agent_purpose="provide flats' details for building complex 'andersen' ('Андерсен').")
    ho_7ya = create_handoff_tool_no_history(
        agent_name = "7ya_flat_info_retriever", 
        agent_purpose="provide flats' details for building complex '7ya' ('7Я', 'Семья').")

    ho_tools = [
        create_handoff_tool_no_history(
            agent_name = "kb_agent", 
            agent_purpose=
                "retrieve from database and provide information about:"
                " (1) building complexes available for sales;"
                " (2) developers;"
                " (3) facilities available for the complex;" 
                " (4) financial conditions like loan availability, discounts and so on"),
        create_handoff_tool_no_history(
            agent_name = "schedule_call_agent", 
            agent_purpose=
                "Schedules date and time for a call with manager\n" 
                "Instruct agent to schedule call time include into instructions current user datetime\n"
                "Agent will check availables slots for manager and return assigned time\n"),
        ho_vesna,
        ho_andersen,
        ho_7ya,
        #get_flats_info_for_complex
    ]

    with open("prompts/working_prompt_super.txt", encoding="utf-8") as f:
        prompt_txt = f.read()
    prompt_txt = f"{prompt_txt}\nСписок жилых комплексов: {COMPLEX_LIST}\n\n"

    supervisor_agent = create_supervisor(
        model=agent_llm, #init_chat_model("openai:gpt-4.1"),
        agents=[kb_agent, schedule_call_agent, db_vesna, db_andersen, db_7ya],
        #agents=[kb_agent, contact_agent],
        prompt=prompt_txt,
        tools=ho_tools,
        add_handoff_messages=False,
        add_handoff_back_messages=True,
        output_mode="last_message",
        parallel_tool_calls=False,
        supervisor_name="neuro7"
    ).compile(name="neuro7", debug = config.DEBUG_WORKFLOW)

    memory = MemorySaver()
    return (
        StateGraph(State)
        .add_node("fetch_user_info", user_info)
        #.add_node("intent_extract", update_customer_ctx)
        .add_node("reset_memory", reset_memory)
        .add_node("introduce_and_respond", introduce_and_respond)
        .add_node("supervisor", supervisor_agent)
        .add_edge(START, "fetch_user_info")
        #.add_edge("fetch_user_info", "intent_extract") 
        #.add_conditional_edges("intent_extract", reset_memory_condition)
        .add_conditional_edges("fetch_user_info", reset_memory_condition)
        .add_edge("reset_memory", END)
        .add_conditional_edges("fetch_user_info", check_introduction_needed)
        .add_edge("introduce_and_respond", "supervisor")
    ).compile(checkpointer=memory, debug=config.DEBUG_WORKFLOW)



if __name__ == "__main__":
    agent = initialize_agent(model=ModelType.GPT)

    #show_graph(assistant_graph)
    from langchain_core.messages import HumanMessage

    # Let's create an example conversation a user might have with the assistant
    tutorial_questions = [
        "В каких ЖК вы предлагаете квартиры?",
        "Какие есть двушки в 7Я?",
        #"А кто строил?",
        #"А какие объекты уже сдали?",
        #"А Александрит ваш?",
        "Я хочу в районе весенней что-нибудь.",
        "Не, не весна-парк, другой в этом районе.",
        "Ну а чё у вас в районе Вессенней?",
        "А Андерсен в каком районе?",
        "А когда сдаёте его?",
        "Расскажи подробнее. У меня сын. Чё там для него есть? Ну вообще - там магазы, чё-нить такое",
        "Подбери мне там квартиру стоимостью до 10 млн и площадью от 40 кв.м. и напиши, какие финансовые усоловия - скидки там и пр",
        "А можешь скинуть список вариантов квартир?",
        "Давай созвонимся сегодня после 17:00",
    ]

    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            # The passenger_id is used in our flight tools to
            # fetch the user's flight information
            "user_info": "3442 587242",
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }

    #_printed = set()
    for question in tutorial_questions:

        response = agent.invoke({"messages": [HumanMessage(content=[{"type": "text", "text": question}])]}, config)

        #events = agent.stream(
        #    {"messages": [HumanMessage(content=[{"type": "text", "text": question}])]}, config, stream_mode="values"
        #)
        answer = response['messages'][-1].content
        print("USER: ", question)
        print("-------------------")
        print("ASSISTANT:")
        print(answer)
        #for event in events:
        #    #_print_event(event, _printed)
        #    _print_response(event, _printed)
        print("===================")

    #print("RESET")
    #events = assistant_graph.invoke(
    #    {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]}, config, stream_mode="values"
    #)


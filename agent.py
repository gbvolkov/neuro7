
import uuid
import os

os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

from utils import ModelType

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

from langchain_core.messages.modifier import RemoveMessage

from state import State
from assistant import Assistant, assistant_factory
from utils import create_tool_node_with_fallback, show_graph, _print_event, _print_response
from user_info import user_info

import config


from kb_agent import kb_agent
from contact_agent import contact_agent
from pricing_agent import create_flat_info_retriever, get_retrieval_agent
from pricing_tools import get_flats_info_for_complex
from supervisor_tools import create_pricing_handoff_tool, create_handoff_tool_no_history
from intent_extractor import update_customer_ctx

from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_handoff_tool
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_gigachat import GigaChat
from langchain_mistralai import ChatMistralAI

#agent_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=1)
agent_llm = ChatMistralAI(model="mistral-large-latest", temperature=1, frequency_penalty=0.3)

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


def initialize_agent(model: ModelType = ModelType.GPT):
    #db_vesna = create_flat_info_retriever("vesna")
    #db_andersen = create_flat_info_retriever("andersen")
    #db_7ya = create_flat_info_retriever("7ya")
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
        create_handoff_tool(agent_name = "contact_agent"),
        ho_vesna,
        ho_andersen,
        ho_7ya,
        #get_flats_info_for_complex
    ]

    with open("prompts/working_prompt_super.txt", encoding="utf-8") as f:
        prompt_txt = f.read()
    supervisor_agent = create_supervisor(
        model=agent_llm, #init_chat_model("openai:gpt-4.1"),
        agents=[kb_agent, contact_agent, db_vesna, db_andersen, db_7ya],
        #agents=[kb_agent, contact_agent],
        prompt=prompt_txt,
        tools=ho_tools,
        add_handoff_messages=False,
        add_handoff_back_messages=False,
        output_mode="last_message",
        parallel_tool_calls=False,
        supervisor_name="neuro7"
    ).compile(name="neuro7", debug = True)

    memory = MemorySaver()
    return (
        StateGraph(State)
        .add_node("fetch_user_info", user_info)
        #.add_node("intent_extract", update_customer_ctx)
        .add_node("reset_memory", reset_memory)
        .add_node("assistant", supervisor_agent)
        .add_edge(START, "fetch_user_info")
        #.add_edge("fetch_user_info", "intent_extract") 
        #.add_conditional_edges("intent_extract", reset_memory_condition)
        .add_conditional_edges("fetch_user_info", reset_memory_condition)
        .add_edge("reset_memory", END)
    ).compile(checkpointer=memory, debug=True)



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


from typing_extensions import TypedDict, Annotated, Dict, List

import config

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase

from langchain_core.tools import tool

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_gigachat import GigaChat

agent_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
#agent_llm = GigaChat(0
#            credentials=config.GIGA_CHAT_AUTH, 
#            model="GigaChat-Pro",
#            verify_ssl_certs=False,
#            temperature=0,
#            scope = config.GIGA_CHAT_SCOPE)

llm_query_gen = ChatOpenAI(model="gpt-4.1", temperature=0)
#llm_query_gen = ChatOpenAI(model="o4-mini")
#init_chat_model("gpt-4.1", model_provider="openai", temperature=0)
#llm = init_chat_model("gpt-4.1-nano", model_provider="openai", temperature=0)



db_7ya = SQLDatabase.from_uri("sqlite:///data/pricing/7ya.db")
db_7ya.name = "7ya"
db_vesna = SQLDatabase.from_uri("sqlite:///data/pricing/vesna.db")
db_vesna.name = "vesna"
db_andersen = SQLDatabase.from_uri("sqlite:///data/pricing/andersen.db")
db_andersen.name = "andersen"


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    messages: List[Dict[str, str]]

system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Always include prices, number of rooms and sizes into response.
Include into query only flats where price value is defined.
Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question. Always include: price_value, rooms, area_total, renovation.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

IMPORTANT: use only following fields for WHERE clause: price_value, rooms, area_total, renovation. Do not use other fields!!!
Use for query only fields that matches with user request. Do not extend query for other fields.
Field renovation can be one of: 'черновая отделка' or 'под ключ'.
Include results as folliwing: {return_condition}



Only use the following tables:
{table_info}
"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def create_flat_info_retriever(complex_id: str):
    complex_id = complex_id

    if complex_id == "vesna":
        db = db_vesna
    elif complex_id == "7ya":
        db = db_7ya
    elif complex_id == "andersen":
        db = db_andersen

    def write_query(state: State):
        """Generate SQL query to fetch information."""
        if db.name == "vesna":
            top_k = 3
            return_condition = "\nInclude only 3 cheapest flats."
            #return_condition = ("\nWrite a SQLite query to retrieve three records with cherapest flats:\n"
            #    "1. The cheapest entry.\n"
            #    "2. The most expensive entry.\n"
            #    "Combine the results using UNION ALL so that both rows are returned together.\n"
            #    "**Important:** Ensure each part of the UNION uses a subquery or appropriate SQLite syntax, since each SELECT uses ORDER BY with LIMIT. Use aliases for any subqueries as needed. Provide the final SQL query only, no explanations.")

        else:
            top_k = 2
            return_condition = ("\nInclude (1 cheapest flat and 1 most expensive flat with renovation == 'черновая отделка') and (1 cheapest flat and 1 most expensive flat with renovation == 'под ключ') ."
                "Combine the results using UNION ALL so that both rows are returned together.\n"
                "**Important:** Ensure each part of the UNION uses a subquery or appropriate SQLite syntax, since each SELECT uses ORDER BY with LIMIT. Use aliases for any subqueries as needed. Provide the final SQL query only, no explanations.\n"
                "Example properly formatted query with union:\n"
                "  SELECT internal_id, price_value, rooms\n"
                "    FROM (\n"
                "    SELECT internal_id, price_value, rooms\n"
                "        FROM Properties\n"
                "        ORDER BY price_value ASC\n"
                "        LIMIT 1\n"
                "    ) AS cheapest\n"
                "    UNION ALL\n"
                "    SELECT internal_id, price_value, rooms\n"
                "    FROM (\n"
                "        SELECT internal_id, price_value, rooms\n"
                "        FROM Properties\n"
                "        ORDER BY price_value DESC\n"
                "        LIMIT 1\n"
                "    ) AS most_expensive;"
                )
            #return_condition = ("\nWrite a SQLite query to retrieve two records:\n"
            #    "1. The cheapest flat.\n"
            #    "2. The most expensive flat.\n"
            #    "Combine the results using UNION ALL so that both rows are returned together.\n"
            #    "**Important:** Ensure each part of the UNION uses a subquery or appropriate SQLite syntax, since each SELECT uses ORDER BY with LIMIT. Use aliases for any subqueries as needed. Provide the final SQL query only, no explanations.")

        prompt = query_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": top_k,
                "table_info": db.get_table_info(),
                "input": state["question"],
                "return_condition": return_condition
            }
        )
        structured_llm = llm_query_gen.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        return {"query": result["query"]}


    def execute_query(state: State):
        """Execute SQL query."""
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        return {"result": execute_query_tool.invoke(state["query"])}

    def generate_answer(state: State):
        """Answer question using retrieved information as context."""
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n"
            "If result is empty inform user that there are no records meeting given criteria.\n"
            "Respond with list of flats satisfying criteria\n"
            "Include into response all fields, except technical\n"
            "Include into result price_value, rooms, area_total, renovation\n"
            "Do not include into response any technical fields (for example:ID).\n\n"
            f'Question: {state["question"]}\n'
            f'SQL Query: {state["query"]}\n'
            f'SQL Result: {state["result"]}'
        )
        result = agent_llm.invoke(prompt)
        answer = result.content
        return {"result": answer, "messages": [{"role": "assistant", "content": answer}]}

    flat_info_retriever = (
        StateGraph(State)
        .add_sequence([write_query, execute_query, generate_answer])
        .add_edge(START, "write_query")
    ).compile(
        name=f"{complex_id}_flat_info_retriever",
        debug=config.DEBUG_WORKFLOW,
    )

    return flat_info_retriever


def get_retrieval_tool(complex_id: str):
    flat_info_retriever = create_flat_info_retriever(complex_id)

    @tool
    def retrieve_flat_info(user_question: str):
        """Возвращает информацию по квартирам в определённом жилом комплексе (ЖК).
    Returns information of apartments in the residential complex by id.

    Args:
    user_question: the question user is interested to get information about."""
        response = flat_info_retriever.invoke({"question": user_question})
        answer = response.get("result") if isinstance(response, dict) else response
        return answer

    return retrieve_flat_info

def get_retrieval_agent(complex_id: str):
    retrieval_tool = get_retrieval_tool(complex_id)
    prompt = (
            f"You are an agent retrieving information about prices, sizing and other information aboutn flats available within building complex {complex_id}.\n\n"
            "INSTRUCTIONS:\n"
            f"- Assist ONLY with tasks related to retrieval information about building complex {complex_id}\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond with list of flats returned by your tools\n"
            "- Keep maximum information from all returned records\n"
            "- Include into result price_value, rooms, area_total, renovation\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        )

    return create_react_agent(
        model=agent_llm,
        tools=[retrieval_tool],# search_kb],
        prompt=prompt,
        name=f"{complex_id}_flat_info_retriever",
        debug=config.DEBUG_WORKFLOW,
    )

if __name__ == "__main__":
    import os
    os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    agent = get_retrieval_agent("7ya")
    result = agent.invoke({"question": "КАкие есть двушки в ЖК 7я?"})
    print(result)


    from agents.tools.supervisor_tools import create_pricing_handoff_tool
    from langgraph_supervisor import create_supervisor

    ho_vesna = create_pricing_handoff_tool("vesna_flat_info_retriever")
    db_vesna = create_flat_info_retriever("vesna")

    ho_tools = [
        ho_vesna,
    ]
    #with open("prompts/working_prompt_super.txt", encoding="utf-8") as f:
    #    prompt_txt = f.read()
    prompt_txt = """You are a supervisor managing a few agents agents
- agents retrieving information on flats within building complexes. Assign ONLY tasks related to retrieving information on flats within building complexes to this agent.
Assign work to one agent at a time, do not call agents in parallel.
Do not do any work yourself."""
    supervisor_agent = create_supervisor(
        model=init_chat_model("openai:gpt-4.1-mini"),
        agents=[db_vesna],
        prompt=prompt_txt,
        tools=ho_tools,
        add_handoff_back_messages=False,
        output_mode="full_history",
    ).compile(name="supervisor")#, debug = True)
    result = supervisor_agent.invoke({"question": "Мне нужны все квартиры стоимостью до 10 млн и площадью от 40 кв.м."})

    #flat_info_retriever = create_flat_info_retriever("vesna")
    #result = flat_info_retriever.invoke({"question": "Мне нужны все квартиры стоимостью до 10 млн и площадью от 40 кв.м."})
    #retrieve_flat_info = create_flat_info_retriever("vesna")
    #result = retrieve_flat_info("Мне нужны все квартиры стоимостью до 10 млн и площадью от 40 кв.м.")
    print(result)
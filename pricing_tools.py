from typing_extensions import TypedDict, Annotated

import config

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase

from langchain_core.tools import tool

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

#llm = init_chat_model("gpt-4.1-mini", model_provider="openai")
llm_query_gen = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm = init_chat_model("gpt-4.1-nano", model_provider="openai", temperature=0)

system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Always include prices, number of rooms, sizes and renovation status into response.
Do NOT include into result flats where prices OR number of rooms OR sizes OR  renovation status NOT defined.
Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.
Do not use building_name field for query information.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Never use the following fields for the query: category, property_type, building_state.
Field renovation can be one of: 'черновая отделка' or 'под ключ'.
Include results as following: {return_condition}



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


def write_query(state: State, db: SQLDatabase = db_vesna):
    """Generate SQL query to fetch information."""
    if db.name == "vesna":
        top_k = 3
        return_condition = "\nInclude only 3 cheapest flats."
    else:
        top_k = 2
        return_condition = ("\nInclude 1 cheapest flat and 1 most expensive flat."
            "Combine the results using UNION ALL so that both rows are returned together.\n"
            "**Important:** Ensure each part of the UNION uses a subquery or appropriate SQLite syntax, since each SELECT uses ORDER BY with LIMIT. Use aliases for any subqueries as needed. Provide the final SQL query only, no explanations.\b"
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
            "    ) AS most_expensive;")
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
    return result["query"] #{"query": result["query"]}

def execute_query(state: State, db: SQLDatabase = db_vesna):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return execute_query_tool.invoke(state["query"]) #{"result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question. Inlcude maximum availble information into answer\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return response.content #{"answer": response.content}

@tool
def get_flats_info_for_complex(complex_id: str, user_question: str) -> str:
    """Возвращает информацию по квартирам в определённом жилом комплексе (ЖК).
Этот инструмент не возвращает общую информацию по жилым комплексам"
Returns information of apartments in the residential complex by id./
Ths instrumet cannot provide common information about the building complexes

Args:
    complex_id: id of the complex, Available values: vesna, 7ya, andersen.
    user_question: the question user is interested to get information about."""
    if complex_id == "vesna":
        db = db_vesna
    elif complex_id == "7ya":
        db = db_7ya
    elif complex_id == "andersen":
        db = db_andersen
    else:
        raise ValueError(f"Unknown complex id: {complex_id}")
    
    state: State = {"question": user_question}
    state["query"] = write_query(state, db)
    state["result"] = execute_query(state, db)
    state["answer"] = generate_answer(state)

    return state["answer"]


#from langgraph.graph import START, StateGraph

#graph_builder = StateGraph(State).add_sequence(
#    [write_query, execute_query, generate_answer]
#)
#graph_builder.add_edge(START, "write_query")
#graph = graph_builder.compile()


#query = write_query({"question": "Мне нужны все квартиры стоимостью до 10 млн и площадью от 40 кв.м."})

#print(execute_query(query))
#for step in graph.stream(
#    {"question": "Хочу купить квартиру стоимостью до 10 млн и площадью от 40 кв.м."}, stream_mode="updates"
#):
#    print(step)
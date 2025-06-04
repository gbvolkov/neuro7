from typing_extensions import TypedDict, Annotated, Dict, List

import config

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

db_7ya = SQLDatabase.from_uri("sqlite:///data/pricing/7ya.db")
db_7ya.name = "7ya"
db_vesna = SQLDatabase.from_uri("sqlite:///data/pricing/vesna.db")
db_vesna.name = "vesna"

system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

"**Important:** In case UNION is necessary, ensure each part of the UNION uses a subquery or appropriate SQLite syntax, since each SELECT uses ORDER BY with LIMIT. Use aliases for any subqueries as needed. Provide the final SQL query only, no explanations.\b"
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

Always include prices, number of rooms and sizes into response.
Include into query only flats where price value is defined.
Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

IMPORTANT: use only following fields for WHERE clause: price_value, rooms, area_total, renovation. Do not use other fields!!!
Use for query only fields that matches with user request. Do not extend query for other fields.
Field renovation can be one of: 'черновая отделка' or 'под ключ'.
Return only: {return_condition}



Only use the following tables:
{table_info}
"""

system_message = """
Given an input question, create a syntactically correct {dialect} WHERE clause for query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

IMPORTANT: return only WHERE clause!

Include into query only flats where price value is defined.
Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

IMPORTANT: use only following fields for WHERE clause: price_value, rooms, area_total, renovation. Do not use other fields!!!
Field renovation can be one of: 'черновая отделка' or 'под ключ'.

Only use the following tables:
{table_info}
"""

llm_query_gen = ChatOpenAI(model="gpt-4.1", temperature=0)

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)



def write_query(db, question: str):
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
        return_condition = ("\nInclude 1 cheapest flat and 1 most expensive flat."
            "Combine the results using UNION ALL so that both rows are returned together.\n"
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
            "input": question
        }
    )
    structured_llm = llm_query_gen.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return result["query"]

query = write_query(db = db_vesna, question = "Уточнить диапазон цен на квартиры в жилом комплексе Весна с ценой в районе 15 млн")

print(query)
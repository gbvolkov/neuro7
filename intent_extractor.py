# intent_extractor.py
from typing import Optional
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Literal, Annotated, Optional
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate

from state import State, CustomerCtx         # keep your existing TypedDicts


# --- schema --------------------------------------------------------------
class FlatFilters(TypedDict, total=False):
    rooms:       Annotated[Optional[int],   "Number of rooms"]
    area_min:    Annotated[Optional[float], "m² lower bound"]
    area_max:    Annotated[Optional[float], "m² upper bound"]
    price_min:   Annotated[Optional[int],   "Currency lower"]
    price_max:   Annotated[Optional[int],   "Currency upper"]

class CustomerCtx(TypedDict):
    last_question:       str
    complex:             Optional[Literal["vesna", "andersen", "7ya"]]
    flat_filters:        FlatFilters
    call_requested:      bool
    time_slot_requested: Optional[str]
    time_slot_agreed:    Optional[str]
    chat_summary:        Annotated[Optional[str], "Summary of a chat with client. Shall include information about client, it's family, reason for purchasing flat, wshed building complex, wished financial conditions]"]

# --- prompt --------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a JSON formatter.  Fill the schema and output *only* JSON."
        "The 'complex' field MUST be one of: 'vesna', 'andersen', '7ya' (lower-case)."
        "If the user didn't mention a known complex, output null."
    ),
    ("human", "Conversation so far:\n{history}\n\nUser message:\n{user_msg}")
])

# --- model with structured output ---------------------------------------
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
extract_ctx: Runnable = prompt | llm.with_structured_output(CustomerCtx)

def update_customer_ctx(state: State) -> State:
    """LLM-powered slot filling using structured output."""
    history_text = "\n".join(
        m.content[0]["text"] for m in state["messages"][-6:-1]
        if m.type == "human"
    )
    user_msg = state["messages"][-1].content[0]["text"]

    raw_ctx: CustomerCtx = extract_ctx.invoke({
        "history": history_text,
        "user_msg": user_msg
    })

    merged_ctx = {**state.get("customer_ctx", {}), **raw_ctx}
    return {"customer_ctx": merged_ctx}

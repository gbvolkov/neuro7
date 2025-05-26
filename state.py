from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages, Messages


def add_messages_no_img(msgs1: Messages, msgs2: Messages) -> Messages:
    # Need to clean up all user messages excepting the last message with type "human" (after which can follow messages with other types)
    msgs = [msg for msg in msgs1 if msg.type == "human"][::-1]
    if len(msgs) > 1:
        msg = msgs[1]
        cleaned_content = [content for content in msg.content if content.get("type", "") != "image_url"]
        msg.content = cleaned_content

    return add_messages(msgs1, msgs2)

class UserInfo(TypedDict):
    user_name: str
    phone_number: str
    purpose: str
    interest: str

# state.py  (new file or extend the existing State TypedDict)

from typing_extensions import TypedDict, Literal, NotRequired

class FlatFilters(TypedDict, total=False):
    rooms: NotRequired[int]          # how many flats / rooms are requested
    area_min: NotRequired[float]
    area_max: NotRequired[float]
    price_min: NotRequired[int]
    price_max: NotRequired[int]

class CustomerCtx(TypedDict):
    last_question: str
    complex: str | None
    flat_filters: FlatFilters
    call_requested: bool
    time_slot_requested: str | None   # free-form, keep it text – parsing later
    time_slot_agreed: str | None      # filled only after contact_agent responds


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages_no_img]
    #customer_ctx: CustomerCtx
    user_info: UserInfo
from langchain_core.tools import tool
from utils import sub_dict
from datetime import datetime
import json

complexes = json.loads(open("data/residential_complexes.json", "r", encoding="utf-8").read())
complexes_idx = {rec["id"]: rec for rec in complexes}

@tool
def get_list_of_complexes() -> list[dict]:
    """Возвращает список жилых комплексов (ЖК), доступных к продаже.
Returns list of residential complexes, available for sale."""
    return sub_dict(complexes, ["id", "name", "district", "ready_date", "number_of_houses", "level"])

@tool
def get_developer_info() -> dict:
    """Возвращает информацию о застройщике.
Returns information of the developer."""
    return {"name": "ГК Новый Дом", 
            "address": ": ул. Жигура, 26 (ТЦ «Семёрочка», 2 этаж)",
            "working_hours": "пн-чт: 10:00 – 19:00; пт: 10:00 – 17:00; сб: 10:00 – 16:00; вс: выходной",
            "completed_complexes": ["ЖК «Изумрудный» ул. Майора Филипова", 
                          "ЖК «Антарес» ул. Кирова 33", 
                          "Поселок-парк «Весна» ул. Старцева 55, 57", 
                          "ЖК «Современник» Можжевеловая 18", 
                          "ЖК «АЛЕКСАНДРИТ» Жигура 12а", 
                          "Жилой дом на Тухачесвкого 30", 
                          "ЖК Семерочка ул. Жируга 26", 
                          "ТЦ Семерочка ул. Жируга 26", 
                          "Бизнес центр Seven ул. Жируга 26а"]}

@tool
def get_complex_info(complex_id: str, list_of_fields: list[str]) -> dict:
    """Возвращает информацию по определённому жилому комплексу (ЖК).
Returns information of the residential complex by id.

Args:
    complex_id: id of the complex
    list_of_fields: list of fields to return. Available fields: general_info, pricing, features, financial_conditions, managers_info"""

    return sub_dict([complexes_idx[complex_id]], list_of_fields)[0]


@tool
def agree_call(requested_time_slot: str) -> dict:
    """Возвращает предложение по времени созвона с менеджером.

Args:
    requested_time: период, в который клиент хочет созвониться. Может быть одним из: morning (сегодня до полудня), evening (сегодня после полудня), tomorrow (завтра), any (любое время)"""
    
    now = datetime.now().hour
    if now < 10:
        return {"time_slot": "morning"}
    elif now < 15:
        return {"time_slot": "evening"}
    else:
        return {"time_slot": "tomorrow"}

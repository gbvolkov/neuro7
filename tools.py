from langchain_core.tools import tool

from utils import sub_dict
from datetime import datetime
import json

complexes = json.loads(open("data/residential_complexes.json", "r", encoding="utf-8").read())
complexes_idx = {rec["id"]: rec for rec in complexes}

@tool
def get_list_of_complexes() -> list[dict]:
    """Возвращает список жилых комплексов (ЖК), доступных к продаже.
Список содержит: id комплекса, название, альтернативное название, район, год постройки, количество домов, уровень комфорта
Returns list of residential complexes, available for sale.
List contains: complex id, name, alternative name, district, year of construction, number of houses, comfort level"""
    return sub_dict(complexes, ["id", "name", "alternative_name", "district", "ready_date", "number_of_houses", "comfort_level"])

@tool
def get_developer_info() -> dict:
    """Возвращает информацию о застройщике.
ВАЖНО: Этот инструмент не возвращает информацию о жилых комплексах!!!
Returns information of the developer.
IMPORTANT: This tool does not contain information on building complexes!!!"""
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
    """Возвращает расширенную информацию по определённому жилому комплексу (ЖК).
Returns extended information of the residential complex by id.

Args:
    complex_id: id of the complex. Can be one of the following values: vesna, 7ya, andersen. 
    list_of_fields: list of fields to return. Available fields: name, alternative_name, district, ready_date, number_of_houses, comfort_level, general_info, features, financial_conditions, managers_info"""
    try:
        found_complex = complexes_idx[complex_id]
    except Exception:
        found_complex = {} #complexes_idx['vesna']
    return sub_dict([found_complex], list_of_fields)[0]


@tool
def agree_call(requested_time_slot: str = None) -> dict:
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


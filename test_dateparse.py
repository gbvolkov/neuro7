
import dateparser  
import datetime
from zoneinfo import ZoneInfo

def pase_date(text):

    tz = tz = ZoneInfo("Europe/Moscow")
    user_dt = datetime.datetime.now()
    mgr_now = user_dt.astimezone(tz)

    return dateparser.parse(
        text,
        languages=["ru"],
        settings={
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": mgr_now.replace(tzinfo=None),  # naive for dateparser
            "TIMEZONE": str(tz),
            "RETURN_AS_TIMEZONE_AWARE": False               # we add tz below
        },
    )


tests = [
    "Ну можно завтра часика в два.",
    "Да чем быстрее, тем лучше",
    "Ну давай вечерком",
    "давай в пятницу утром",
    "следующий понедельник в 9",
    "послезавтра часа в три",
]

for test in tests:
    print(f"{test!r:35} →  {pase_date(test)}")
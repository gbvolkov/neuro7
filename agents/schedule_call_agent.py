from __future__ import annotations
from datetime import datetime, date, time, timedelta
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import json, re
from pathlib import Path
import dateparser                                  # ★ new dependency

# ─────── DATA CLASSES ────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class Config:
    tz: ZoneInfo
    weekly: dict[int, tuple[time, time] | None]
    tod_aliases: dict[str, time]
    urgent_re: re.Pattern
    holidays: set[date]
    working_weekends: set[date]


# ─────── FACTORY ─────────────────────────────────────────────────────────────
def scheduler_factory(config_path: str | Path):
    """
    Build a ready-to-use  schedule_call(user_dt, desired_phrase)  function
    from the JSON *config_path*.
    """
    cfg = _load_config(config_path)

    # ---------- helpers bound to *cfg* ----------------
    def get_day_schedule(d: date) -> tuple[time, time] | None:
        if d in cfg.holidays:
            return None
        wd = d.weekday()
        if d in cfg.working_weekends and wd == 6:   # working Sunday
            return cfg.weekly.get(0)                # treat like Monday
        return cfg.weekly.get(wd)

    def parse_desired(text: str, user_dt: datetime) -> datetime:
        """Convert free-form Russian phrase → candidate datetime (manager TZ)."""
        text = text.lower().strip()
        mgr_now = user_dt.astimezone(cfg.tz)

        # 1) urgent words → “now + 20 min”
        if cfg.urgent_re.search(text):
            return mgr_now + timedelta(minutes=20)

        # 2) try dateparser (handles “завтра часика в два”, “в следующий понедельник” …)
        dt = dateparser.parse(
            text,
            languages=["ru"],
            settings={
                "PREFER_DATES_FROM": "future",
                "RELATIVE_BASE": mgr_now.replace(tzinfo=None),  # naive for dateparser
                "TIMEZONE": str(cfg.tz),
                "RETURN_AS_TIMEZONE_AWARE": False               # we add tz below
            },
        )
        if dt:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=cfg.tz)
            return dt

        # 3) fallback: vague word → alias time
        for word, alias_time in cfg.tod_aliases.items():
            if word in text:
                candidate = datetime.combine(mgr_now.date(), alias_time, tzinfo=cfg.tz)
                if candidate < mgr_now:              # if already passed today → tomorrow
                    candidate += timedelta(days=1)
                return candidate

        # 4) nowhere else matched → default to “regular” (first opening slot today / next)
        first_open = get_day_schedule(mgr_now.date())
        base_time = first_open[0] if first_open else time(10, 0)
        cand = datetime.combine(mgr_now.date(), base_time, tzinfo=cfg.tz)
        if cand < mgr_now:
            cand += timedelta(days=1)
        return cand

    def next_valid(dt: datetime) -> datetime:
        while True:
            sched = get_day_schedule(dt.date())
            if sched is None:
                dt = datetime.combine(dt.date() + timedelta(days=1), time(0), tzinfo=cfg.tz)
                continue
            start_t, end_t = sched
            if dt.time() < start_t:
                dt = dt.replace(hour=start_t.hour, minute=start_t.minute, second=0, microsecond=0)
                return dt
            if start_t <= dt.time() <= end_t:
                return dt
            dt = datetime.combine(dt.date() + timedelta(days=1), time(0), tzinfo=cfg.tz)

    # ---------- public API ----------------
    def schedule_call_complex(user_dt: datetime, desired_phrase: str) -> datetime:
        """
        user_dt        – caller’s *aware* datetime
        desired_phrase – free Russian text (“Ну можно завтра часика в два” …)
        RETURNS        – first legal slot (aware, manager TZ)
        """
        first_guess = parse_desired(desired_phrase, user_dt)
        return next_valid(first_guess)

    def schedule_call(user_dt: datetime, desired_dt: datetime) -> datetime:
        """
        user_dt        – caller’s current datetime
        desired_dt     – datetiem desired for call
        RETURNS        – first legal slot (aware, manager TZ)
        """
        return next_valid(desired_dt)

    return schedule_call


# ─────── CONFIG LOADER ──────────────────────────────────────────────────────
def _load_config(path: str | Path) -> Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    tz = ZoneInfo(raw["timezone"])

    weekly = {
        int(k): None if v is None else tuple(time.fromisoformat(t) for t in v)
        for k, v in raw["weekly_schedule"].items()
    }

    tod_aliases = {k: time.fromisoformat(v) for k, v in raw["timeofday_aliases"].items()}

    # compile one big regex for “urgent” phrases (case-insensitive)
    urgent_re = re.compile("|".join(map(re.escape, raw["urgent_patterns"])), re.I)

    cal = raw["calendar"]
    holidays = {date.fromisoformat(d) for d in cal.get("holidays", [])}
    working_weekends = {date.fromisoformat(d) for d in cal.get("working_weekends", [])}

    return Config(tz, weekly, tod_aliases, urgent_re, holidays, working_weekends)

from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langgraph.types import Send

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

import config



agent_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
schedule_call = scheduler_factory("manager_config.json")

schedule_call_agent = create_react_agent(
    model=agent_llm,
    tools=[schedule_call],
    prompt=(
        "Вы — вспомогательный агент. На вход получаете запрос пользователя на звонок с менеджером\n"
        "Вы должны вернуть дату и время, когда менеджер будет звонить клиенту.\n"
    ),
    name="schedule_call_agent",
    debug=config.DEBUG_WORKFLOW,
)

# ─────── DEMO ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sc = scheduler_factory("manager_config.json")

    msk_now = datetime(2025, 6, 5, 12, 15, tzinfo=ZoneInfo("Europe/Moscow"))

    tests = [
        "Ну можно завтра часика в два.",
        "Да чем быстрее, тем лучше",
        "Ну давай вечерком",
        "давай в пятницу утром",
        "следующий понедельник в 9",
        "послезавтра",
    ]

    for phrase in tests:
        print(f"{phrase!r:35} →  {sc(msk_now, phrase)}")

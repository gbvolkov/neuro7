from __future__ import annotations
from datetime import datetime, date, time, timedelta
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import json
from pathlib import Path

# ─────── DATA CLASSES ────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class Config:
    tz: ZoneInfo
    weekly: dict[int, tuple[time, time] | None]        # 0-6 → (start, end) / None
    tod_defaults: dict[str, time]                      # "morning"/"evening"/"regular"
    holidays: set[date]
    working_weekends: set[date]


# ──────── FACTORY ────────────────────────────────────────────────────────────
def scheduler_factory(config_path: str | Path):
    """
    Load JSON at *config_path* and return a ready-to-use
        schedule_call(user_dt: datetime, desired: str) -> datetime
    """
    cfg = _load_config(config_path)

    # ---------- helpers bound to *cfg* ----------------
    def get_day_schedule(d: date) -> tuple[time, time] | None:
        if d in cfg.holidays:
            return None
        wd = d.weekday()

        # working weekends override Sunday/Saturday closures
        if d in cfg.working_weekends and wd == 6:
            return cfg.weekly.get(0)
        return cfg.weekly.get(wd)

    def parse_desired(desired: str, user_dt: datetime) -> datetime:
        desired = desired.lower().strip()
        mgr_now = user_dt.astimezone(cfg.tz)
        target_date = mgr_now.date()

        # recognise "tomorrow"
        if "завтра" in desired:
            target_date += timedelta(days=1)

        # "now" (special shortcut)
        if desired == "сейчас":
            return mgr_now + timedelta(minutes=20)

        # bucket → default time
        if "утро" in desired:
            base_time = cfg.tod_defaults["morning"]
        elif "вечер" in desired:
            base_time = cfg.tod_defaults["evening"]
        else:
            base_time = cfg.tod_defaults["regular"]

        return datetime.combine(target_date, base_time, tzinfo=cfg.tz)

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

    # ---------- the function exposed to callers ----------------
    def schedule_call(user_dt: datetime, desired_when: str) -> datetime:
        """
        user_dt       – tz-aware current time supplied by caller
        desired_when  – 'now', 'today evening', 'tomorrow', …
        RETURNS       – tz-aware manager-local datetime
        """
        first_guess = parse_desired(desired_when, user_dt)
        return next_valid(first_guess)

    return schedule_call


# ──────── CONFIG LOADER ──────────────────────────────────────────────────────
def _load_config(path: str | Path) -> Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    tz = ZoneInfo(raw["timezone"])

    weekly = {
        int(k): None if v is None else tuple(time.fromisoformat(t) for t in v)
        for k, v in raw["weekly_schedule"].items()
    }
    tod_defaults = {k: time.fromisoformat(v) for k, v in raw["timeofday_defaults"].items()}

    holidays = {date.fromisoformat(d) for d in raw["calendar"].get("holidays", [])}
    working_weekends = {date.fromisoformat(d) for d in raw["calendar"].get("working_weekends", [])}

    return Config(tz, weekly, tod_defaults, holidays, working_weekends)


# ──────── DEMO ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Build a scheduler from the example JSON  (adjust path as needed)
    schedule_call = scheduler_factory("manager_config.json")

    # Suppose the user is in Moscow and it is 2025-06-06 12:15 MSK (UTC+3)
    from zoneinfo import ZoneInfo
    moscow_now = datetime(2025, 6, 5, 10, 35, tzinfo=ZoneInfo("Europe/Moscow"))

    # Russia-wide state holidays (mock)
    calendar = {
        "holidays": ["2025-06-12", "2025-06-13", "2025-11-04"],
        "working_weekends": ["2025-02-22", "2025-06-08", "2025-11-01"]   # Sat & Sun flagged as workdays
    }

    # 1️⃣ “now”                       → should be today 20 min later if still open
    print("Call @", schedule_call(moscow_now, "сейчас"))

    # 2️⃣ “today evening”             → today 17:00 if open, else next slot
    print("Call @", schedule_call(moscow_now, "завтра вечером"))

    # 3️⃣ “tomorrow morning”          → tomorrow 10:00 or later after weekend/holiday
    print("Call @", schedule_call(moscow_now, "завтра утром"))

    # 3️⃣ “tomorrow morning”          → tomorrow 10:00 or later after weekend/holiday
    print("Call @", schedule_call(moscow_now, "завтра вечером"))
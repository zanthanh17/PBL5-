# app/utils/att_logic.py
import os, datetime, pytz
from sqlalchemy.orm import Session
from sqlalchemy import text

DEBOUNCE_SEC = int(os.getenv("DEBOUNCE_SEC", "30"))   # giây
MIN_SESSION_MIN = int(os.getenv("MIN_SESSION_MIN", "10"))
WORKDAY_TZ = os.getenv("WORKDAY_TZ", "Asia/Ho_Chi_Minh")

def vn_now():
    tz = pytz.timezone(WORKDAY_TZ)
    return datetime.datetime.now(tz)

def day_range(dt=None):
    tz = pytz.timezone(WORKDAY_TZ)
    if dt is None: dt = vn_now()
    start = tz.localize(datetime.datetime(dt.year, dt.month, dt.day, 0,0,0))
    end   = start + datetime.timedelta(days=1)
    return start, end

def last_event_today(db: Session, emp_id: str):
    start, end = day_range()
    sql = text("""
      SELECT id, ts, type, decision, score
      FROM attendance
      WHERE emp_id = :emp_id AND decision='accepted' AND ts >= :start AND ts < :end
      ORDER BY ts DESC LIMIT 1
    """)
    return db.execute(sql, {"emp_id": emp_id, "start": start, "end": end}).mappings().first()

def last_event_any(db: Session, emp_id: str):
    sql = text("""
      SELECT id, ts, type
      FROM attendance
      WHERE emp_id=:emp_id AND decision='accepted'
      ORDER BY ts DESC LIMIT 1
    """)
    return db.execute(sql, {"emp_id": emp_id}).mappings().first()

def should_debounce(last_ts, now_dt):
    if not last_ts: return False
    delta = (now_dt - last_ts).total_seconds()
    return delta < DEBOUNCE_SEC

def decide_type(db: Session, emp_id: str, now_dt):
    """
    Quy tắc:
     - Hôm nay chưa có gì → checkin
     - Gần nhất hôm nay là checkin:
         + nếu cách >= MIN_SESSION_MIN → checkout
         + nếu chưa đủ → vẫn checkin (nhưng debounce sẽ chặn ghi thêm)
     - Gần nhất là checkout → checkin
    """
    le = last_event_today(db, emp_id)
    if not le:
        return "checkin"
    last_type = le["type"]; last_ts = le["ts"]
    gap_min = (now_dt - last_ts).total_seconds() / 60.0
    if last_type == "checkin":
        return "checkout" if gap_min >= MIN_SESSION_MIN else "checkin"
    else:
        return "checkin"

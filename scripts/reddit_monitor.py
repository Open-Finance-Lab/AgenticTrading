"""
reddit_monitor.py — enhanced alerts + leaderboard with subreddit, top posts,
top authors, and earnings-call heuristics.
"""

from __future__ import annotations
import os
import re
import math
import json
import time
import asyncio
import sqlite3
from collections import defaultdict, deque
from statistics import mean, pstdev
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Set

import asyncpraw
from asyncpraw.models import Submission, Comment

# Optional sentiment (enable with env var and install transformers)
ENABLE_SENTIMENT = os.getenv("ENABLE_SENTIMENT", "0") == "1"
FINBERT = None
if ENABLE_SENTIMENT:
    try:
        from transformers import pipeline
        FINBERT = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception as e:
        print("Warning: sentiment enabled but transformers pipeline failed:", e)
        FINBERT = None

# --------------------------- Config ---------------------------
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
USER_AGENT = os.getenv("REDDIT_USER_AGENT", "MomentumMonitor/2.0 by /u/yourusername")
SUBREDDITS = os.getenv("REDDIT_SUBS", "wallstreetbets,stocks,options,investing").split(",")
BUCKET_SECONDS = int(os.getenv("BUCKET_SECONDS", "300"))
HISTORY_BUCKETS = int(os.getenv("HISTORY_BUCKETS", "288"))
ACC_Z_THRESHOLD = float(os.getenv("ACC_Z_THRESHOLD", "2.5"))
VEL_MIN = int(os.getenv("VEL_MIN", "5"))
ALERT_COOLDOWN_MIN = int(os.getenv("ALERT_COOLDOWN_MIN", "30"))
DB_PATH = os.getenv("DB_PATH", "data/momentum.db")
LEADERBOARD_INTERVAL = int(os.getenv("LEADERBOARD_INTERVAL", "15"))

ENABLE_KLEINBERG = os.getenv("ENABLE_KLEINBERG", "1") == "1"
ENABLE_HAWKES = os.getenv("ENABLE_HAWKES", "1") == "1"
ENABLE_NGRAM_THEMES = os.getenv("ENABLE_NGRAM_THEMES", "1") == "1"
WHO_TALKS_WEIGHT = float(os.getenv("WHO_TALKS_WEIGHT", "0.25"))

# --------------------------- Regex ---------------------------
CASHTAG = re.compile(r"\$([A-Z]{1,5})(?![A-Za-z])")
SYMBOL_WORD = re.compile(r"\b([A-Z]{2,5})\b")
WORD = re.compile(r"[A-Za-z$]{3,}")

# earnings / time detection
EARN_RE = re.compile(r"\b(earnings call|earnings|earnings results|conference call|quarterly results|eps|beat|miss|guidance|pre-?market|after-?hours|after hours|premarket|ah)\b", re.I)
DATE_RE = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}(?:,?\s+\d{4})?\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", re.I)
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b|\b\d{1,2}\s*(?:AM|PM|am|pm)\b")
BEAT_MISS_RE = re.compile(r"\b(beat|miss|beats|missed|beat expectations|miss expectations)\b", re.I)
PRE_AH_RE = re.compile(r"\b(pre-?market|after-?hours|after hours|premarket|ah)\b", re.I)

# --------------------------- Kleinberg & Hawkes (unchanged) ---------------------------
class KleinbergBurst:
    def __init__(self, s=2.0, gamma=1.0):
        self.s = s
        self.gamma = gamma
        self.state = 0
        self.cost_prev = 0.0
    def update(self, x, mean_rate):
        if mean_rate <= 0: mean_rate = 1e-6
        lambda0 = mean_rate
        lambda1 = self.s * mean_rate
        c0 = - (x * math.log(lambda0 + 1e-9) - lambda0)
        c1 = - (x * math.log(lambda1 + 1e-9) - lambda1) + self.gamma
        if self.cost_prev + c1 < self.cost_prev + c0:
            self.state = 1
            self.cost_prev += c1
        else:
            self.state = 0
            self.cost_prev += c0
        return self.state

class HawkesEstimator:
    def __init__(self, window=3600):
        self.window = window
        self.times = deque(maxlen=5000)
    def add(self, ts):
        self.times.append(ts)
    def branching_ratio(self, now):
        short = [t for t in self.times if now - t <= 900]
        long = [t for t in self.times if now - t <= self.window]
        rs = len(short) / 900.0 if short else 0.0
        rl = len(long) / float(self.window) if long else 1e-9
        return max(0.0, min(0.99, rs / rl - 1.0))

# --------------------------- Storage (added helpful queries) ---------------------------
class Store:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init()
    def _init(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS events(
                ts INTEGER, sub TEXT, kind TEXT, id TEXT, author TEXT, title TEXT, body TEXT, score INTEGER
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS alerts(ts INTEGER, entity TEXT, tier TEXT, payload TEXT)
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS authors(author TEXT PRIMARY KEY, influence REAL)
        """)
        self.conn.commit()
    def add_event(self, ts, sub, kind, _id, author, title, body, score):
        try:
            self.conn.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?)", (ts, sub, kind, _id, author, title, body, score))
        except Exception:
            pass
    def add_alert(self, ts, entity, tier, payload):
        try:
            self.conn.execute("INSERT INTO alerts VALUES(?,?,?,?)", (ts, entity, tier, payload))
        except Exception:
            pass
    def get_author_influence(self, author):
        row = self.conn.execute("SELECT influence FROM authors WHERE author=?", (author,)).fetchone()
        return row[0] if row else 0.0
    def update_author_influence(self, author, value, alpha=0.1):
        row = self.conn.execute("SELECT influence FROM authors WHERE author=?", (author,)).fetchone()
        if row:
            inf = (1-alpha)*row[0] + alpha*value
            self.conn.execute("UPDATE authors SET influence=? WHERE author=?", (inf, author))
        else:
            self.conn.execute("INSERT INTO authors(author,influence) VALUES(?,?)", (author, value))
    def commit(self):
        try:
            self.conn.commit()
        except Exception:
            pass

    # helpful queries for richer alerts
    def top_events_for(self, entity:str, limit:int=5) -> List[Tuple[str,str,int]]:
        q = self.conn.execute(
            "SELECT id, title, score FROM events WHERE (title LIKE ? OR body LIKE ?) ORDER BY rowid DESC LIMIT ?",
            (f"%{entity.replace('#:','')}%", f"%{entity.replace('#:','')}%", limit)
        )
        return q.fetchall()

    def top_authors_for(self, entity:str, limit:int=5) -> List[Tuple[str,int]]:
        q = self.conn.execute(
            "SELECT author, COUNT(*) as c FROM events WHERE (title LIKE ? OR body LIKE ?) AND author IS NOT NULL GROUP BY author ORDER BY c DESC LIMIT ?",
            (f"%{entity.replace('#:','')}%", f"%{entity.replace('#:','')}%", limit)
        )
        return q.fetchall()

    def events_for_entity(self, entity:str, limit:int=200):
        q = self.conn.execute(
            "SELECT ts, sub, kind, id, author, title, body, score FROM events WHERE (title LIKE ? OR body LIKE ?) ORDER BY rowid DESC LIMIT ?",
            (f"%{entity.replace('#:','')}%", f"%{entity.replace('#:','')}%", limit)
        )
        return q.fetchall()

# --------------------------- Notifier ---------------------------
class Notifier:
    async def send(self, payload: dict):
        # pretty print alert for human consumption
        print("ALERT:", json.dumps(payload, indent=2))

# --------------------------- Monitor ---------------------------
class MomentumMonitor:
    def __init__(self, symbols_path:str="us_symbols.txt", themes_path: str|None=None):
        self.symbols = self._load_symbols(symbols_path)
        self.themes = self._load_themes(themes_path)
        self.subs = [s.strip() for s in SUBREDDITS if s.strip()]
        self.bucket_epoch = self._now_bucket()
        self.counts = defaultdict(lambda: deque(maxlen=HISTORY_BUCKETS))
        self.authors = defaultdict(lambda: deque(maxlen=HISTORY_BUCKETS))
        self.subspread = defaultdict(lambda: deque(maxlen=HISTORY_BUCKETS))
        self.upvotes = defaultdict(lambda: deque(maxlen=HISTORY_BUCKETS))
        self.comments = defaultdict(lambda: deque(maxlen=HISTORY_BUCKETS))
        self.author_set_bucket = defaultdict(set)
        self.sub_set_bucket = defaultdict(set)
        self.author_influence = defaultdict(float)
        self.hawkes = defaultdict(HawkesEstimator)
        self.kleinberg = defaultdict(KleinbergBurst)
        self.ng_counts = defaultdict(lambda: deque(maxlen=HISTORY_BUCKETS))
        self.last_alert = {}
        self.store = Store(DB_PATH)
        self.notifier = Notifier()

    def _load_symbols(self, path:str) -> Set[str]:
        out=set()
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    for line in f:
                        s=line.strip().upper()
                        if s: out.add(s)
        except Exception:
            pass
        return out

    def _load_themes(self, path):
        defaults = {
            "short squeeze": ["short squeeze","gamma squeeze","shortsqueeze","gamma"],
            "rate cut": ["rate cut","fed cut","cut rates","pivot"],
            "earnings": ["earnings","eps","guidance","beat","miss","conference call"],
            "crypto": ["bitcoin","btc","ethereum","eth"]
        }
        if path and os.path.exists(path):
            try:
                with open(path,"r") as f:
                    return json.load(f)
            except Exception:
                return defaults
        return defaults

    def _now_bucket(self) -> int:
        return int(datetime.now(timezone.utc).timestamp() // BUCKET_SECONDS)

    def _ensure_bucket(self):
        nowb = self._now_bucket()
        if nowb != self.bucket_epoch:
            self.bucket_epoch = nowb
            for d in (self.counts,self.authors,self.subspread,self.upvotes,self.comments,self.ng_counts):
                for k in list(d.keys()):
                    d[k].append(0)
            self.author_set_bucket = defaultdict(set)
            self.sub_set_bucket = defaultdict(set)

    def _boot_entity(self, entity:str):
        if not self.counts[entity]:
            for d in (self.counts,self.authors,self.subspread,self.upvotes,self.comments):
                d[entity].extend([0]*(HISTORY_BUCKETS-1) + [0])

    def _extract_entities_and_ngrams(self, text:str):
        cashtags = {m.group(1) for m in CASHTAG.finditer(text)}
        words = {m.group(1) for m in SYMBOL_WORD.finditer(text)}
        tickers = {w for w in words if w in self.symbols}
        lower = text.lower()
        themes = set()
        for name, keys in self.themes.items():
            for k in keys:
                if k in lower:
                    themes.add(f"#:{name}")
                    break
        grams = [w.lower() for w in WORD.findall(text) if len(w)>=3]
        return (cashtags | tickers | themes), grams

    def _influence(self, author:str|None) -> float:
        if not author:
            return 0.0
        dbv = self.store.get_author_influence(author)
        return 0.5 * self.author_influence.get(author, 0.0) + 0.5 * dbv

    def _update_influence(self, author:str|None, score:int):
        if not author:
            return
        try:
            val = math.log1p(max(0, score))
        except Exception:
            val = 0.0
        prev = self.author_influence.get(author, 0.0)
        new = 0.9 * prev + 0.1 * val
        self.author_influence[author] = new
        self.store.update_author_influence(author, new)

    def _update_series(self, entity:str, author:str|None, sub:str|None, upvote_inc:int=0, comment_inc:int=0):
        self._boot_entity(entity)
        w = 1.0 + WHO_TALKS_WEIGHT * self._influence(author)
        self.counts[entity][-1] += w
        if author and author not in self.author_set_bucket[entity]:
            self.author_set_bucket[entity].add(author)
            self.authors[entity][-1] += 1
        if sub and sub not in self.sub_set_bucket[entity]:
            self.sub_set_bucket[entity].add(sub)
            self.subspread[entity][-1] += 1
        self.upvotes[entity][-1] += upvote_inc
        self.comments[entity][-1] += comment_inc
        if ENABLE_HAWKES:
            self.hawkes[entity].add(int(time.time()))

    def _update_ngram(self, gram:str):
        key = f"ng:{gram}"
        if not self.ng_counts[key]:
            self.ng_counts[key].extend([0]*(HISTORY_BUCKETS-1) + [0])
        self.ng_counts[key][-1] += 1

    def _series_z(self, series:deque) -> float:
        s = list(series)
        if len(s) < 6: return 0.0
        mu = mean(s[:-1]) if len(s) > 1 else 0.0
        sd = pstdev(s[:-1]) if len(s) > 2 else 1.0
        sd = sd if sd else 1.0
        return (s[-1] - mu) / sd

    def _accel_z(self, series:deque) -> float:
        s = list(series)
        if len(s) < 6: return 0.0
        diffs = [s[i] - s[i-1] for i in range(1, len(s))]
        return self._series_z(diffs)

    def _score(self, e:str):
        v = float(self.counts[e][-1]) if self.counts[e] else 0.0
        vz = self._series_z(self.counts[e])
        az = self._accel_z(self.counts[e])
        sz = self._series_z(self.subspread[e])
        ez = self._series_z(self.upvotes[e])
        uz = self._series_z(self.comments[e])
        authors_list = list(self.authors[e]) or [0.0]
        uniq = min(1.0, authors_list[-1] / max(1.0, mean(authors_list[-6:] or [1.0])))
        score = 0.35 * az + 0.2 * vz + 0.15 * sz + 0.2 * max(0.0, ez + 0.5 * uz) + 0.1 * uniq
        return score, v, az, vz, sz

    def _tier(self, score:float, az:float, burst:int, b_ratio:float):
        if score >= 4.5 and az >= 3.0 and (not ENABLE_KLEINBERG or burst) and (not ENABLE_HAWKES or b_ratio >= 0.3):
            return "FRENZY"
        if score >= 3.0 and az >= ACC_Z_THRESHOLD:
            return "SURGE"
        if score >= 2.0 and az >= 1.5:
            return "WATCH"
        return None

    # ----- extra context helpers -----
    def _top_subs_for(self, entity:str, limit:int=5) -> List[Tuple[str,int]]:
        # count subs from stored events (fast & persistent)
        rows = self.store.events_for_entity(entity, limit=500)
        counts = defaultdict(int)
        for ts, sub, kind, _id, author, title, body, score in rows:
            if sub:
                counts[sub] += 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:limit]

    def _top_authors_for(self, entity:str, limit:int=5):
        return self.store.top_authors_for(entity, limit=limit)

    def _extract_earnings_info(self, entity:str, limit:int=200) -> dict:
        rows = self.store.events_for_entity(entity, limit=limit)
        info = {"mentions": 0, "dates": {}, "times": {}, "pre_ah": 0, "beat": 0, "miss": 0, "examples": []}
        for ts, sub, kind, _id, author, title, body, score in rows:
            text = f"{title or ''}\n{body or ''}"
            if EARN_RE.search(text):
                info["mentions"] += 1
                # extract dates/times
                for m in DATE_RE.findall(text):
                    info["dates"][m] = info["dates"].get(m, 0) + 1
                for m in TIME_RE.findall(text):
                    info["times"][m] = info["times"].get(m, 0) + 1
                if PRE_AH_RE.search(text):
                    info["pre_ah"] += 1
                if BEAT_MISS_RE.search(text):
                    # count beat/miss words
                    if re.search(r"\bbeat", text, re.I):
                        info["beat"] += 1
                    if re.search(r"\bmiss", text, re.I):
                        info["miss"] += 1
                # collect a short example
                snippet = (title or body or "")[:200]
                info["examples"].append({"ts": ts, "sub": sub, "author": author, "snippet": snippet})
            if len(info["examples"]) >= 3:
                break
        return info

    # ---------- top threads ----------
    def _top_threads(self, entity:str, limit:int=3):
        rows = self.store.top_events_for(entity, limit=limit)
        out=[]
        for _id, title, score in rows:
            url = f"https://reddit.com/{_id}" if _id else ""
            out.append({"title": (title or "")[:120], "url": url, "score": score})
        return out

    # ---------- alerts ----------
    async def _send_alert(self, e:str, tier:str, score:float, v:float, az:float, vz:float, sz:float):
        top_threads = self._top_threads(e, limit=3)
        top_subs = self._top_subs_for(e, limit=5)
        top_authors = self._top_authors_for(e, limit=5)
        earnings = self._extract_earnings_info(e, limit=300) if ("earn" in e.lower() or "#:earnings" in e.lower() or any(k in e.lower() for k in ["earn", "eps"]) ) else {}
        sentiment = None
        if FINBERT and top_threads:
            try:
                titles = [t["title"] for t in top_threads]
                out = FINBERT(titles)
                # aggregate simple average label/score
                pos = sum(1 for r in out if r.get("label","").lower().startswith("pos")) if out else 0
                sentiment = {"n": len(out), "pos_count": pos, "samples": out}
            except Exception:
                sentiment = None

        payload = {
            "entity": e,
            "tier": tier,
            "score": round(score,2),
            "velocity": v,
            "accel_z": round(az,2),
            "vel_z": round(vz,2),
            "subspread_z": round(sz,2),
            "top_subreddits": top_subs,
            "top_authors": top_authors,
            "top_threads": top_threads,
            "earnings_info": earnings,
        }
        if sentiment:
            payload["sentiment"] = sentiment

        # only send if there is a meaningful context to show
        if top_threads or top_subs or top_authors or earnings.get("mentions",0) > 0:
            await self.notifier.send(payload)
            self.last_alert[e] = int(time.time())
            self.store.add_alert(int(time.time()), e, tier, json.dumps(payload))
        else:
            # debug: suppress alert without context (this prevents empty FRENZY alerts)
            print(f"[suppressed alert — no context] entity={e} score={score:.2f} v={v:.2f} az={az:.2f}")

    # ---------- detection ----------
    async def detect(self):
        now = int(time.time())
        for e in list(self.counts.keys()):
            if len(self.counts[e]) < 6:
                continue
            score, v, az, vz, sz = self._score(e)
            burst = 0
            if ENABLE_KLEINBERG:
                mean_rate = max(1e-6, mean(list(self.counts[e])[-12:]) / BUCKET_SECONDS)
                burst = self.kleinberg[e].update(self.counts[e][-1], mean_rate)
            b_ratio = 0.0
            if ENABLE_HAWKES:
                b_ratio = self.hawkes[e].branching_ratio(now)
            tier = self._tier(score, az, burst, b_ratio)
            if not tier:
                continue
            if v < VEL_MIN:
                continue
            last = self.last_alert.get(e, 0)
            if now - last < ALERT_COOLDOWN_MIN * 60:
                continue
            await self._send_alert(e, tier, score, v, az, vz, sz)
        self.store.commit()

    # ---------- leaderboard ----------
    def _print_leaderboard(self, top_n:int=10):
        self._ensure_bucket()
        rows=[]
        for e in list(self.counts.keys()):
            score, v, az, vz, sz = self._score(e)
            burst = self.kleinberg[e].update(v, mean(list(self.counts[e])[-12:]) / BUCKET_SECONDS if len(self.counts[e])>=1 else 0.0)
            b_ratio = self.hawkes[e].branching_ratio(int(time.time()))
            tier = self._tier(score, az, burst, b_ratio) or "-"
            rows.append((e, round(score,2), int(round(v)), round(az,2), int(self.subspread[e][-1] if self.subspread[e] else 0), tier))
        rows = sorted(rows, key=lambda r: r[1], reverse=True)[:top_n]
        print("\n===== Financial Momentum Leaderboard =====")
        print(f"{'Entity':<18} {'Score':<6} {'Vel':<4} {'AccZ':<6} {'Sub':<3} {'Tier':<7}")
        for e, sc, v, az, sub, tier in rows:
            # get small context preview
            top_subs = ", ".join(f"{s}({c})" for s,c in self._top_subs_for(e, limit=3)[:3]) or "-"
            print(f"{e:<18} {sc:<6} {v:<4} {az:<6} {sub:<3} {tier:<7}  top_subs: {top_subs}")
        print("========================================\n")

    # ---------- handlers ----------
    async def handle_submission(self, s: Submission):
        try:
            ts = int(getattr(s, "created_utc", time.time()))
            sub = s.subreddit.display_name if s.subreddit else None
            # author name resiliently
            author = None
            if getattr(s, "author", None):
                author = getattr(s.author, "name", None) or str(s.author)
            title = s.title or ""
            body = s.selftext or ""
            text = (title + " " + body)[:15000]
            entities, grams = self._extract_entities_and_ngrams(text)
            for e in entities:
                self._update_series(e, author, sub, upvote_inc=getattr(s,"score",0))
            if ENABLE_NGRAM_THEMES:
                for g in grams:
                    self._update_ngram(g)
            self._update_influence(author, getattr(s,"score",0))
            self.store.add_event(ts, sub, "submission", getattr(s,"id",""), author, title[:500], body[:1000], getattr(s,"score",0))
        except Exception as exc:
            print("handle_submission err:", exc)

    async def handle_comment(self, c: Comment):
        try:
            ts = int(getattr(c, "created_utc", time.time()))
            sub = c.subreddit.display_name if c.subreddit else None
            author = None
            if getattr(c, "author", None):
                author = getattr(c.author, "name", None) or str(c.author)
            body = c.body or ""
            entities, grams = self._extract_entities_and_ngrams(body)
            for e in entities:
                self._update_series(e, author, sub, comment_inc=1)
            if ENABLE_NGRAM_THEMES:
                for g in grams:
                    self._update_ngram(g)
            self._update_influence(author, getattr(c,"score",0))
            self.store.add_event(ts, sub, "comment", getattr(c,"id",""), author, "", body[:1000], getattr(c,"score",0))
        except Exception as exc:
            print("handle_comment err:", exc)

    # ---------- streaming consumers ----------
    async def _consume_submissions(self, subreddit_obj):
        async for subm in subreddit_obj.stream.submissions(skip_existing=True):
            self._ensure_bucket()
            await self.handle_submission(subm)
            await self.detect()

    async def _consume_comments(self, subreddit_obj):
        async for com in subreddit_obj.stream.comments(skip_existing=True):
            self._ensure_bucket()
            await self.handle_comment(com)
            await self.detect()

    async def _leaderboard_loop(self):
        while True:
            try:
                self._print_leaderboard(top_n=10)
            except Exception as e:
                print("leaderboard err:", e)
            await asyncio.sleep(LEADERBOARD_INTERVAL)

    # ---------- runner ----------
    async def run(self):
        if not CLIENT_ID or not CLIENT_SECRET or not USER_AGENT:
            print("ERROR: Missing Reddit credentials. Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT.")
            return
        reddit = asyncpraw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)
        try:
            subreddit_obj = await reddit.subreddit("+".join(self.subs))
        except Exception as e:
            if "401" in str(e) or "unauthorized" in str(e).lower():
                print("\nERROR: Reddit returned 401 Unauthorized. Check your Reddit app (script type) and env vars.")
            print("run err:", e)
            try:
                await reddit.close()
            except Exception:
                pass
            return
        try:
            t1 = asyncio.create_task(self._consume_submissions(subreddit_obj))
            t2 = asyncio.create_task(self._consume_comments(subreddit_obj))
            t3 = asyncio.create_task(self._leaderboard_loop())
            await asyncio.gather(t1, t2, t3)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print("run err:", e)
        finally:
            try:
                await reddit.close()
            except Exception:
                pass

# --------------------------- main ---------------------------
if __name__ == "__main__":
    monitor = MomentumMonitor()
    try:
        asyncio.run(monitor.run())
    except KeyboardInterrupt:
        print("Interrupted — exiting.")

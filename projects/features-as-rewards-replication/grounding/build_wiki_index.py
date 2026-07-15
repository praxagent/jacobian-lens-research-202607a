"""Stream a Wikipedia pages-articles .bz2 dump into a SQLite FTS5 index (title + body).

Direct bz2 -> FTS5, no multi-GB intermediate extract. Articles only (ns 0), redirects
skipped, wikitext markup stripped with a fast regex pass (good enough for retrieval
grounding). One-time build; the resulting .db is the pinned, reproducible grounding
snapshot (record the dump date + the .bz2 SHA-256 in the per-arm manifest).

  .venv/bin/python grounding/build_wiki_index.py \
     --dump /home/ubuntu/wiki-snapshot/enwiki-20260701-pages-articles.xml.bz2 \
     --db   /home/ubuntu/wiki-snapshot/enwiki-20260701-fts.db
"""
from __future__ import annotations

import argparse
import bz2
import re
import sqlite3
import time
import xml.etree.ElementTree as ET

# --- fast, approximate wikitext stripping (retrieval quality, not perfect rendering) ---
_RE = [
    (re.compile(r"<ref[^>]*>.*?</ref>", re.S | re.I), " "),
    (re.compile(r"<ref[^>]*/>", re.I), " "),
    (re.compile(r"<!--.*?-->", re.S), " "),
    (re.compile(r"\{\{[^{}]*\}\}"), " "),          # simple templates (repeat below for nesting)
    (re.compile(r"\{\|.*?\|\}", re.S), " "),        # tables
    (re.compile(r"\[\[(?:File|Image|Category):[^\]]*\]\]", re.I), " "),
    (re.compile(r"<[^>]+>"), " "),                  # html tags
    (re.compile(r"'''?|=={1,6}"), " "),             # bold/italic/headings markup
    (re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]"), r"\1"),  # [[a|b]] -> b
    (re.compile(r"\[https?://\S+\s+([^\]]+)\]"), r"\1"),      # [url text] -> text
    (re.compile(r"\[https?://\S+\]"), " "),
    (re.compile(r"&[a-z]+;"), " "),
    (re.compile(r"[ \t]+"), " "),
    (re.compile(r"\n{2,}"), "\n"),
]


def strip_wikitext(t: str) -> str:
    for _ in range(3):                               # a few passes for nested {{...}}
        t = _RE[3][0].sub(" ", t)
    for rx, rep in _RE:
        t = rx.sub(rep, t)
    return t.strip()


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def build(dump: str, db: str, limit: int | None, log_every: int):
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("DROP TABLE IF EXISTS wiki")
    con.execute("CREATE VIRTUAL TABLE wiki USING fts5(title, body, tokenize='porter unicode61')")
    ins = "INSERT INTO wiki(title, body) VALUES (?, ?)"

    t0 = time.time()
    n = kept = 0
    batch = []
    title = ns = text = None
    with bz2.open(dump, "rb") as f:
        for event, elem in ET.iterparse(f, events=("end",)):
            tag = local(elem.tag)
            if tag == "title":
                title = elem.text
            elif tag == "ns":
                ns = elem.text
            elif tag == "text":
                text = elem.text
            elif tag == "page":
                n += 1
                if ns == "0" and text and not text.lstrip()[:9].upper().startswith("#REDIRECT"):
                    body = strip_wikitext(text)
                    if len(body) > 40:
                        batch.append((title or "", body))
                        kept += 1
                if len(batch) >= 2000:
                    con.executemany(ins, batch); con.commit(); batch.clear()
                if n % log_every == 0:
                    rate = n / max(1e-9, time.time() - t0)
                    print(f"  pages={n:,} kept={kept:,} {rate:,.0f}/s "
                          f"elapsed={time.time()-t0:,.0f}s", flush=True)
                title = ns = text = None
                elem.clear()
                if limit and n >= limit:
                    break
    if batch:
        con.executemany(ins, batch); con.commit()
    con.execute("INSERT INTO wiki(wiki) VALUES('optimize')"); con.commit()
    con.close()
    print(f"DONE pages={n:,} articles_indexed={kept:,} in {time.time()-t0:,.0f}s -> {db}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--limit", type=int, default=None, help="max pages (smoke)")
    ap.add_argument("--log-every", type=int, default=100000)
    a = ap.parse_args()
    build(a.dump, a.db, a.limit, a.log_every)

"""Stream a Wikipedia pages-articles .bz2 dump into a SQLite FTS5 index (title + intro).

v2 — throughput rewrite after the 20k-page smoke measured ~88 pages/s with ElementTree +
full-body stripping (~70h full build; unacceptable). Changes:
  * hand-rolled streaming tag scanner (no XML tree) — the dump's page structure is flat
    and line-friendly, ET.iterparse was the parse bottleneck;
  * index each article's INTRO ONLY (first --intro-chars of wikitext, capped BEFORE the
    regex pass) — identity/date/definition facts live in the lead section, and stripping
    cost is O(len);
  * light single-pass strip; unicode61 tokenizer (no porter) for cheaper inserts.
Grounding quality tradeoff (intro-only) is disclosed in GROUNDING.md; deep-body facts fall
back to 'Insufficient' at grading time, which is an exclusion, not a wrong label.

  .venv/bin/python grounding/build_wiki_index.py \
     --dump /home/ubuntu/wiki-snapshot/enwiki-20260701-pages-articles.xml.bz2 \
     --db   /home/ubuntu/wiki-snapshot/enwiki-20260701-fts.db
"""
from __future__ import annotations

import argparse
import bz2
import html
import re
import sqlite3
import time

RX_STRIP = re.compile(
    r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>|<!--.*?-->|\{\{[^{}]*\}\}|\{\|.*?\|\}|<[^>]+>|'''?|=={1,6}",
    re.S | re.I)
RX_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
RX_EXT = re.compile(r"\[https?://\S+( [^\]]+)?\]")
RX_WS = re.compile(r"[ \t]{2,}")


def strip_intro(t: str) -> str:
    t = RX_STRIP.sub(" ", t)
    t = RX_STRIP.sub(" ", t)          # second pass for one level of nesting
    t = RX_LINK.sub(r"\1", t)
    t = RX_EXT.sub(r"\1", t)
    t = html.unescape(t)
    return RX_WS.sub(" ", t).strip()


RX_TITLE = re.compile(rb"<title>(.*?)</title>", re.S)
RX_NS = re.compile(rb"<ns>(\d+)</ns>")
RX_TEXT = re.compile(rb"<text[^>]*>", re.S)


def pages(dump_path: str, intro_bytes: int = 6000):
    """Yield (title, intro_wikitext). v3: bzcat subprocess (C decompress on the 2nd core)
    + binary chunk scanning with compiled regex; only the kept intro is ever decoded.
    Falls back to Python bz2 if bzcat is unavailable."""
    import subprocess
    try:
        proc = subprocess.Popen(["bzcat", dump_path], stdout=subprocess.PIPE,
                                bufsize=1 << 22)
        stream = proc.stdout
    except FileNotFoundError:
        stream = bz2.open(dump_path, "rb")
    buf = b""
    CHUNK = 1 << 23  # 8MB
    while True:
        chunk = stream.read(CHUNK)
        if not chunk:
            break
        buf += chunk
        start = 0
        while True:
            pe = buf.find(b"</page>", start)
            if pe == -1:
                break
            ps = buf.rfind(b"<page>", start, pe)
            page = buf[ps if ps != -1 else start: pe]
            start = pe + 7
            m = RX_NS.search(page)
            if not m or m.group(1) != b"0":
                continue
            t = RX_TITLE.search(page)
            x = RX_TEXT.search(page)
            if not t or not x:
                continue
            body = page[x.end(): x.end() + intro_bytes]
            if body.lstrip()[:12].upper().startswith(b"#REDIRECT"):
                continue
            yield (html.unescape(t.group(1).decode("utf-8", "replace")),
                   body.decode("utf-8", "replace"))
        buf = buf[start:]


def build(dump: str, db: str, limit: int | None, log_every: int, intro_chars: int):
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("DROP TABLE IF EXISTS wiki")
    con.execute("CREATE VIRTUAL TABLE wiki USING fts5(title, body, tokenize='unicode61')")
    ins = "INSERT INTO wiki(title, body) VALUES (?, ?)"
    t0 = time.time()
    n = kept = 0
    batch = []
    for title, raw in pages(dump, intro_bytes=intro_chars + 2000):
        n += 1
        body = strip_intro(raw[:intro_chars])
        if len(body) > 40:
            batch.append((title, body))
            kept += 1
        if len(batch) >= 4000:
            con.executemany(ins, batch); con.commit(); batch.clear()
        if n % log_every == 0:
            rate = n / max(1e-9, time.time() - t0)
            print(f"  pages={n:,} kept={kept:,} {rate:,.0f}/s elapsed={time.time()-t0:,.0f}s",
                  flush=True)
        if limit and n >= limit:
            break
    if batch:
        con.executemany(ins, batch); con.commit()
    con.execute("INSERT INTO wiki(wiki) VALUES('optimize')"); con.commit()
    con.close()
    print(f"DONE pages={n:,} articles_indexed={kept:,} in {time.time()-t0:,.0f}s -> {db}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=200000)
    ap.add_argument("--intro-chars", type=int, default=4000)
    a = ap.parse_args()
    build(a.dump, a.db, a.limit, a.log_every, a.intro_chars)

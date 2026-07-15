"""BM25 retrieval over the local Wikipedia FTS5 index (free, deterministic)."""
from __future__ import annotations

import re
import sqlite3

_SAFE = re.compile(r"[^\w\s]")


def _fts_query(text: str) -> str:
    """Sanitize an entity string into an FTS5 MATCH query (OR of its terms)."""
    toks = [t for t in _SAFE.sub(" ", text).split() if len(t) > 1][:12]
    return " OR ".join(f'"{t}"' for t in toks) if toks else '""'


class WikiIndex:
    def __init__(self, db_path: str):
        self.con = sqlite3.connect(db_path)
        self.con.row_factory = sqlite3.Row

    def search(self, query: str, k: int = 3, snippet_chars: int = 500):
        """Top-k (title, snippet) for an entity/claim string, BM25-ranked."""
        q = _fts_query(query)
        try:
            rows = self.con.execute(
                "SELECT title, substr(body,1,?) AS snip, bm25(wiki) AS score "
                "FROM wiki WHERE wiki MATCH ? ORDER BY score LIMIT ?",
                (snippet_chars, q, k)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [{"title": r["title"], "snippet": r["snip"], "score": r["score"]} for r in rows]

    def close(self):
        self.con.close()

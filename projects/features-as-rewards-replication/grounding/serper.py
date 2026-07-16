"""Serper.dev search client — prepaid credits, disk cache, hard in-code caps.

Reads SERPER_DEV_API_KEY (TJ's explicit naming: this is serper.dev, NOT serpapi).
Every query is cached to a jsonl on disk keyed by the exact query string, so re-runs
and validation rounds never re-spend credits, and the archived evidence in receipts is
exactly what the judges saw. Caps: max_queries per run (fails closed), like GuardedLLM.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests


class SerperExceeded(RuntimeError):
    pass


class Serper:
    def __init__(self, api_key, cache_path, max_queries=500, k=5):
        self.key = api_key
        self.cache_path = Path(cache_path)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                try:
                    o = json.loads(line)
                    self.cache[o["q"]] = o["results"]
                except Exception:
                    pass
        self.max_queries = max_queries
        self.spent = 0          # NEW paid queries this run (cache hits are free)
        self.k = k

    def search(self, query):
        """Top-k organic results [{title, link, snippet}]. Cache-first; fails closed."""
        if query in self.cache:
            return self.cache[query][: self.k]
        if self.spent >= self.max_queries:
            raise SerperExceeded(f"query cap {self.max_queries} reached")
        for attempt in range(3):
            try:
                r = requests.post("https://google.serper.dev/search",
                                  headers={"X-API-KEY": self.key,
                                           "Content-Type": "application/json"},
                                  json={"q": query}, timeout=20)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        organic = [{"title": o.get("title", ""), "link": o.get("link", ""),
                    "snippet": o.get("snippet", "")}
                   for o in r.json().get("organic", [])][:8]
        self.spent += 1
        self.cache[query] = organic
        with open(self.cache_path, "a") as f:
            f.write(json.dumps({"q": query, "results": organic}) + "\n")
        return organic[: self.k]

    def summary(self):
        return {"new_queries_spent": self.spent, "cache_size": len(self.cache),
                "max_queries": self.max_queries}

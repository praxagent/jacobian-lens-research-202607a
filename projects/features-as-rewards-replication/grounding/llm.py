"""Budget-guarded OpenRouter chat client — fails closed on spend.

The guard that was missing when the Pro-review call overran its authorization. Every
paid-API runner in this project uses this. Two hard limits, checked BEFORE each call:
  * --max-usd:   running spend (from response usage x pinned rates) may not cross this.
  * --max-calls: total calls may not exceed this.
Plus a per-call output cap (max_tokens) so a single call's cost is bounded. Together the
worst-case spend is bounded by max_calls x (max_in + max_out) x rate, and the run stops
the instant running spend reaches max_usd. --dry-run makes zero paid calls.

Rates are pinned per model (USD per 1M tokens); pass the current OpenRouter rates.
"""
from __future__ import annotations

import time


class BudgetExceeded(RuntimeError):
    pass


class GuardedLLM:
    def __init__(self, api_key, model, rate_in, rate_out, max_usd, max_calls,
                 max_tokens=700, temperature=0.0, dry_run=False, base_url=None,
                 warn_frac=0.8):
        # NOTE (TJ 2026-07-15): max_usd is a RUNAWAY BACKSTOP, not a tight budget. Set it
        # with generous headroom over the expected cost so a legitimately-completing run is
        # never killed a few dollars short. The true hard wall is the prepaid/per-key
        # ceiling on the OpenRouter account ($50), which cannot be exceeded regardless.
        # warn_frac emits a visibility warning before any stop.
        self.model = model
        self.rate_in = rate_in
        self.rate_out = rate_out
        self.max_usd = float(max_usd)
        self.max_calls = int(max_calls)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.dry_run = dry_run
        self.warn_frac = warn_frac
        self._warned = False
        self.spent = 0.0
        self.calls = 0
        self.halted = False
        self._client = None
        if not dry_run:
            from openai import OpenAI
            self._client = OpenAI(base_url=base_url or "https://openrouter.ai/api/v1",
                                  api_key=api_key)

    def _cost(self, usage):
        return (usage.prompt_tokens * self.rate_in
                + usage.completion_tokens * self.rate_out) / 1_000_000

    def can_call(self):
        return (not self.halted and self.calls < self.max_calls
                and self.spent < self.max_usd)

    def chat(self, messages, max_tokens=None):
        """Returns (text, usage_dict). Raises BudgetExceeded before any call that would
        breach a ceiling. In dry_run, returns ('', estimate) and makes no paid call."""
        if self.dry_run:
            # rough estimate: assume the call would spend an output cap's worth
            est_out = max_tokens or self.max_tokens
            est_in = sum(len(m.get("content", "")) for m in messages) // 4
            est = (est_in * self.rate_in + est_out * self.rate_out) / 1_000_000
            self.spent += est
            self.calls += 1
            return "", {"dry_run": True, "est_usd": est, "cum_usd": self.spent}
        if not self.can_call():
            raise BudgetExceeded(
                f"halt: spent=${self.spent:.4f}/{self.max_usd} calls={self.calls}/{self.max_calls}")
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=self.temperature)
                # some OpenRouter providers return an error payload with usage=None —
                # treat as a transient provider failure, not a valid response
                if resp.usage is None or not resp.choices:
                    raise RuntimeError(
                        f"provider returned no usage/choices: {getattr(resp, 'error', None)}")
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        self.calls += 1
        cost = self._cost(resp.usage)
        self.spent += cost
        if not self._warned and self.spent >= self.warn_frac * self.max_usd:
            self._warned = True
            print(f"  [budget] ${self.spent:.3f}/{self.max_usd} spent "
                  f"({self.warn_frac:.0%} of the backstop) — approaching the in-code limit; "
                  f"raise --max-usd if the run needs to finish (hard wall is the $50 key cap)")
        if self.spent >= self.max_usd:
            self.halted = True     # backstop reached; stop launching NEW calls (finish current)
        txt = resp.choices[0].message.content or ""
        return txt, {"prompt_tokens": resp.usage.prompt_tokens,
                     "completion_tokens": resp.usage.completion_tokens,
                     "usd": cost, "cum_usd": self.spent}

    def summary(self):
        return {"model": self.model, "calls": self.calls,
                "spent_usd": round(self.spent, 4), "max_usd": self.max_usd,
                "halted": self.halted, "dry_run": self.dry_run}

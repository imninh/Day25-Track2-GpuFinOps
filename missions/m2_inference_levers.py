"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Includes the "Your Turn" Ext 3: cache is only applied when the measured prefix
reuse beats `cache_break_even_reads()` (see `finops/pricing.py`).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import Counter
from missions._common import load_csv, num
from finops import pricing

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# One-time cost to write/cache a 1M-token prefix per model tier.
CACHE_WRITE_COST_PER_M = {"small": 0.50, "large": 1.50}


def cache_economics(rows) -> dict:
    """Ext 3 — is prompt caching worth it given the real prefix reuse?

    Prefixes are keyed by (team, project) — each team's system prompt. The
    average number of reads per prefix is compared with the break-even number
    of reads for each model tier.
    """
    cache = [r for r in rows if int(num(r["cached_input_tokens"])) > 0]
    if not cache:
        return {"avg_cache_reads": 0.0, "by_tier": {}}
    prefix_count = Counter((r["team"], r["project"]) for r in cache)
    avg_reads = sum(prefix_count[(r["team"], r["project"])] for r in cache) / len(cache)
    by_tier = {}
    for tier, (pin, _pout) in MODEL_PRICES.items():
        write = CACHE_WRITE_COST_PER_M[tier]
        be = pricing.cache_break_even_reads(write, pin)
        by_tier[tier] = {
            "break_even_reads": round(be, 2),
            "worth_it": pricing.cache_is_worth_it(avg_reads, write, pin),
        }
    return {"avg_cache_reads": round(avg_reads, 2), "by_tier": by_tier}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    econ = cache_economics(rows)
    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        # Ext 3: only take the cache discount if reuse beats break-even
        tier_worth = econ["by_tier"].get(r["route_tier"], {}).get("worth_it", True)
        cached_in = cached if tier_worth else 0
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached_in, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print(f"\n[Ext 3] cache economics: avg reads/prefix = {econ['avg_cache_reads']}")
        for tier, info in econ["by_tier"].items():
            print(f"  {tier:6} model: break-even = {info['break_even_reads']:>6.1f} reads"
                  f"  -> cache worth it? {info['worth_it']}")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_econ": econ,
    }


if __name__ == "__main__":
    run()

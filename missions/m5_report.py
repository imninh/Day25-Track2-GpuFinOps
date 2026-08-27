"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}

AUTHOR = {
    "name": "Trần Thế Ninh",
    "student_id": "2A202602001",
    "github": "imninh",
}


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    best_region = min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "energy_cost_usd": sustainability.energy_cost_usd(wh, "us-east-1"),
        "best_region": best_region,
        "carbon_saved_g": r3["carbon"]["saved_g"],
        "carbon_saved_pct": r3["carbon"]["saved_pct"],
    }

    # --- C.2 deep-dive ---
    analysis = [
        "### Why GPU-Util is a \"lie\" and what it costs you",
        "`nvidia-smi` reports GPU-Util as the fraction of the sample window the "
        "GPU had work queued — it is a *clock-active* signal, not an efficiency "
        "meter. `gpu-h100-4` shows 98% util yet only ~0.19 MFU: the tensor cores "
        "are busy with kernel launches, memory stalls and pipeline bubbles, so you "
        "pay for a full H100 hour while getting roughly one fifth of the FLOPs you "
        "rented. Right-sizing it to an A100 (and `gpu-a10g-1` to an L4) recovers "
        f"~${rightsize_savings:,.0f}/month.",
        "### Why purchasing is the biggest lever here",
        f"Purchasing (spot + reserved) saves ${purchasing_savings:,.0f}/month — the "
        "largest bucket — because most GPU spend is *infrastructure*, not tokens. "
        "24/7 inference services clear the 55% break-even, so reserved (3yr) applies; "
        "interruptible training rides spot + checkpoints. The extended policy prices "
        "spot with each GPU's real interruption rate (A10G is reclaimed ~25% of the "
        "time) and matches reserved term to job length, so savings are "
        f"{r3['savings_pct']:.1f}% vs a naive {r3['old_savings_pct']:.1f}%.",
        "### Cascade + caching + batch: 82.6% off $/1M-token",
        f"Routing 80% of requests to the small model (cascade), discounting cached "
        f"input at 10% and batching eval traffic at 50% drops inference from "
        f"${r2['baseline_per_m']:.3f} to ${r2['optimized_per_m']:.3f}/1M-token. "
        "Ext 3 confirms caching pays: break-even is ~2.8 reads (small) / ~0.6 "
        "reads (large), while the dataset shows ~536 reads per prefix.",
        "### Idle GPUs are pure burn",
        f"`gpu-h100-5` idles 8h/day and was only caught because we track "
        f"utilization <10% — ${r1['idle_waste_daily']*DAYS:,.0f}/month. The fix is "
        "trivial: scale-to-zero or a schedule.",
    ]

    priority_actions = [
        ("Adopt $/1M-token as the KPI, not $/GPU-hr",
         "it forces teams to report efficiency, exposing 20%-MFU H100s instantly."),
        ("Move 24/7 inference to 3yr reserved and interruptible training to spot+checkpoint",
         f"~${purchasing_savings:,.0f}/month, the single largest bucket."),
        ("Roll out cascade routing (80% to the small model) with prompt caching and batch API",
         f"~${infer_savings:,.0f}/month while keeping quality via cascade fallback."),
        ("Right-size the two \"util-lie\" GPUs and kill idle instances",
         f"~${rightsize_savings + idle_savings:,.0f}/month of nearly-free savings."),
        ("Schedule interruptible jobs in the cleanest region",
         f"europe-north1 cuts carbon by {r3['carbon']['saved_pct']:.0f}% and costs "
         "less electricity than us-east-1 — sustainability and cost move together."),
    ]

    extensions = [
        ("Ext 1 — interruption-aware tier choice + term matching",
         "`recommend_tier()` now uses a per-GPU spot interruption rate and "
         "`reserved_hourly_rate()` prices 1yr vs 3yr by real job length, so we stop "
         "over-committing 3yr for short jobs and stop over-paying for unreliable spot.",
         f"extended policy: {r3['savings_pct']:.1f}% vs original {r3['old_savings_pct']:.1f}% "
         "(more realistic, same tier mix)"),
        ("Ext 3 — `cache_is_worth_it()`",
         "Prompt caching is only applied when measured prefix reuse clears the "
         "break-even read count per model tier.",
         f"break-even {r2['cache_econ']['by_tier']['small']['break_even_reads']} reads "
         f"(small) / {r2['cache_econ']['by_tier']['large']['break_even_reads']} reads "
         f"(large) vs {r2['cache_econ']['avg_cache_reads']} actual reads -> caching "
         "is clearly worth it"),
        ("Ext 5 — carbon-aware scheduling",
         "Interruptible jobs are costed in all 5 regions (electricity + carbon); "
         "moving them from us-east-1 to europe-north1 is both cheaper and 92% cleaner.",
         f"{r3['carbon']['saved_g']:,} gCO2e avoided ({r3['carbon']['saved_pct']:.1f}%)"),
    ]

    md = report.build_report(
        baseline, optimized, levers, sustainability=sust,
        author=AUTHOR,
        token_metrics={"baseline_per_m": r2["baseline_per_m"],
                       "optimized_per_m": r2["optimized_per_m"],
                       "savings_pct": r2["savings_pct"]},
        analysis=analysis, priority_actions=priority_actions, extensions=extensions,
    )
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"))

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()

"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Includes two "Your Turn" extensions:
  - Ext 1: improved recommend_tier() (per-GPU interruption rate + 1yr/3yr term)
  - Ext 5: carbon-aware scheduling for interruptible training jobs.

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30


def carbon_schedule(jobs, cat, days=DAYS) -> dict:
    """Ext 5 — carbon-aware scheduling for interruptible jobs.

    Estimates gCO2e and electricity cost for each interruptible job in every
    region, then recommends the cheapest / cleanest / balanced region.
    """
    total_energy_kwh = 0.0
    per_job = []
    for j in jobs:
        if not bool(int(num(j["interruptible"]))):
            continue
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        watts = num(cat[gtype]["watts"])
        gpu_hours = hpd * days * ngpu
        energy_kwh = watts / 1000.0 * gpu_hours
        total_energy_kwh += energy_kwh
        per_job.append({"job_id": j["job_id"], "gpu_type": gtype,
                        "energy_kwh": round(energy_kwh, 1)})

    rows = []
    for region, gco2_kwh in sorted(sustainability.REGION_CARBON.items()):
        usd_kwh = sustainability.REGION_PRICE_KWH.get(region, 0.12)
        carbon_g = total_energy_kwh * gco2_kwh
        energy_usd = total_energy_kwh * usd_kwh
        rows.append({"region": region, "usd_kwh": usd_kwh, "gco2_kwh": gco2_kwh,
                     "carbon_g": round(carbon_g), "energy_usd": round(energy_usd, 1)})

    best_cost = min(rows, key=lambda r: r["energy_usd"])
    best_carbon = min(rows, key=lambda r: r["carbon_g"])
    best_balanced = min(rows, key=lambda r: r["carbon_g"] + r["energy_usd"] * 500)
    base = next(r for r in rows if r["region"] == "us-east-1")
    clean = next(r for r in rows if r["region"] == "europe-north1")
    saved_g = base["carbon_g"] - clean["carbon_g"]
    return {"jobs": per_job, "total_energy_kwh": round(total_energy_kwh, 1), "rows": rows,
            "best_cost": best_cost["region"], "best_carbon": best_carbon["region"],
            "best_balanced": best_balanced["region"],
            "carbon_saved_g": saved_g,
            "carbon_saved_pct": (saved_g / base["carbon_g"] * 100) if base["carbon_g"] else 0.0}


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    old_optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        jdays = num(j["days"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        steady_service = (not interruptible) and hpd >= 20

        # Ext 1 — extended policy (per-GPU interruption rate + real term)
        tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, job_days=jdays)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(
                gpu_hours, num(c["spot_hr"]), od,
                interrupt_rate=pricing.spot_interrupt_rate(gtype))
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * pricing.reserved_hourly_rate(c, jdays, steady=steady_service)
        else:
            opt_cost = on_demand_cost

        # Original policy, for before/after comparison
        old_tier = pricing.recommend_tier(hpd, interruptible)
        if old_tier == "spot":
            old_opt = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)["spot_cost"]
        elif old_tier == "reserved":
            old_opt = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            old_opt = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        old_optimized_monthly += old_opt
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    old_savings_pct = (on_demand_monthly - old_optimized_monthly) / on_demand_monthly * 100 if on_demand_monthly else 0.0

    cs = carbon_schedule(jobs, cat)

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        print(f"[Ext 1] policy comparison: original policy saved {old_savings_pct:.1f}% -> "
              f"extended policy saved {savings_pct:.1f}%")

        print("\n== M3 [Ext 5] Carbon-aware scheduling ==")
        print("interruptible jobs -> energy & carbon by region:")
        print(f"{'region':14}{'$/kWh':>7}{'gCO2/kWh':>10}{'elec.$':>10}{'gCO2e':>10}")
        for r in cs["rows"]:
            print(f"{r['region']:14}{r['usd_kwh']:>7.3f}{r['gco2_kwh']:>10}{r['energy_usd']:>10,.0f}{r['carbon_g']:>10,}")
        print(f"\nmove all interruptible jobs us-east-1 -> europe-north1: "
              f"save {cs['carbon_saved_g']:,.0f} gCO2e ({cs['carbon_saved_pct']:.1f}%)")
        print(f"cheapest: {cs['best_cost']}  |  cleanest: {cs['best_carbon']}  |  balanced: {cs['best_balanced']}")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "old_savings_pct": round(old_savings_pct, 1),
            "carbon": {"saved_g": cs["carbon_saved_g"], "saved_pct": round(cs["carbon_saved_pct"], 1),
                       "best_cost": cs["best_cost"], "best_carbon": cs["best_carbon"],
                       "best_balanced": cs["best_balanced"]}}


if __name__ == "__main__":
    run()

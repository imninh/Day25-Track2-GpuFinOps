# NimbusAI — GPU Cost Optimization Report

**Author:** Trần Thế Ninh  
**Student ID:** 2A202602001  
**GitHub:** imninh  
**Period:** monthly  

**Baseline spend:** $27,133  
**Optimized spend:** $15,016  
**Projected savings:** $12,117  (**45%**)

## Executive summary — measure in $/1M-token

The lab's yardstick is cost per 1M tokens served, not $/GPU-hr: two
teams can pay the same $/GPU-hr while one serves 10x more tokens.

- **Baseline:** $6.488/1M-token (large model everywhere, no cache, no batch)
- **Optimized:** $1.126/1M-token (cascade + prompt caching + batch)
- **Inference savings:** 82.6%

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,212 |
| Purchasing (spot/reserved) | $9,650 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Analysis

### Why GPU-Util is a "lie" and what it costs you

`nvidia-smi` reports GPU-Util as the fraction of the sample window the GPU had work queued — it is a *clock-active* signal, not an efficiency meter. `gpu-h100-4` shows 98% util yet only ~0.19 MFU: the tensor cores are busy with kernel launches, memory stalls and pipeline bubbles, so you pay for a full H100 hour while getting roughly one fifth of the FLOPs you rented. Right-sizing it to an A100 (and `gpu-a10g-1` to an L4) recovers ~$655/month.

### Why purchasing is the biggest lever here

Purchasing (spot + reserved) saves $9,650/month — the largest bucket — because most GPU spend is *infrastructure*, not tokens. 24/7 inference services clear the 55% break-even, so reserved (3yr) applies; interruptible training rides spot + checkpoints. The extended policy prices spot with each GPU's real interruption rate (A10G is reclaimed ~25% of the time) and matches reserved term to job length, so savings are 37.6% vs a naive 39.1%.

### Cascade + caching + batch: 82.6% off $/1M-token

Routing 80% of requests to the small model (cascade), discounting cached input at 10% and batching eval traffic at 50% drops inference from $6.488 to $1.126/1M-token. Ext 3 confirms caching pays: break-even is ~2.8 reads (small) / ~0.6 reads (large), while the dataset shows ~536 reads per prefix.

### Idle GPUs are pure burn

`gpu-h100-5` idles 8h/day and was only caught because we track utilization <10% — $600/month. The fix is trivial: scale-to-zero or a schedule.

## Recommended actions (by ROI)

**1. Adopt $/1M-token as the KPI, not $/GPU-hr** — it forces teams to report efficiency, exposing 20%-MFU H100s instantly.
**2. Move 24/7 inference to 3yr reserved and interruptible training to spot+checkpoint** — ~$9,650/month, the single largest bucket.
**3. Roll out cascade routing (80% to the small model) with prompt caching and batch API** — ~$1,212/month while keeping quality via cascade fallback.
**4. Right-size the two "util-lie" GPUs and kill idle instances** — ~$1,255/month of nearly-free savings.
**5. Schedule interruptible jobs in the cleanest region** — europe-north1 cuts carbon by 92% and costs less electricity than us-east-1 — sustainability and cost move together.

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1
- Electricity cost per query: $0.00003
- Carbon avoided by moving interruptible jobs to europe-north1: 1,479,450 gCO2e (92.1%)

## 'Your Turn' extensions performed

### Ext 1 — interruption-aware tier choice + term matching
`recommend_tier()` now uses a per-GPU spot interruption rate and `reserved_hourly_rate()` prices 1yr vs 3yr by real job length, so we stop over-committing 3yr for short jobs and stop over-paying for unreliable spot.
**Result:** extended policy: 37.6% vs original 39.1% (more realistic, same tier mix)

### Ext 3 — `cache_is_worth_it()`
Prompt caching is only applied when measured prefix reuse clears the break-even read count per model tier.
**Result:** break-even 2.78 reads (small) / 0.56 reads (large) vs 536.11 actual reads -> caching is clearly worth it

### Ext 5 — carbon-aware scheduling
Interruptible jobs are costed in all 5 regions (electricity + carbon); moving them from us-east-1 to europe-north1 is both cheaper and 92% cleaner.
**Result:** 1,479,450 gCO2e avoided (92.1%)

---
_Figures are June-2026 as-of snapshots; re-baseline before acting._
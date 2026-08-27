"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


# Per-GPU spot interruption risk (illustrative 2026) — H100 spot is usually
# reclaimed less often than small inference GPUs like A10G/L4.
GPU_SPOT_INTERRUPT_RATE = {
    "H100": 0.05, "H200": 0.05, "B200": 0.08,
    "A100": 0.12, "A10G": 0.25, "L4": 0.20, "MI300X": 0.15,
}

# Pathologically unreliable spot — rework cost eats the discount entirely.
HIGH_INTERRUPT_RATE = 0.60


def spot_interrupt_rate(gpu_type: str | None) -> float:
    """Interruption rate for a GPU type (used to price spot honestly)."""
    return GPU_SPOT_INTERRUPT_RATE.get(gpu_type, 0.15)


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45,
                   gpu_type: str | None = None, job_days: float | None = None) -> str:
    """Pick a purchasing tier from duty cycle + interruptibility + GPU/term specifics.

    Extended policy (beyond the documented simple one):
      1. Spot is priced with the GPU's real interruption rate instead of a flat 5%
         — `spot_interrupt_rate()`/`spot_checkpoint_cost()` reflect rework cost
         per GPU type (A10G/L4 spot is reclaimed far more often than H100/A100).
      2. Reserved term is matched to the job's real length via
         `reserved_hourly_rate()` — 3yr for steady services, 1yr for bounded jobs.
    Calling with only (hours_per_day, interruptible) reproduces the original policy.
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        rate = spot_interrupt_rate(gpu_type)
        if rate <= HIGH_INTERRUPT_RATE:
            return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


def reserved_hourly_rate(catalog_row: dict, job_days: float | None = None,
                         steady: bool = False) -> float:
    """Pick the reserved hourly rate: 1yr for short-lived jobs, 3yr for long-lived ones.

    A 3yr commitment is cheaper per hour but over-pays when the job finishes in
    under a year. `steady=True` (always-on inference service) keeps 3yr; bounded
    jobs under a year are priced at the 1yr rate instead of being over-committed.
    """
    one_yr = float(catalog_row.get("reserved_1yr_hr", 0.0) or 0.0)
    three_yr = float(catalog_row.get("reserved_3yr_hr", 0.0) or 0.0)
    if steady:
        return three_yr
    if job_days is not None and job_days < 365:
        return one_yr
    return three_yr


def cache_break_even_reads(write_cost_per_m: float, price_in_per_m: float,
                           read_discount: float = 0.10) -> float:
    """Number of cache reads needed to pay back the one-time write cost.

    Each read of a cached 1M-token prefix saves `price_in * (1 - read_discount)`.
    """
    savings_per_read = price_in_per_m * (1.0 - read_discount)
    if savings_per_read <= 0:
        return float("inf")
    return write_cost_per_m / savings_per_read


def cache_is_worth_it(avg_cache_reads: float, write_cost_per_m: float,
                      price_in_per_m: float, read_discount: float = 0.10) -> bool:
    """True when repeated reads of a cached prefix save more than the write cost.

    Caching is not free: writing a prefix costs `write_cost_per_m`. It only pays
    off when the prefix is reused enough times to amortize that write.
    """
    return avg_cache_reads >= cache_break_even_reads(write_cost_per_m, price_in_per_m,
                                                     read_discount)


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }

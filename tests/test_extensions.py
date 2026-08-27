import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from finops import pricing, sustainability
from missions import m2_inference_levers, m3_purchasing


# ---- Ext 3: cache_is_worth_it / cache_break_even_reads ----

def test_cache_break_even_reads():
    # each read saves price_in * (1 - read_discount) = 0.9 per 1M-token prefix
    assert abs(pricing.cache_break_even_reads(0.9, 1.0) - 1.0) < 1e-9
    assert abs(pricing.cache_break_even_reads(1.8, 1.0) - 2.0) < 1e-9
    assert pricing.cache_break_even_reads(1.0, 0.0) == float("inf")


def test_cache_is_worth_it():
    # write_cost=1.0, price_in=1.0, read_discount=0.10 -> break-even = 1.11 reads
    assert pricing.cache_is_worth_it(2.0, 1.0, 1.0) is True
    assert pricing.cache_is_worth_it(1.0, 1.0, 1.0) is False
    assert pricing.cache_is_worth_it(0.5, 1.0, 1.0) is False
    # tiny prefix reuse (single shot) never pays back the write
    assert pricing.cache_is_worth_it(1, 5.0, 1.0) is False


def test_m2_cache_economics_present():
    from missions._common import load_csv
    rows = load_csv("token_usage.csv")
    econ = m2_inference_levers.cache_economics(rows)
    assert econ["avg_cache_reads"] > 0
    assert econ["by_tier"]["small"]["break_even_reads"] > 0
    assert isinstance(econ["by_tier"]["large"]["worth_it"], bool)


# ---- Ext 1: recommend_tier with per-GPU interruption rate + term ----

def test_recommend_tier_backwards_compatible():
    # calling without the new args reproduces the documented simple policy
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"


def test_recommend_tier_interruption_rate():
    # low-interruption GPUs stay on spot
    assert pricing.recommend_tier(8, True, gpu_type="H100") == "spot"
    # pathologically unreliable spot falls through to on_demand
    assert pricing.recommend_tier(8, True, gpu_type="A10G") == "spot"
    # a GPU with >60% hourly interruption -> rework eats the discount
    from unittest.mock import patch
    with patch.object(pricing, "GPU_SPOT_INTERRUPT_RATE", {"L4": 0.70}):
        assert pricing.recommend_tier(8, True, gpu_type="L4") == "on_demand"


def test_reserved_hourly_rate_term_matching():
    row = {"reserved_1yr_hr": "0.80", "reserved_3yr_hr": "0.60"}
    # steady always-on service -> 3yr (cheapest per hour)
    assert abs(pricing.reserved_hourly_rate(row, steady=True) - 0.60) < 1e-9
    # bounded job under a year -> 1yr, avoid over-committing 3 years
    assert abs(pricing.reserved_hourly_rate(row, job_days=90) - 0.80) < 1e-9
    # long-lived bounded job -> 3yr
    assert abs(pricing.reserved_hourly_rate(row, job_days=1000) - 0.60) < 1e-9


def test_spot_interrupt_rate_per_gpu():
    assert pricing.spot_interrupt_rate("H100") < pricing.spot_interrupt_rate("A10G")


# ---- Ext 5: carbon-aware scheduling ----

def test_m3_carbon_report():
    res = m3_purchasing.run(verbose=False)
    assert res["carbon"]["saved_g"] > 0
    assert res["carbon"]["saved_pct"] > 0
    assert res["carbon"]["best_carbon"] == "europe-north1"
    assert 55 <= res["carbon"]["saved_pct"] <= 100


def test_region_cleanest_vs_cheapest():
    clean = min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)
    cheap = min(sustainability.REGION_PRICE_KWH, key=sustainability.REGION_PRICE_KWH.get)
    assert clean == "europe-north1"
    assert cheap == "us-east-wa"       # cheapest $, not cleanest -> trade-off exists
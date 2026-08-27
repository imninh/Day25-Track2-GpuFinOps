"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 author: dict | None = None, token_metrics: dict | None = None,
                 analysis: list | None = None, priority_actions: list | None = None,
                 extensions: list | None = None) -> str:
    """Return a markdown cost-optimization report.

    `token_metrics`   {"baseline_per_m": float, "optimized_per_m": float, "savings_pct": float}
    `analysis`        list[str] — root-cause / lever deep-dive paragraphs.
    `priority_actions`list[(action, reason)] ordered by ROI.
    `extensions`      list[(name, summary, result)] — "Your Turn" work.
    """
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
    ]
    if author:
        lines += [
            f"**Author:** {author.get('name', '')}  ",
            f"**Student ID:** {author.get('student_id', '')}  ",
            f"**GitHub:** {author.get('github', '')}  ",
            f"**Period:** {period}  ",
            "",
        ]
    else:
        lines.append(f"**Period:** {period}  ")
    lines += [
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
    ]
    if token_metrics:
        lines += [
            "## Executive summary — measure in $/1M-token",
            "",
            "The lab's yardstick is cost per 1M tokens served, not $/GPU-hr: two",
            "teams can pay the same $/GPU-hr while one serves 10x more tokens.",
            "",
            f"- **Baseline:** ${token_metrics['baseline_per_m']:.3f}/1M-token "
            f"(large model everywhere, no cache, no batch)",
            f"- **Optimized:** ${token_metrics['optimized_per_m']:.3f}/1M-token "
            f"(cascade + prompt caching + batch)",
            f"- **Inference savings:** {token_metrics.get('savings_pct', 0):.1f}%",
            "",
        ]
    lines += [
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) |",
        "|---|---|",
    ]
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} |")

    if analysis:
        lines += ["", "## Analysis", ""]
        for para in analysis:
            lines.append(para)
            lines.append("")

    if priority_actions:
        lines += ["## Recommended actions (by ROI)", ""]
        for i, (action, reason) in enumerate(priority_actions, 1):
            lines.append(f"**{i}. {action}** — {reason}")
        lines.append("")

    if sustainability:
        wh = sustainability.get("wh_per_query", 0)
        lines += [
            "## Sustainability",
            "",
            f"- Energy per query: {wh:.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Cheapest+cleanest region: {sustainability.get('best_region', 'n/a')}",
        ]
        if sustainability.get("energy_cost_usd") is not None:
            lines.append(f"- Electricity cost per query: "
                         f"${sustainability['energy_cost_usd']:.5f}")
        if sustainability.get("carbon_saved_g") is not None:
            lines.append(f"- Carbon avoided by moving interruptible jobs to "
                         f"{sustainability.get('best_region', '')}: "
                         f"{sustainability['carbon_saved_g']:,.0f} gCO2e "
                         f"({sustainability.get('carbon_saved_pct', 0):.1f}%)")
        lines.append("")

    if extensions:
        lines += ["## 'Your Turn' extensions performed", ""]
        for name, summary, result in extensions:
            lines.append(f"### {name}")
            lines.append(summary)
            lines.append(f"**Result:** {result}")
            lines.append("")

    lines += ["---", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str) -> str:
    """Write a simple savings bar chart PNG. Returns the path. No-op if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    names = list(levers.keys())
    vals = [levers[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, vals, color="#2e548a")
    ax.set_ylabel("Savings (USD / month)")
    ax.set_title("GPU cost savings by FinOps lever")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
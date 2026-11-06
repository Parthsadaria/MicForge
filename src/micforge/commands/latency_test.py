"""CLI command to run end-to-end latency benchmarks and stage breakdowns."""
from __future__ import annotations

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..audio.latency import measure_end_to_end_latency
from ..config import AppConfig
from ..ui.theme import console


def run_latency_test(
    app: AppConfig,
    mic: int | str | None = None,
    out: int | str | None = None,
) -> int:
    """Run interactive or automated latency test and display breakdown."""
    console.print("\n[bold cyan]⏱️ Running end-to-end audio latency benchmark…[/]\n")

    input_dev = mic or app.input_device
    output_dev = out or app.output_device or app.monitor_device

    result = measure_end_to_end_latency(
        input_device=input_dev,
        output_device=output_dev,
        sample_rate=app.sample_rate,
        block_size=app.block_size,
        jitter_blocks=app.jitter_blocks,
    )

    # 1. Main Latency Banner
    lat_style = "bold green" if result.one_way_ms < 60.0 else "bold yellow" if result.one_way_ms < 100.0 else "bold red"
    summary_text = (
        f"[{lat_style}]Estimated One-Way Latency: ~{result.one_way_ms:.1f} ms[/]\n"
        f"[dim]Measured Round-Trip Delay: {result.roundtrip_ms:.1f} ms  │  "
        f"Confidence: {result.confidence_score * 100:.0f}% ({result.method})[/]"
    )
    console.print(Panel(summary_text, title="[bold cyan]⚡ LATENCY BENCHMARK RESULT[/]", border_style="cyan", box=box.ROUNDED))

    # 2. Per-Stage Latency Breakdown Table
    t = Table(box=box.SIMPLE_HEAD, header_style="bold cyan")
    t.add_column("Pipeline Stage", width=28)
    t.add_column("Observed Delay", justify="right", width=16)
    t.add_column("Description")

    bd = result.breakdown
    t.add_row("1. Input Hardware Buffer", f"{bd.input_hardware_ms:.1f} ms", f"{app.block_size} samples @ {app.sample_rate} Hz")
    t.add_row("2. 11-Stage DSP & Neural AI", f"{bd.dsp_pipeline_ms:.1f} ms", "Vectorized biquad SOS + CoFi-Lite inference")
    t.add_row("3. Ring Buffer Jitter Cushion", f"{bd.jitter_cushion_ms:.1f} ms", f"{app.jitter_blocks} blocks cushion (prevents crackle)")
    t.add_row("4. Output Hardware Buffer", f"{bd.output_hardware_ms:.1f} ms", f"Virtual cable / headphone DAC render block")
    t.add_section()
    t.add_row("[bold]Total Pipeline Estimate[/]", f"[bold]{bd.total_estimated_ms:.1f} ms[/]", "[bold]Direct one-way mouth-to-Discord latency[/]")

    console.print(t)

    # 3. Actionable Guidance
    console.print("[bold cyan]Actionable Guidance & Tuning:[/\n]")
    for rec in result.recommendations:
        console.print(f"  [dim]•[/] {rec}")
    console.print()

    return 0


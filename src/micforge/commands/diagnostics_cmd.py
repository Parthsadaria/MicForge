"""CLI command to export diagnostics and display the plain-English doctor summary."""
from __future__ import annotations

import sys
from pathlib import Path

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..config import AppConfig
from ..diagnostics import collect_diagnostics, export_diagnostics_zip
from ..ui.theme import console


def run_diagnostics_command(
    app: AppConfig,
    export_path: str | Path | None = None,
    export: bool = False,
) -> int:
    """Run diagnostics checks, print plain-English summary, and optionally export .zip."""
    console.print("\n[bold cyan]🔍 Inspecting Micforge system health & audio pipeline…[/]\n")

    report = collect_diagnostics(app)

    # 1. Status Banner
    status_style = "bold green" if report.overall_status == "HEALTHY" else "bold yellow" if report.overall_status == "WARNING" else "bold red"
    status_icon = "✔" if report.overall_status == "HEALTHY" else "!" if report.overall_status == "WARNING" else "✖"
    console.print(Panel(
        f"[{status_style}]{status_icon} SYSTEM STATUS: {report.overall_status}[/]\n{report.summary_text}",
        border_style=status_style.split()[-1],
        box=box.ROUNDED,
    ))

    # 2. Checks Table
    t = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", show_lines=False)
    t.add_column("", width=3)
    t.add_column("Component", width=22)
    t.add_column("Assessment")

    for c in report.doctor_checks:
        st = c["status"]
        mark = "[bold green]✔[/]" if st == "OK" else "[bold yellow]![/]" if st == "WARN" else "[bold red]✖[/]"
        t.add_row(mark, c["name"], c["detail"])

    console.print(t)

    # 3. Next Steps / Actionable Advice
    if report.next_steps:
        console.print("[bold cyan]Recommended Next Steps:[/]")
        for step in report.next_steps:
            console.print(f"  [dim]•[/] {step}")
        console.print()

    console.print("[dim]📖 For in-depth troubleshooting guides, see:[/] [bold cyan]docs/TROUBLESHOOTING.md[/]\n")

    # 4. Optional Export
    if export or export_path:
        out = export_diagnostics_zip(report, Path(export_path) if export_path else None)
        console.print(Panel(
            f"[bold green]✔ Diagnostics exported successfully![/]\n[white]{out}[/]\n[dim]All passwords, API tokens, and home directory paths have been securely redacted.[/]",
            border_style="green",
            box=box.ROUNDED,
        ))

    return 0 if report.overall_status != "CRITICAL" else 1


"""Command-line preset management for MicForge."""
from __future__ import annotations

import json
from pathlib import Path

from rich import box
from rich.table import Table

from ..config import AppConfig, ChainConfig
from ..errors import ConfigError
from ..presets import (
    BUILTIN,
    clone_preset,
    delete_preset,
    export_preset,
    get_preset,
    import_preset,
    list_presets,
    save_preset,
    user_preset_dir,
)
from ..ui.theme import console


def _format_summary(cfg: ChainConfig) -> str:
    """Format short summary of enabled stages in a preset."""
    parts = []
    if cfg.aec.enabled:
        parts.append("AEC")
    if cfg.denoise.enabled:
        parts.append(f"Denoise({int(cfg.denoise.strength * 100)}%)")
    if cfg.deverb.enabled:
        parts.append(f"DeVerb({int(cfg.deverb.amount * 100)}%)")
    if cfg.proximity.enabled:
        parts.append(f"Body({int(cfg.proximity.amount * 100)}%)")
    if cfg.gate.enabled:
        parts.append(f"Gate({int(cfg.gate.threshold_db)}dB)")
    if cfg.eq_enabled and cfg.eq:
        parts.append(f"EQ({len(cfg.eq)}b)")
    if cfg.compressor.enabled:
        parts.append("Comp")
    return ", ".join(parts) if parts else "Bypass"


def run_preset_list(app: AppConfig) -> None:
    """List all available built-in and custom user presets."""
    presets = list_presets()
    t = Table(box=box.ROUNDED, header_style="bold cyan")
    t.add_column("Status", justify="center", width=8)
    t.add_column("Preset Name", style="bold white")
    t.add_column("Type", justify="center")
    t.add_column("DSP Stages Active", style="dim")
    t.add_column("Description")

    active_name = app.preset

    for p in presets:
        is_active = (p.name == active_name)
        status_badge = "[bold green]● Active[/]" if is_active else "[dim]○[/]"
        type_badge = "[green]Built-in[/]" if p.builtin else "[cyan]Custom[/]"

        try:
            cfg = get_preset(p.name)
            summary = _format_summary(cfg)
        except Exception:
            summary = "Error reading"

        t.add_row(status_badge, p.name, type_badge, summary, p.description)

    console.print(t)
    console.print(f"[dim]Presets directory: {user_preset_dir()}[/]")
    console.print("[dim]Use `micforge preset save <name>` or `micforge live -p <name>`[/]")


def run_preset_show(name: str) -> None:
    """Show detailed JSON configuration for a preset."""
    try:
        cfg = get_preset(name)
    except ConfigError as exc:
        console.print(f"[bold red]✖ {exc}[/]")
        return

    console.print(f"[bold cyan]Preset:[/] [bold green]{name}[/]")
    console.print_json(json.dumps(cfg.to_dict()))


def run_preset_save(
    app: AppConfig,
    name: str,
    from_preset: str | None = None,
    description: str = "",
    force: bool = False,
) -> None:
    """Save a preset (derived from current active config or an existing preset)."""
    base_name = from_preset or app.preset
    cfg = get_preset(base_name)
    desc = description or f"Custom preset based on {base_name}"

    path = save_preset(name, cfg, overwrite=force, description=desc)
    console.print(f"[bold green]✔ Saved custom preset:[/] [bold white]{name}[/]")
    console.print(f"  [dim]Location:[/] {path}")
    console.print(f"  [dim]Run it with:[/] [cyan]micforge live -p {name}[/]")


def run_preset_copy(source: str, target: str, force: bool = False) -> None:
    """Copy an existing preset to a new custom preset."""
    clone_preset(source, target, overwrite=force)
    console.print(f"[bold green]✔ Duplicated preset:[/] [cyan]{source}[/] → [bold white]{target}[/]")
    console.print(f"  [dim]Run it with:[/] [cyan]micforge live -p {target}[/]")


def run_preset_delete(name: str, force: bool = False) -> None:
    """Delete a custom user preset."""
    if name in BUILTIN:
        console.print(f"[bold red]✖ Cannot delete built-in preset '{name}'. Built-ins are permanent.[/]")
        return

    if not force:
        # Prompt for confirmation in console
        from rich.prompt import Confirm
        if not Confirm.ask(f"Are you sure you want to permanently delete custom preset '{name}'?"):
            console.print("[dim]Cancelled.[/]")
            return

    delete_preset(name)
    console.print(f"[bold green]✔ Deleted preset:[/] [white]{name}[/]")


def run_preset_export(name: str, target_path: str | None = None) -> None:
    """Export a preset to a portable JSON file."""
    dest = Path(target_path) if target_path else Path(f"{name}_preset.json")
    out = export_preset(name, dest)
    console.print(f"[bold green]✔ Exported preset '{name}' to:[/] [cyan]{out}[/]")


def run_preset_import(source_path: str, name: str | None = None, force: bool = False) -> None:
    """Import a preset JSON file."""
    preset_name = import_preset(source_path, name=name, overwrite=force)
    console.print(f"[bold green]✔ Successfully imported preset:[/] [bold white]{preset_name}[/]")
    console.print(f"  [dim]Run it with:[/] [cyan]micforge live -p {preset_name}[/]")


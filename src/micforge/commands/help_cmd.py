"""Interactive, dynamic command-line help and usage reference for Micforge."""
from __future__ import annotations

import argparse
from typing import Any

from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .. import __version__
from ..ui.theme import console


# Enhanced metadata enriching argparse introspection with practical examples and tips
COMMAND_METADATA: dict[str, dict[str, Any]] = {
    "run": {
        "category": "🎙️ Live Real-Time Audio",
        "headline": "Start the real-time DSP rack, virtual cable routing, and live HUD.",
        "examples": [
            ("Start streaming with default settings", "micforge run"),
            ("Use gaming preset with 128-sample low latency", "micforge run -p gaming --block 128"),
            ("Listen to your processed voice in headphones", "micforge run --monitor"),
            ("Run silently in system tray without terminal", "micforge run --tray"),
        ],
        "tips": [
            "Press [Tab] anytime during live streaming to toggle the ultra-clean Gamer HUD.",
            "Press [b] to toggle bypass and [x] to mute your microphone.",
            "In Discord or OBS, remember to select 'CABLE Output' as your recording mic.",
        ],
    },
    "loopback": {
        "category": "🎙️ Live Real-Time Audio",
        "headline": "Hear yourself live with instant A/B raw vs. processed comparison.",
        "examples": [
            ("Start loopback monitor on default headphones", "micforge loopback"),
            ("Test warm broadcast preset", "micforge loopback -p broadcast"),
            ("Specify headphones device explicitly", "micforge loopback --out \"Headphones\""),
        ],
        "tips": [
            "Use headphones to prevent feedback howling.",
            "Press [b] to toggle instant A/B comparison between raw mic and processed audio.",
        ],
    },
    "test": {
        "category": "🎙️ Live Real-Time Audio",
        "headline": "Record a voice sample, show before/after spectral metrics, and play both.",
        "examples": [
            ("Record 8 seconds test sample", "micforge test"),
            ("Quick 4-second test with podcast preset", "micforge test -p podcast -s 4"),
        ],
        "tips": [
            "Speak naturally with background noise (fans/typing) to judge noise reduction quality.",
        ],
    },
    "setup": {
        "category": "🛠️ Hardware, Calibration & Latency",
        "headline": "Interactive setup wizard to select microphone, virtual cable, and headphones.",
        "examples": [
            ("Launch interactive device setup", "micforge setup"),
        ],
        "tips": [
            "On Windows, always pick the WASAPI endpoints for the lowest latency and clock stability.",
        ],
    },
    "devices": {
        "category": "🛠️ Hardware, Calibration & Latency",
        "headline": "List all detected audio hardware, default endpoints, and host APIs.",
        "examples": [
            ("List standard devices", "micforge devices"),
            ("List all audio host APIs (WASAPI, MME, DirectSound)", "micforge devices --all"),
        ],
        "tips": [
            "Use device index numbers (e.g. --mic 1) or matching substrings (e.g. --mic Realtek).",
        ],
    },
    "calibrate": {
        "category": "🛠️ Hardware, Calibration & Latency",
        "headline": "Learn your room silence for 3–5s to configure noise floor and gate threshold.",
        "examples": [
            ("Perform standard room calibration", "micforge calibrate"),
            ("Calibrate for 6 seconds", "micforge calibrate -s 6"),
            ("Reset calibration back to factory baseline", "micforge calibrate --reset"),
        ],
        "tips": [
            "Stay silent and keep PC fans running normally while calibrating.",
        ],
    },
    "profile": {
        "category": "🛠️ Hardware, Calibration & Latency",
        "headline": "Manage voice-aware noise profiles and acoustic environments.",
        "examples": [
            ("List all available noise profiles", "micforge profile list"),
            ("Inspect fan noise profile details", "micforge profile show fan_on"),
            ("Switch active environment to mechanical keyboard", "micforge profile switch mechanical_keyboard"),
            ("Trigger immediate quiet-room relearn", "micforge profile relearn"),
        ],
        "tips": [
            "Continuous learning adapts slowly (≤1 dB/s) and never learns when you speak.",
            "Switch to 'fan_on' during heavy gaming sessions when PC fans ramp up.",
        ],
    },
    "latency": {
        "category": "🛠️ Hardware, Calibration & Latency",
        "headline": "Measure physical end-to-end loopback latency with logarithmic chirp.",
        "examples": [
            ("Run standard loopback latency benchmark", "micforge latency"),
            ("Benchmark specific mic and headphones", "micforge latency --mic 1 --out 2"),
        ],
        "tips": [
            "Uses a 60ms acoustic chirp cross-correlation with sub-millisecond precision.",
            "Provides tailored tuning tips for buffer sizes and Windows WASAPI settings.",
        ],
    },
    "settings": {
        "category": "🎛️ Tuning, Presets & Studio GUI",
        "headline": "Open native multi-tab Control Center (Simple/Advanced DSP, Hotkeys, Devices).",
        "examples": [
            ("Open Settings GUI", "micforge settings"),
            ("Alias command", "micforge hotkeys"),
        ],
        "tips": [
            "Features Simple Mode (4 macro knobs: Clarity, Warmth, Noise Removal, Room Cleanup).",
            "Includes one-click '📦 Export Diagnostics (.zip)' in the Routing & System tab.",
        ],
    },
    "hotkeys": {
        "category": "🎛️ Tuning, Presets & Studio GUI",
        "headline": "Alias for 'settings' — configure global hotkeys, audio cues, and startup.",
        "examples": [
            ("Open Settings and Hotkeys GUI", "micforge hotkeys"),
        ],
        "tips": [
            "Hotkeys work system-wide even when full-screen games or Discord have focus.",
        ],
    },
    "preset": {
        "category": "🎛️ Tuning, Presets & Studio GUI",
        "headline": "List, inspect, clone, export, and manage DSP chain presets.",
        "examples": [
            ("List all available presets", "micforge preset list"),
            ("Inspect 'discord' preset settings", "micforge preset show discord"),
            ("Clone 'gaming' into a new custom preset", "micforge preset save my_stream --from gaming"),
            ("Export custom preset to JSON file", "micforge preset export my_stream ~/preset.json"),
            ("Import preset from JSON file", "micforge preset import ~/preset.json"),
        ],
        "tips": [
            "Built-in presets (discord, gaming, podcast, noisy_room) are immutable and safe.",
        ],
    },
    "presets": {
        "category": "🎛️ Tuning, Presets & Studio GUI",
        "headline": "Quick alias to list presets or print preset details.",
        "examples": [
            ("List all presets", "micforge presets"),
            ("Print preset configuration", "micforge presets podcast"),
        ],
        "tips": [],
    },
    "process": {
        "category": "📁 Offline File Audio Processing",
        "headline": "Process an offline WAV audio file through the 11-stage DSP rack.",
        "examples": [
            ("Process raw recording", "micforge process raw.wav clean.wav"),
            ("Process with broadcast preset", "micforge process voice.wav out.wav -p broadcast"),
        ],
        "tips": [
            "Supports 16-bit, 24-bit, and 32-bit float WAV audio at any sample rate.",
        ],
    },
    "analyze": {
        "category": "📁 Offline File Audio Processing",
        "headline": "Analyze a WAV audio file (loudness, SNR, clarity, spectral profile).",
        "examples": [
            ("Inspect audio quality of recording", "micforge analyze recording.wav"),
        ],
        "tips": [
            "Reports RMS energy, peak ceiling, dynamic crest factor, and acoustic suggestions.",
        ],
    },
    "doctor": {
        "category": "🩺 Diagnostics, Models & System",
        "headline": "Health check for audio drivers, buffer speed, and redacted .zip diagnostics.",
        "examples": [
            ("Run doctor health assessment", "micforge doctor"),
            ("Export complete redacted diagnostic .zip bundle", "micforge doctor --export"),
            ("Export to specific file destination", "micforge doctor --export ~/diag.zip"),
        ],
        "tips": [
            "Automatically sanitizes passwords, tokens, API keys, URLs, and home paths.",
            "Links directly to docs/TROUBLESHOOTING.md for instant fix instructions.",
        ],
    },
    "model": {
        "category": "🩺 Diagnostics, Models & System",
        "headline": "Inspect active CoFi-Lite neural speech model or export PyTorch weights to ONNX.",
        "examples": [
            ("Show active neural model status & RAM footprint", "micforge model info"),
            ("Export PyTorch checkpoint to ONNX", "micforge model export checkpoint.tar model.onnx"),
        ],
        "tips": [
            "Micforge bundles an ultra-compact ~83k parameter CoFi-Lite model requiring <100MB RAM.",
        ],
    },
    "version": {
        "category": "🩺 Diagnostics, Models & System",
        "headline": "Display Micforge version, model status, and runtime environment.",
        "examples": [
            ("Show version info", "micforge version"),
        ],
        "tips": [],
    },
    "help": {
        "category": "🩺 Diagnostics, Models & System",
        "headline": "Display dynamic, categorized command reference and deep-dive usage guides.",
        "examples": [
            ("Show complete categorized command table", "micforge help"),
            ("Deep dive into a specific command", "micforge help run"),
            ("Inspect latency test options", "micforge help latency"),
        ],
        "tips": [
            "Type 'micforge help <command>' anytime to see full arguments, flags, and pro tips.",
        ],
    },
}

CATEGORIES_ORDER = [
    "🎙️ Live Real-Time Audio",
    "🛠️ Hardware, Calibration & Latency",
    "🎛️ Tuning, Presets & Studio GUI",
    "📁 Offline File Audio Processing",
    "🩺 Diagnostics, Models & System",
]


def _extract_parser_info(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Dynamically introspect an argparse ArgumentParser."""
    subparsers_action = None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparsers_action = action
            break

    commands_info: dict[str, dict[str, Any]] = {}
    if not subparsers_action:
        return commands_info

    for cmd_name, subparser in subparsers_action.choices.items():
        # Get positional arguments and options
        positionals: list[dict[str, Any]] = []
        options: list[dict[str, Any]] = []

        for action in subparser._actions:
            if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
                continue

            info = {
                "dest": action.dest,
                "help": action.help or "",
                "default": action.default if action.default != "==SUPPRESS==" else None,
                "choices": list(action.choices) if action.choices else None,
                "type": action.type.__name__ if hasattr(action.type, "__name__") else None,
            }

            if action.option_strings:
                info["flags"] = "/".join(action.option_strings)
                options.append(info)
            else:
                info["name"] = action.dest
                positionals.append(info)

        commands_info[cmd_name] = {
            "help": subparser.description or subparser.format_help().split("\n")[0] or "",
            "positionals": positionals,
            "options": options,
            "parser": subparser,
        }

    return commands_info


def run_help(command_name: str | None = None) -> int:
    """Entry point for `micforge help [command]`."""
    from ..cli import build_parser

    parser = build_parser()
    discovered_commands = _extract_parser_info(parser)

    # 1. SPECIFIC COMMAND DEEP DIVE
    if command_name:
        cmd_key = command_name.lower().strip()
        if cmd_key not in discovered_commands:
            console.print(f"\n[bold red]✖ Unknown command:[/] '[bold white]{command_name}[/]'")
            # Suggest matching commands
            matches = [c for c in discovered_commands if cmd_key in c or c in cmd_key]
            if matches:
                console.print(f"[yellow]  Did you mean:[/] [cyan]{', '.join(matches)}[/]")
            console.print("[dim]  Run [bold white]micforge help[/] to see all available commands.\n[/]")
            return 1

        _print_command_deep_dive(cmd_key, discovered_commands[cmd_key])
        return 0

    # 2. COMPLETE CATEGORIZED COMMAND DIRECTORY
    _print_all_commands_overview(discovered_commands)
    return 0


def _print_all_commands_overview(discovered_commands: dict[str, dict[str, Any]]) -> None:
    """Print the complete categorized reference table."""
    banner = Text()
    banner.append("🎙️  MICFORGE COMMAND REFERENCE  ", style="bold cyan")
    banner.append(f"v{__version__}\n", style="bold green")
    banner.append("Real-time local DSP rack, AI speech enhancement, diagnostics, and routing.", style="dim")
    console.print(Panel(banner, box=box.ROUNDED, border_style="cyan"))

    # Group commands into categories
    categorized: dict[str, list[str]] = {cat: [] for cat in CATEGORIES_ORDER}
    uncategorized: list[str] = []

    for cmd_name in sorted(discovered_commands.keys()):
        meta = COMMAND_METADATA.get(cmd_name)
        cat = meta.get("category") if meta else None
        if cat and cat in categorized:
            categorized[cat].append(cmd_name)
        else:
            uncategorized.append(cmd_name)

    if uncategorized:
        categorized["Other Commands"] = uncategorized

    for category, cmds in categorized.items():
        if not cmds:
            continue

        table = Table(
            title=f"[bold cyan]{category}[/]",
            title_justify="left",
            box=box.SIMPLE_HEAD,
            header_style="bold white",
            show_lines=False,
            padding=(0, 1),
        )
        table.add_column("Command", style="bold yellow", width=14)
        table.add_column("Description", style="white", width=42)
        table.add_column("Typical Example", style="dim cyan")

        for cmd_name in cmds:
            meta = COMMAND_METADATA.get(cmd_name, {})
            headline = meta.get("headline")
            if not headline:
                headline = discovered_commands[cmd_name]["help"] or "Execute command"

            examples = meta.get("examples", [])
            example_cmd = examples[0][1] if examples else f"micforge {cmd_name}"

            table.add_row(cmd_name, headline, example_cmd)

        console.print(table)
        console.print()

    # Helpful Footer
    console.print(
        "[bold cyan]Tip:[/] Type [bold white]micforge help <command>[/] "
        "(e.g. [green]micforge help run[/], [green]micforge help latency[/], [green]micforge help profile[/]) "
        "for detailed flags, syntax, and options.\n"
    )


def _print_command_deep_dive(cmd_name: str, info: dict[str, Any]) -> None:
    """Print an exhaustive, beautiful deep dive for a single command."""
    meta = COMMAND_METADATA.get(cmd_name, {})
    headline = meta.get("headline") or info.get("help") or "Command reference"
    category = meta.get("category", "General")

    # 1. Header Banner
    header = Text()
    header.append(f"COMMAND: micforge {cmd_name}\n", style="bold cyan")
    header.append(f"{headline}\n", style="bold white")
    header.append(f"Category: {category}", style="dim")
    console.print(Panel(header, box=box.ROUNDED, border_style="cyan"))

    # 2. Syntax / Usage Line
    positionals = info.get("positionals", [])
    options = info.get("options", [])

    syntax_parts = [f"micforge {cmd_name}"]
    if options:
        syntax_parts.append("[options]")
    for pos in positionals:
        syntax_parts.append(f"<{pos['dest']}>")

    syntax_str = " ".join(syntax_parts)
    console.print(f"[bold cyan]Usage:[/] [bold green]{escape(syntax_str)}[/]\n")

    # 3. Positional Arguments (if any)
    if positionals:
        pos_table = Table(title="[bold]Positional Arguments[/]", box=box.SIMPLE_HEAD, header_style="bold cyan")
        pos_table.add_column("Argument", style="bold yellow", width=18)
        pos_table.add_column("Choices / Type", style="dim", width=20)
        pos_table.add_column("Description")

        for pos in positionals:
            choices_str = ", ".join(map(str, pos["choices"])) if pos.get("choices") else (pos.get("type") or "string")
            pos_table.add_row(
                pos["dest"],
                f"[dim]{escape(choices_str)}[/]",
                escape(pos.get("help", "")),
            )

        console.print(pos_table)
        console.print()

    # 4. Optional Flags (if any)
    if options:
        opt_table = Table(title="[bold]Options & Flags[/]", box=box.SIMPLE_HEAD, header_style="bold cyan")
        opt_table.add_column("Flag", style="bold yellow", width=22)
        opt_table.add_column("Default", style="dim", width=16)
        opt_table.add_column("Description")

        for opt in options:
            default_val = str(opt["default"]) if opt.get("default") is not None and opt["default"] is not False else "-"
            opt_table.add_row(
                opt["flags"],
                f"[dim]{escape(default_val)}[/]",
                escape(opt.get("help", "")),
            )

        console.print(opt_table)
        console.print()

    # 5. Practical Copy-Paste Examples
    examples = meta.get("examples", [])
    if examples:
        console.print("[bold cyan]Practical Examples:[/]")
        for desc, ex_code in examples:
            console.print(f"  • [dim]{desc}:[/]")
            console.print(f"    [bold white]{escape(ex_code)}[/]")
        console.print()

    # 6. Pro Tips & Best Practices
    tips = meta.get("tips", [])
    if tips:
        console.print("[bold cyan]Pro Tips & Best Practices:[/]")
        for tip in tips:
            console.print(f"  💡 {escape(tip)}")
        console.print()

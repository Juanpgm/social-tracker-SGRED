"""The `social-tracker` command-line interface.

Every write needs a `--reason` and is attributed to an actor (`--actor`, else
`$SOCIAL_TRACKER_ACTOR`, else `cli:<OS user>`); both go to the audit log through the services.

Exit codes: 0 success, 1 the operation was refused (invalid input, conflict, unknown source,
storage failure...), 2 a usage error (missing option, unknown command). Refusals print one
`error: ...` line (plus the individual problems of a rejected import) on stderr and nothing on
stdout; results go to stdout. Messages come from the services and never echo values, keys or
paths the operator typed, so a pasted secret is not repeated back.

Only `social_tracker.bootstrap` knows about storage: this module never imports it.
"""

import os
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Annotated

import typer

from social_tracker.bootstrap import build_container
from social_tracker.control.domain import CollectionState, format_confidence
from social_tracker.control.service import SourceOverview, SystemOverview
from social_tracker.entrypoints.inputs import (
    parse_fields,
    read_import_text,
    resolve_actor,
    write_export,
)
from social_tracker.shared.errors import DomainError, ValidationError
from social_tracker.sources.service import ImportConflictError, ImportReport
from social_tracker.sources.yaml_io import MAX_FILE_BYTES

app = typer.Typer(
    help="Manage the sources of the social risk monitor and the analysis switches.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,  # a traceback must never print local secrets
)
sources_app = typer.Typer(help="Register and maintain sources.", no_args_is_help=True)
app.add_typer(sources_app, name="sources")
analysis_app = typer.Typer(
    help="Turn analysis on or off, for the whole system or for one source.", no_args_is_help=True
)
app.add_typer(analysis_app, name="analysis")

SourceId = Annotated[
    str,
    typer.Argument(metavar="SOURCE_ID", help="Source id: 3-64 lowercase letters, digits or _."),
]
Reason = Annotated[
    str,
    typer.Option("--reason", "-r", help="Why you are making this change (kept in the audit log)."),
]
Actor = Annotated[
    str | None,
    typer.Option(
        "--actor",
        help="Who is making this change (kept in the audit log). "
        "Default: $SOCIAL_TRACKER_ACTOR, else cli:<OS user>.",
    ),
]
Fields = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        metavar="KEY=VALUE",
        help="A source field, repeatable. VALUE is read as JSON when it parses (600, true, null, "
        '["a","b"]) and as plain text otherwise; quote JSON strings to force text.',
    ),
]

Switchable = Annotated[
    str | None,
    typer.Option("--source", help="Switch this source only (default: the whole system)."),
]

# One string per paragraph: the help text is shown as written, so no hard line breaks inside.
_IMPORT_HELP = (
    "Import sources from a YAML file, all or nothing.\n\n"
    "WARNING: an import replaces stored sources wholesale. An entry with the id of an existing "
    "source overwrites it, and every field the entry omits goes back to its default: an entry "
    "without `enabled` DISABLES a source that was enabled (enabled defaults to false) and "
    "omitted optional fields such as keywords are reset. Sources that are not in the file are "
    "left alone, and timestamps in the file are ignored. Run with --dry-run first to see what "
    "would be added, updated or left unchanged.\n\n"
    f"The file must be UTF-8 (a BOM is accepted) and at most {MAX_FILE_BYTES} bytes."
)
_ADD_HELP = (
    "Register a source (disabled until you enable it). The other fields depend on the type, "
    "for example:\n\n"
    'sources add city_news --type news_rss --name "City News" '
    "--set feed_url=https://example.org/rss.xml -r 'new feed'"
)
_ADD_RESERVED = frozenset({"id", "type", "name"})
_MAX_PROBLEMS_SHOWN = 20


@contextmanager
def _guard() -> Iterator[None]:
    """Turn a refused operation into one `error:` line on stderr and exit code 1."""
    try:
        yield
    except DomainError as exc:
        typer.echo(f"error: {exc.message}", err=True)
        if isinstance(exc, ImportConflictError):
            for problem in exc.problems[:_MAX_PROBLEMS_SHOWN]:
                typer.echo(f"  entry {problem.index}: {problem.message}", err=True)
            if len(exc.problems) > _MAX_PROBLEMS_SHOWN:
                typer.echo(f"  and {len(exc.problems) - _MAX_PROBLEMS_SHOWN} more", err=True)
        raise typer.Exit(1) from None


def _on_off(flag: bool) -> str:
    return "on" if flag else "off"


def _render_sources(rows: Sequence[SourceOverview]) -> str:
    header = ("ID", "TYPE", "ENABLED", "STATE", "ANALYSIS", "NAME")
    body = [
        (
            o.source.id,
            o.source.type,
            "yes" if o.source.enabled else "no",
            o.state.value,
            _on_off(o.analysis),
            o.source.name,
        )
        for o in rows
    ]
    widths = [max(len(row[i]) for row in (header, *body)) for i in range(5)]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=False))
        + f"  {row[5]}"
        for row in (header, *body)
    )


def _render_status(overview: SystemOverview) -> str:
    rows = overview.sources
    by_state = Counter(o.state for o in rows)
    states = ", ".join(f"{s.value} {by_state[s]}" for s in CollectionState if by_state[s])
    return "\n".join(
        [
            f"global analysis: {_on_off(overview.global_analysis)}",
            f"alert threshold: {format_confidence(overview.alert_min_confidence)}",
            f"sources: {len(rows)} total, {sum(o.source.enabled for o in rows)} enabled",
            f"collection states: {states or 'none'}",
            f"effective analysis: {sum(o.effective for o in rows)} source(s)",
        ]
    )


def _render_import(report: ImportReport) -> str:
    counts = (len(report.added), len(report.updated), len(report.unchanged))
    if report.dry_run:
        head = "dry run, nothing written: would add {}, update {}, leave {} unchanged"
    else:
        head = "imported: {} added, {} updated, {} unchanged"
    lines = [head.format(*counts)]
    for label, found in (
        ("added", report.added),
        ("updated", report.updated),
        ("unchanged", report.unchanged),
    ):
        if found:
            lines.append(f"{label}: {', '.join(found)}")
    return "\n".join(lines)


@app.callback()
def _main() -> None:
    # Source names may hold characters a piped Windows console cannot encode: escape them
    # instead of dying with a traceback in the middle of a listing. Only stdout needs it:
    # error messages never carry user text.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="backslashreplace")


# ---- sources ----


@sources_app.command("add", help=_ADD_HELP)
def sources_add(
    source_id: SourceId,
    source_type: Annotated[
        str,
        typer.Option("--type", help="One of news_rss, instagram, live_stream, recorded_audio."),
    ],
    name: Annotated[str, typer.Option("--name", help="Display name, unique ignoring case.")],
    reason: Reason,
    actor: Actor = None,
    fields: Fields = None,
) -> None:
    with _guard():
        data = parse_fields(fields or [], reserved=_ADD_RESERVED)
        data.update({"id": source_id, "type": source_type, "name": name})
        with closing(build_container()) as services:
            source = services.sources.add(
                data, actor=resolve_actor(actor, os.environ), reason=reason
            )
    typer.echo(f"added {source.id} ({source.type})")


@sources_app.command("list")
def sources_list() -> None:
    """List every source with its collection state and analysis switch."""
    with _guard(), closing(build_container()) as services:
        rows = services.control.overview().sources
    typer.echo(_render_sources(rows) if rows else "No sources registered.")


@sources_app.command("edit")
def sources_edit(
    source_id: SourceId, reason: Reason, actor: Actor = None, fields: Fields = None
) -> None:
    """Change fields of a source. Id, type and enabled cannot be edited."""
    with _guard():
        changes = parse_fields(fields or [])
        if not changes:
            raise ValidationError("nothing to change: give at least one --set KEY=VALUE")
        with closing(build_container()) as services:
            change = services.sources.edit(
                source_id, changes, actor=resolve_actor(actor, os.environ), reason=reason
            )
    typer.echo(f"updated {source_id}" if change.changed else f"no changes for {source_id}")


def _set_enabled(source_id: str, enabled: bool, reason: str, actor: str | None) -> None:
    with _guard(), closing(build_container()) as services:
        who = {"actor": resolve_actor(actor, os.environ), "reason": reason}
        registry = services.sources
        change = (registry.enable if enabled else registry.disable)(source_id, **who)
    word = "enabled" if enabled else "disabled"
    typer.echo(f"{word} {source_id}" if change.changed else f"{source_id} is already {word}")


@sources_app.command("enable")
def sources_enable(source_id: SourceId, reason: Reason, actor: Actor = None) -> None:
    """Enable a source (its collection state moves from disabled to idle)."""
    _set_enabled(source_id, True, reason, actor)


@sources_app.command("disable")
def sources_disable(source_id: SourceId, reason: Reason, actor: Actor = None) -> None:
    """Disable a source (its collection stops and its failure count resets)."""
    _set_enabled(source_id, False, reason, actor)


@sources_app.command("delete")
def sources_delete(source_id: SourceId, reason: Reason, actor: Actor = None) -> None:
    """Delete a source and its runtime state. The audit events are kept."""
    with _guard(), closing(build_container()) as services:
        services.sources.delete(source_id, actor=resolve_actor(actor, os.environ), reason=reason)
    typer.echo(f"deleted {source_id}")


@sources_app.command("import", help=_IMPORT_HELP)
def sources_import(
    file: Annotated[
        Path,
        typer.Argument(metavar="FILE", help="YAML file with a top-level `sources:` list."),
    ],
    reason: Annotated[
        str | None,
        typer.Option(
            "--reason", "-r", help="Why you are importing (audit log). Required unless --dry-run."
        ),
    ] = None,
    actor: Actor = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would change without writing anything.")
    ] = False,
) -> None:
    if reason is None and not dry_run:
        raise typer.BadParameter("required unless --dry-run", param_hint="--reason")
    with _guard():
        text = read_import_text(file)
        with closing(build_container()) as services:
            report = services.sources.import_sources(
                text, actor=resolve_actor(actor, os.environ), reason=reason, dry_run=dry_run
            )
    typer.echo(_render_import(report))


@sources_app.command("export")
def sources_export(
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write to this file instead of stdout.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite the --output file if it exists.")
    ] = False,
) -> None:
    """Export every source as YAML (secret references only, never secrets) to stdout or a file."""
    with _guard(), closing(build_container()) as services:
        text = services.sources.export_sources()
        if output is None:
            typer.echo(text.encode("utf-8"), nl=False)
        else:
            write_export(output, text, force=force)
    if output is not None:
        typer.echo(f"exported to {output}")


# ---- analysis and status ----


def _switch(on: bool, source_id: str | None, reason: str, actor: str | None) -> None:
    with _guard(), closing(build_container()) as services:
        who = {"actor": resolve_actor(actor, os.environ), "reason": reason}
        if source_id is None:
            subject = "global analysis"
            change = services.control.set_global_analysis(on, **who)
        else:
            subject = f"analysis for {source_id}"
            change = services.control.set_source_analysis(source_id, on, **who)
    state = _on_off(on)
    typer.echo(f"{subject} is now {state}" if change.changed else f"{subject} is already {state}")


@analysis_app.command("on")
def analysis_on(reason: Reason, actor: Actor = None, source: Switchable = None) -> None:
    """Turn analysis on. Nothing runs until the system AND the source switches are on and the
    source is enabled."""
    _switch(True, source, reason, actor)


@analysis_app.command("off")
def analysis_off(reason: Reason, actor: Actor = None, source: Switchable = None) -> None:
    """Turn analysis off (the system starts with it off)."""
    _switch(False, source, reason, actor)


@app.command("status")
def status() -> None:
    """Show the analysis switches, the alert threshold and a summary of the sources."""
    with _guard(), closing(build_container()) as services:
        overview = services.control.overview()
    typer.echo(_render_status(overview))

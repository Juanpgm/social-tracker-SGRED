"""The `social-tracker` command-line interface.

Every write needs a `--reason` and is attributed to an actor (`--actor`, else
`$SOCIAL_TRACKER_ACTOR`, else `cli:<OS user>`); both go to the audit log through the services.

Exit codes: 0 success, 1 the operation was refused (invalid input, conflict, unknown source,
storage failure...), 2 a usage error (missing option, unknown command), 70 an unexpected failure,
that is a bug (BSD `EX_SOFTWARE`). Refusals print one `error: ...` line (plus the individual
problems of a rejected import) on stderr and nothing on stdout; results go to stdout. An
unexpected failure prints `error: unexpected failure (<ExceptionType>); this is a bug` and the
frames where it happened, never the exception message or local variables.

Messages of refusals come from the services and do not echo the values or keys the operator
typed (`--set`, reasons, actors), so a pasted secret is not repeated back. Paths are different:
an operator-chosen path may appear in what `status` and `sources export -o` print, with control
and hidden characters escaped, and never in an error message.

Only `social_tracker.bootstrap` knows about storage: this module never imports it.
"""

import logging
import math
import os
import sys
import traceback
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Annotated

import typer

from social_tracker.bootstrap import build_container, build_worker
from social_tracker.collector.service import CollectOutcome
from social_tracker.collector.worker import DEFAULT_TICK_S, PassSummary
from social_tracker.control.domain import CollectionState, format_confidence
from social_tracker.control.service import SourceOverview, SystemOverview
from social_tracker.entrypoints.inputs import (
    parse_fields,
    read_import_text,
    resolve_actor,
    write_export,
)
from social_tracker.entrypoints.signals import stop_on_signals
from social_tracker.items.domain import DEFAULT_LIST_LIMIT, MAX_LIST_LIMIT, Item
from social_tracker.shared.errors import DomainError, ValidationError
from social_tracker.shared.text import escape_unprintable
from social_tracker.sources.service import ImportConflictError, ImportReport
from social_tracker.sources.yaml_io import MAX_FILE_BYTES

EXIT_REFUSED = 1
EXIT_UNEXPECTED = 70  # sysexits.h EX_SOFTWARE: an internal error, i.e. a bug

app = typer.Typer(
    help="Manage the sources of the social risk monitor and the analysis switches.\n\n"
    "Exit status: 0 success, 1 refused (invalid input, conflict, storage failure...), "
    "2 usage error, 70 unexpected failure (a bug).",
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
items_app = typer.Typer(help="Look at the items collected from the sources.", no_args_is_help=True)
app.add_typer(items_app, name="items")

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
    """Turn a refused operation into one `error:` line on stderr and exit code 1, and any other
    failure (a bug) into a one-line report, the frames without values, and exit code 70."""
    try:
        yield
    except (typer.Exit, typer.Abort, typer.TyperException):
        raise  # click's own control flow: a usage error, --help, an interrupted prompt
    except DomainError as exc:
        typer.echo(f"error: {exc.message}", err=True)
        if isinstance(exc, ImportConflictError):
            for problem in exc.problems[:_MAX_PROBLEMS_SHOWN]:
                typer.echo(f"  entry {problem.index}: {problem.message}", err=True)
            if len(exc.problems) > _MAX_PROBLEMS_SHOWN:
                typer.echo(f"  and {len(exc.problems) - _MAX_PROBLEMS_SHOWN} more", err=True)
        raise typer.Exit(EXIT_REFUSED) from None
    except Exception as exc:
        typer.echo(f"error: unexpected failure ({type(exc).__name__}); this is a bug", err=True)
        traceback.print_tb(exc.__traceback__, file=sys.stderr)
        raise typer.Exit(EXIT_UNEXPECTED) from None


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


def _render_status(overview: SystemOverview, db_path: Path) -> str:
    rows = overview.sources
    by_state = Counter(o.state for o in rows)
    states = ", ".join(f"{s.value} {by_state[s]}" for s in CollectionState if by_state[s])
    return "\n".join(
        [
            f"database: {escape_unprintable(str(db_path))}",
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


@sources_app.command("reset")
def sources_reset(source_id: SourceId, reason: Reason, actor: Actor = None) -> None:
    """Put a FAILED source back to work (idle, failures forgotten, stale error cleared). A source
    that is not failed is left as it is."""
    with _guard(), closing(build_container()) as services:
        change = services.control.reset_if_failed(
            source_id, actor=resolve_actor(actor, os.environ), reason=reason
        )
    typer.echo(f"reset {source_id}" if change.changed else f"{source_id} is not failed")


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
        typer.echo(f"exported to {escape_unprintable(str(output))}")


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
        overview, db_path = services.control.overview(), services.db_path
    typer.echo(_render_status(overview, db_path))


# ---- web UI ----

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


@app.command("serve")
def serve(
    host: Annotated[
        str, typer.Option("--host", help="Address to listen on. Loopback only unless allowed.")
    ] = DEFAULT_HOST,
    port: Annotated[int, typer.Option("--port", min=1, max=65535, help="TCP port.")] = DEFAULT_PORT,
    allow_non_loopback: Annotated[
        bool,
        typer.Option(
            "--allow-non-loopback",
            help="WARNING: let --host be an address other machines can reach. The UI has no "
            "login: anyone who can connect can change your sources.",
        ),
    ] = False,
    allowed_host: Annotated[
        list[str] | None,
        typer.Option(
            "--allowed-host",
            help="A name clients use to reach the UI besides localhost and --host (repeatable), "
            "for example the name behind a wildcard --host 0.0.0.0. The UI speaks plain HTTP: "
            "traffic is not encrypted and its cookie has no Secure flag, so use it only on a "
            "network you trust or behind a TLS proxy you control.",
        ),
    ] = None,
) -> None:
    """Run the local web UI to manage sources (it never publishes or sends anything)."""
    # Imported here so the other commands do not load the web stack.
    from social_tracker.entrypoints.web.security import is_loopback
    from social_tracker.entrypoints.web.server import run_server

    with _guard():
        if allow_non_loopback and not is_loopback(host):
            typer.echo(
                "warning: the UI has no login and its traffic is not encrypted; every machine "
                "that can reach this address can change your sources",
                err=True,
            )
        run_server(
            host,
            port,
            allow_non_loopback=allow_non_loopback,
            allowed_hosts=tuple(allowed_host or ()),
        )


# ---- items ----

ItemLimit = Annotated[
    int,
    typer.Option(
        "--limit", min=1, max=MAX_LIST_LIMIT, help="How many items to show (newest first)."
    ),
]
_LABEL_CHARS = 100
_NO_ITEMS = "No items."


def _item_label(item: Item) -> str:
    """One line to recognise an item by: its title, else the start of its text. Item text is
    untrusted, so it is flattened to one line, shortened, and any control or hidden character
    (a terminal escape sequence, a right-to-left override...) is shown as an escape."""
    first_line = item.text.split("\n", 1)[0]
    flat = " ".join((item.title or first_line).split())
    if len(flat) > _LABEL_CHARS:
        flat = flat[: _LABEL_CHARS - 3].rstrip() + "..."
    return escape_unprintable(flat)


def _render_items(items: Sequence[Item]) -> str:
    stamps = [f"{item.published_at:%Y-%m-%d %H:%M}" for item in items]
    width = max([len("SOURCE"), *(len(item.source_id) for item in items)])
    lines = [f"{'PUBLISHED':<16}  {'SOURCE':<{width}}  TITLE"]
    lines += [
        f"{stamp:<16}  {item.source_id:<{width}}  {_item_label(item)}"
        for stamp, item in zip(stamps, items, strict=True)
    ]
    return "\n".join(lines)


@items_app.command("list")
def items_list(
    source: Annotated[
        str | None, typer.Option("--source", help="Only items of this source id.")
    ] = None,
    limit: ItemLimit = DEFAULT_LIST_LIMIT,
) -> None:
    """List collected items, newest first."""
    with _guard(), closing(build_container()) as services:
        found = services.items.list_items(source_id=source, limit=limit)
    typer.echo(_render_items(found) if found else _NO_ITEMS)


@items_app.command("search")
def items_search(
    query: Annotated[
        str,
        typer.Argument(
            metavar="QUERY",
            help="Words to find in titles and texts. Every word must match as a whole word, "
            "ignoring case and accents. Operators such as AND, OR, NEAR, quotes and * are "
            "ordinary text here, not search syntax (put -- before a query starting with -).",
        ),
    ],
    limit: ItemLimit = DEFAULT_LIST_LIMIT,
) -> None:
    """Search collected items by words, newest first."""
    with _guard(), closing(build_container()) as services:
        found = services.items.search(query, limit=limit)
    typer.echo(_render_items(found) if found else "No items found.")


# ---- worker ----


@contextmanager
def _logs_to_stderr() -> Iterator[None]:
    """Send the application's log records to the CURRENT stderr for the duration of a run.
    Records name sources, counts and fixed reasons, never URLs, item text or secrets."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger = logging.getLogger("social_tracker")
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def _render_pass(summary: PassSummary, *, reset_analysis: bool = True) -> str:
    if summary.global_off and reset_analysis:
        return (
            "global analysis is off: nothing was collected. A starting worker always switches it "
            "off, so a plain `run --once` can never collect. To check a collection, turn analysis "
            "on with `social-tracker analysis on` and run `social-tracker run --once "
            "--no-reset-analysis`; or leave `run` running and turn analysis on from another "
            "terminal."
        )
    if summary.global_off:
        return (
            "global analysis is off: nothing was collected. Turn it on with "
            "`social-tracker analysis on` and run this command again."
        )
    counts = Counter(result.outcome for result in summary.results)
    skipped = sum(n for outcome, n in counts.items() if outcome.value.startswith("skipped"))
    new = sum(result.new for result in summary.results)
    return (
        f"pass complete: {counts[CollectOutcome.COLLECTED]} collected "
        f"({new} new item(s)), {counts[CollectOutcome.NOT_MODIFIED]} not modified, "
        f"{counts[CollectOutcome.FAILED]} failed, {counts[CollectOutcome.DISCARDED]} discarded, "
        f"{skipped} skipped, {summary.errors} error(s)"
    )


@app.command("run")
def run(
    once: Annotated[
        bool, typer.Option("--once", help="Do one scheduling pass and exit (for diagnostics).")
    ] = False,
    interval_tick: Annotated[
        float,
        typer.Option(
            "--interval-tick",
            min=0.01,
            max=3600,
            metavar="SECONDS",
            help="How long to wait between scheduling passes.",
        ),
    ] = DEFAULT_TICK_S,
    no_reset_analysis: Annotated[
        bool,
        typer.Option(
            "--no-reset-analysis",
            help="With --once only: do not switch global analysis off at start, so one "
            "diagnostic pass can collect after `analysis on`. The long-running worker "
            "always resets it.",
        ),
    ] = False,
) -> None:
    """Run the collection worker until SIGINT/SIGTERM.

    Starting it switches global analysis OFF (audited as system:worker); turn it on afterwards
    with `social-tracker analysis on`. Each pass collects the enabled RSS sources that are due
    (their poll interval, 600 s by default), one at a time, and removes items past the retention
    period. Stopping lets the collection in flight finish (a second Ctrl+C or SIGTERM exits at
    once with status 130 or 143). Run one worker per database.
    """
    if not math.isfinite(interval_tick):
        raise typer.BadParameter("must be a finite number", param_hint="--interval-tick")
    if no_reset_analysis and not once:
        raise typer.BadParameter(
            "only valid with --once (the long-running worker always switches analysis off at "
            "start)",
            param_hint="--no-reset-analysis",
        )
    with _guard(), _logs_to_stderr(), closing(build_worker(tick_s=interval_tick)) as built:
        if not once:
            typer.echo(
                "worker running; global analysis is switched off at start, turn it on with "
                "`social-tracker analysis on`; press Ctrl+C to stop"
            )
        with stop_on_signals(built.worker.request_stop):
            summary = built.worker.run(once=once, reset_analysis=not no_reset_analysis)
    if once and summary is not None:
        typer.echo(_render_pass(summary, reset_analysis=not no_reset_analysis))
    elif not once:
        typer.echo("worker stopped")

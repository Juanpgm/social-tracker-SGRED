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
import threading
import traceback
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Annotated

import typer

from social_tracker.bootstrap import build_catalog, build_container, build_llm, build_worker
from social_tracker.catalog.service import CatalogStatus, Outcome, RefreshReport
from social_tracker.classification.domain import Classification, ClassificationStats
from social_tracker.classification.service import DEFAULT_BATCH, RECLASSIFIABLE, ClassifyReport
from social_tracker.classification.yaml_io import MAX_CONFIG_BYTES
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
from social_tracker.items.domain import DEFAULT_LIST_LIMIT, MAX_LIST_LIMIT, Item, ItemStatus
from social_tracker.llm.domain import EscalationCounts, EscalationState, Judgment
from social_tracker.llm.service import EscalationReport, LlmStatus, PingResult
from social_tracker.shared.errors import DomainError, ValidationError
from social_tracker.shared.text import escape_unprintable
from social_tracker.sources.service import ImportConflictError, ImportReport
from social_tracker.sources.yaml_io import MAX_FILE_BYTES

EXIT_REFUSED = 1
EXIT_NOTHING_ELIGIBLE = 3  # `classify run` with no source it may classify; `catalog refresh`
# while global analysis is off
EXIT_UNEXPECTED = 70  # sysexits.h EX_SOFTWARE: an internal error, i.e. a bug

app = typer.Typer(
    help="Manage the sources of the social risk monitor and the analysis switches.\n\n"
    "Exit status: 0 success, 1 refused (invalid input, conflict, storage failure...), "
    "2 usage error, 3 `classify run` found no eligible source or `catalog refresh` ran with "
    "analysis off, 70 unexpected failure (a bug).",
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
classify_app = typer.Typer(
    help="Run the classification cascade over collected items (CPU first, then the local "
    "language model for what it cannot settle) and look at its verdicts.",
    no_args_is_help=True,
)
app.add_typer(classify_app, name="classify")
topics_app = typer.Typer(
    help="Show and change the vocabulary that decides what the classifier looks at.",
    no_args_is_help=True,
)
app.add_typer(topics_app, name="topics")
catalog_app = typer.Typer(
    help="Refresh and inspect the cache of official earthquake catalogs (USGS and EMSC).",
    no_args_is_help=True,
)
app.add_typer(catalog_app, name="catalog")
llm_app = typer.Typer(
    help="Look at the local language model that judges what the rules cannot settle.",
    no_args_is_help=True,
)
app.add_typer(llm_app, name="llm")

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
ShowIds = Annotated[
    bool, typer.Option("--ids", help="Show the id of each item first (what `classify show` takes).")
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


def _render_items(items: Sequence[Item], *, ids: bool = False) -> str:
    stamps = [f"{item.published_at:%Y-%m-%d %H:%M}" for item in items]
    width = max([len("SOURCE"), *(len(item.source_id) for item in items)])
    id_width = max([len("ID"), *(len(item.id) for item in items)])
    lead = f"{'ID':<{id_width}}  " if ids else ""
    lines = [f"{lead}{'PUBLISHED':<16}  {'SOURCE':<{width}}  TITLE"]
    for stamp, item in zip(stamps, items, strict=True):
        lead = f"{item.id:<{id_width}}  " if ids else ""
        lines.append(f"{lead}{stamp:<16}  {item.source_id:<{width}}  {_item_label(item)}")
    return "\n".join(lines)


@items_app.command("list")
def items_list(
    source: Annotated[
        str | None, typer.Option("--source", help="Only items of this source id.")
    ] = None,
    limit: ItemLimit = DEFAULT_LIST_LIMIT,
    ids: ShowIds = False,
) -> None:
    """List collected items, newest first."""
    with _guard(), closing(build_container()) as services:
        found = services.items.list_items(source_id=source, limit=limit)
    typer.echo(_render_items(found, ids=ids) if found else _NO_ITEMS)


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
    ids: ShowIds = False,
) -> None:
    """Search collected items by words, newest first."""
    with _guard(), closing(build_container()) as services:
        found = services.items.search(query, limit=limit)
    typer.echo(_render_items(found, ids=ids) if found else "No items found.")


# ---- classification ----

_NO_ELIGIBLE = (
    "no source is eligible: global analysis must be on, and a source must be enabled with its "
    "own analysis switch on (see `social-tracker status` and `sources list`)"
)
_NO_VERDICTS = "No classifications yet."


def _render_classify(report: ClassifyReport) -> str:
    line = (
        f"classified {report.classified} item(s), {report.out_of_scope} out of scope, "
        f"{report.skipped} skipped, {report.errors + report.storage_errors} error(s); "
        f"{report.needs_llm} queued for the LLM"
    )
    if report.failed:
        line += f"; {report.failed} parked as classification_failed"
    if report.candidate_cap_hits:
        line += (
            f"; the near-duplicate candidate limit was reached {report.candidate_cap_hits} time(s)"
        )
    return line


@classify_app.command("run")
def classify_run(
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            max=MAX_LIST_LIMIT,
            help="How many pending items to classify, oldest published first.",
        ),
    ] = DEFAULT_BATCH,
) -> None:
    """Classify the pending items of the eligible sources (CPU only, nothing leaves the machine).

    Needs global analysis on and a source that is enabled with its analysis switch on. Items of
    other sources stay as they are. The worker does this after every collection pass; this
    command is for a one-off run.

    Exit status: 0 when it ran (also when nothing was pending), 1 when a storage error stopped
    the run (run it again to resume), 3 when no source is eligible so nothing could be done. An
    item that fails is retried by the next run and parked as classification_failed after three
    consecutive errors; that is reported but does not change the exit status."""
    with _guard(), closing(build_container()) as services:
        report = services.classifier.classify_pending(limit)
    if report.eligible_sources == 0:
        typer.echo(_NO_ELIGIBLE, err=True)
        raise typer.Exit(EXIT_NOTHING_ELIGIBLE)
    typer.echo(_render_classify(report))
    if report.storage_errors:
        typer.echo("error: a storage error stopped the run; run it again to resume", err=True)
        raise typer.Exit(EXIT_REFUSED)
    if report.errors:
        typer.echo(
            f"warning: {report.errors} item(s) could not be classified; they are tried again "
            "on the next run and parked after repeated errors (see `classify stats`)",
            err=True,
        )


def _render_escalation(report: EscalationReport, waiting: int) -> str:
    line = (
        f"judged {report.judged} item(s), {report.invalid} invalid, "
        f"{report.reused} settled from an original, {report.skipped} skipped, "
        f"{report.failed} failed (retried later), {report.parked} parked, "
        f"{report.dropped} dropped; {waiting} still waiting"
    )
    if report.cap_reached:
        line += "; the hourly cap was reached, the rest waits for the next hour"
    if report.stopped:
        line += "; stopped on request"
    if report.busy:
        line += "; another worker holds the model, try again later"
    if report.out_of_time:
        line += "; the time budget of the pass ran out, the rest waits for the next one"
    extras = [
        f"{report.requeued} parked item(s) back in the queue" if report.requeued else "",
        f"{report.released} copy(ies) released from an original that is gone"
        if report.released
        else "",
        f"{report.contradicted} no_risk answer(s) capped for contradicting earlier evidence"
        if report.contradicted
        else "",
    ]
    return "; ".join([line, *[e for e in extras if e]])


@classify_app.command("escalate")
def classify_escalate(
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            max=1000,
            help="How many waiting items to send to the language model at most "
            "(default: SOCIAL_TRACKER_LLM_BATCH, 20).",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Say how many items would be sent; send and change nothing."
        ),
    ] = False,
) -> None:
    """Send the items the CPU stages could not settle to the local language model (one-shot).

    The worker does this after every pass; this command is for a one-off run. The model server
    must run on this machine (SOCIAL_TRACKER_LLM_BASE_URL, loopback only) and the text of the
    items goes nowhere else. Needs global analysis on and an eligible source. One judgment is
    in flight at a time: a worker that is judging makes this wait or give up. A judge that is
    down leaves the CPU verdicts as they are and the items are tried again later with a growing
    delay. At most the hourly cap (SOCIAL_TRACKER_LLM_HOURLY_CAP) are judged per hour.

    Exit status: 0 when it ran (also when nothing was waiting), 1 when a storage error stopped
    the run or the model server failed after some items were judged, 3 when it could do nothing:
    no source is eligible, or nothing could be judged because the model server was not reachable
    (or is not Ollama, or is resting after a failure), another process holds the model, or the
    run was stopped. `--dry-run` sends nothing: it counts the items that would be sent (not those
    that settle without a call) and the items a full queue would drop."""
    stop = threading.Event()
    with _guard(), closing(build_llm(stop=stop)) as built:
        with stop_on_signals(stop.set):
            report = built.service.escalate(limit, dry_run=dry_run)
        waiting = built.service.counts().queued
    if report.eligible_sources == 0:
        typer.echo(_NO_ELIGIBLE, err=True)
        raise typer.Exit(EXIT_NOTHING_ELIGIBLE)
    if dry_run:
        typer.echo(f"dry run: {report.candidates} item(s) would be sent to the language model")
        if report.cap_reached:
            typer.echo("the hourly cap would stop the rest until the next hour")
        if report.pending_drops:
            typer.echo(
                f"{report.pending_drops} item(s) would be dropped from the full queue "
                "(the lowest priority first, the oldest among equals)"
            )
        return
    typer.echo(_render_escalation(report, waiting))
    if report.storage_errors:
        typer.echo("error: a storage error stopped the run; run it again to resume", err=True)
        raise typer.Exit(EXIT_REFUSED)
    if report.unreachable:
        typer.echo(
            "error: the language model server is not reachable, is not Ollama or has no usable "
            "model (see `social-tracker llm status`); the items stay queued and are tried again "
            "later",
            err=True,
        )
    if report.nothing_could_be_judged:
        raise typer.Exit(EXIT_NOTHING_ELIGIBLE)
    if report.unreachable:
        raise typer.Exit(EXIT_REFUSED)


def _render_verdict(c: Classification) -> str:
    flags = [
        name
        for name, on in (
            ("needs_llm", c.needs_llm),
            ("low_confidence", c.low_confidence),
            ("truncated", c.truncated),
        )
        if on
    ]
    topics = ", ".join(t.value for t in c.topics) or "none"
    signals = ", ".join(f"{s.name} ({s.weight:g})" for s in c.signals) or "none"
    lines = [
        f"item: {c.item_id}",
        f"label: {c.label.value}",
        f"confidence: {c.confidence:.3f}",
        f"stage: {c.stage.value}",
        f"rationale: {escape_unprintable(c.rationale)}",
        f"topics: {topics}",
        f"signals: {signals}",
        f"flags: {', '.join(flags) or 'none'}",
    ]
    if c.scope_reason is not None:
        lines.append(f"reason: {c.scope_reason.value}")
    if c.duplicate_of is not None:
        waiting = " (its original still waits for the LLM)" if c.awaiting_original else ""
        lines.append(f"duplicate of: {c.duplicate_of}{waiting}")
    if c.near_of is not None:
        lines.append(f"near copy of: {c.near_of}")
    if c.recycled_of is not None:
        lines.append(f"recycled of: {c.recycled_of}")
    if c.settle_reason is not None and c.settled_at is not None:
        lines.append(f"settled: {c.settle_reason} at {c.settled_at.isoformat()}")
    lines += [
        f"priority: {c.priority:.4f}",
        f"catalog: {c.catalog_status}",
        f"source credibility used: {c.source_credibility_used}",
        f"classified at: {c.classified_at.isoformat()}",
        f"classifier version: {c.classifier_version}",
    ]
    return "\n".join(lines)


def _render_llm_lines(judgment: Judgment | None, state: EscalationState | None) -> list[str]:
    """What the LLM stage kept about an item, for `classify show`."""
    lines = []
    if judgment is not None:
        version = escape_unprintable(judgment.prompt_version)
        if judgment.outcome == "judged":
            model = escape_unprintable(judgment.llm_model or "unknown")
            lines.append(f"llm: judged by {model} (prompt {version})")
        elif judgment.outcome == "invalid":
            lines.append(
                "llm: two invalid answers; the verdict of the earlier stage stands "
                f"(prompt {version})"
            )
        else:
            how = (
                f"judged by {escape_unprintable(judgment.llm_model)}"
                if judgment.llm_model
                else "its original had no valid answer"
            )
            lines.append(f"llm: reused from {judgment.reused_from} ({how})")
    if state is not None:
        detail = [f"attempt {state.attempts}"]
        if state.next_attempt_at is not None:
            detail.append(f"next attempt {state.next_attempt_at.isoformat()}")
        if state.last_error:
            detail.append(f"last error {escape_unprintable(state.last_error)}")
        lines.append(f"escalation: {state.state} ({', '.join(detail)})")
    return lines


@classify_app.command("show")
def classify_show(
    item_id: Annotated[str, typer.Argument(metavar="ITEM_ID", help="Id of a classified item.")],
) -> None:
    """Show the verdict of one item: label, confidence, deciding stage, rationale and flags,
    and, if it went through the language model, which model and prompt version judged it."""
    with _guard(), closing(build_container()) as services:
        verdict = services.classifier.get(item_id)
        judgment = services.escalation.judgment(verdict.item_id)
        state = services.escalation.state(verdict.item_id)
    typer.echo("\n".join([_render_verdict(verdict), *_render_llm_lines(judgment, state)]))


@classify_app.command("reclassify")
def classify_reclassify(
    status: Annotated[
        str,
        typer.Option(
            "--status",
            help="Which items to send back: out_of_scope (after changing the vocabulary) or "
            "classification_failed (to retry them).",
        ),
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit", min=1, max=MAX_LIST_LIMIT, help="How many items to send back at most."
        ),
    ] = DEFAULT_BATCH,
) -> None:
    """Send items back to the queue so that the next run classifies them again.

    Their verdict is forgotten and they become `new`; the next `classify run` (or the worker's
    next pass) judges them with the vocabulary and the rules in force. Use it after
    `topics set-file` for the items the old vocabulary called out of scope."""
    try:
        chosen = ItemStatus(status)
    except ValueError:
        raise typer.BadParameter(
            "must be out_of_scope or classification_failed", param_hint="--status"
        ) from None
    if chosen not in RECLASSIFIABLE:
        raise typer.BadParameter(
            "must be out_of_scope or classification_failed", param_hint="--status"
        )
    with _guard(), closing(build_container()) as services:
        count = services.classifier.reclassify(chosen, limit)
    typer.echo(f"{count} item(s) sent back to the queue; `classify run` classifies them again")


def _render_stats(stats: ClassificationStats, llm: EscalationCounts | None = None) -> str:
    lines = [f"classifications: {stats.total}", "by label:"]
    lines += [f"  {name} {count}" for name, count in stats.by_label]
    if stats.out_of_scope_by_reason:
        lines.append("out of scope by reason:")
        lines += [f"  {name} {count}" for name, count in stats.out_of_scope_by_reason]
    lines += [
        f"queued for the LLM: {stats.needs_llm}",
        f"low confidence: {stats.low_confidence}",
        f"duplicates: {stats.duplicates}",
        f"waiting for their original's judgment: {stats.awaiting_original}",
        f"waiting for an original that is gone (released on the next pass): "
        f"{stats.orphaned_awaiting}",
        f"near copies: {stats.near_duplicates}",
        f"recycled: {stats.recycled}",
        f"truncated: {stats.truncated}",
        f"parked as classification_failed: {stats.failed}",
    ]
    if llm is not None:
        lines += [
            f"llm judged: {llm.judged}",
            f"llm invalid answers: {llm.invalid}",
            f"llm reused by copies: {llm.reused}",
            f"llm waiting for a retry: {llm.failed_waiting}",
            f"llm parked: {llm.parked}",
            f"llm dropped: {llm.dropped}",
        ]
    return "\n".join(lines)


@classify_app.command("stats")
def classify_stats() -> None:
    """Count the stored verdicts: by label, out of scope by reason, and the flags."""
    with _guard(), closing(build_container()) as services:
        stats = services.classifier.stats()
        llm = services.escalation.counts()
    typer.echo(_render_stats(stats, llm) if stats.total else _NO_VERDICTS)


# ---- topics ----


@topics_app.command("show")
def topics_show() -> None:
    """Print the vocabulary in force as YAML (the shipped defaults until one is installed)."""
    with _guard(), closing(build_container()) as services:
        current = services.topics.current()
        text = services.topics.show_text()
    origin = (
        "defaults shipped with the application (version 0)"
        if current.version == 0
        else f"stored version {current.version}"
    )
    typer.echo(f"# topics: {origin}")
    typer.echo(text, nl=False)


@topics_app.command("set-file")
def topics_set_file(
    file: Annotated[
        Path,
        typer.Argument(
            metavar="PATH",
            help=f"YAML file with the vocabulary (UTF-8, at most {MAX_CONFIG_BYTES} bytes). "
            "Unknown keys, duplicates and oversize lists are refused.",
        ),
    ],
    reason: Reason,
    actor: Actor = None,
) -> None:
    """Install a vocabulary from a YAML file as a new version (audited with actor and reason).
    The same vocabulary again changes nothing. Items already classified are not redone."""
    with _guard():
        text = read_import_text(file, max_bytes=MAX_CONFIG_BYTES, what="topic file")
        with closing(build_container()) as services:
            change = services.topics.set_text(
                text, actor=resolve_actor(actor, os.environ), reason=reason
            )
    word = "updated" if change.changed else "unchanged"
    typer.echo(f"topics {word}: version {change.version}")


@topics_app.command("reset")
def topics_reset(reason: Reason, actor: Actor = None) -> None:
    """Go back to the vocabulary shipped with the application (as a new, audited version)."""
    with _guard(), closing(build_container()) as services:
        change = services.topics.reset(actor=resolve_actor(actor, os.environ), reason=reason)
    if change.changed:
        typer.echo(f"topics reset to the defaults: version {change.version}")
    else:
        typer.echo("topics already use the defaults")


# ---- catalog ----

_ANALYSIS_OFF = (
    "global analysis is off: no catalog was fetched. Turn it on with `social-tracker analysis on`."
)


def _render_refresh(report: RefreshReport) -> str:
    lines = []
    for r in report.results:
        name = r.source.value
        removed = f", {r.removed} removed" if r.removed else ""
        cut = " [CUT SHORT: the answer reached the limit]" if r.truncated else ""
        if r.outcome is Outcome.REFRESHED:
            lines.append(
                f"{name}: refreshed ({r.new} new, {r.updated} updated, {r.unchanged} unchanged, "
                f"{r.discarded} discarded{removed}){cut}"
            )
        elif r.outcome is Outcome.DRY_RUN:
            lines.append(
                f"{name}: dry run ({r.new} new, {r.unchanged} unchanged, "
                f"{r.discarded} discarded{removed}){cut}"
            )
        elif r.outcome is Outcome.FAILED:
            lines.append(f"{name}: failed ({escape_unprintable(r.error or 'unknown')})")
        elif r.outcome is Outcome.SKIPPED_BUSY:
            lines.append(f"{name}: skipped, another refresh is running")
        else:
            lines.append(f"{name}: {r.outcome.value}")
    return "\n".join(lines)


@catalog_app.command("refresh")
def catalog_refresh(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Fetch and parse the catalogs and report; store nothing."),
    ] = False,
) -> None:
    """Fetch the USGS and EMSC catalogs now and store their events (one-shot).

    Like the sources, the catalogs are fetched only while global analysis is on; with it off
    nothing is fetched (not even with --dry-run). A manual refresh ignores the refresh
    interval and any backoff, but not a refresh already running. Only earthquake.usgs.gov and
    www.seismicportal.eu are contacted, over https.

    Exit status: 0 when every catalog was refreshed (or is being refreshed elsewhere), 1 when
    a catalog failed (the others are still stored), 3 when global analysis is off."""
    with _guard(), closing(build_catalog()) as built:
        if not built.control.global_analysis_enabled():
            typer.echo(_ANALYSIS_OFF, err=True)
            raise typer.Exit(EXIT_NOTHING_ELIGIBLE)
        report = built.refresher.refresh(force=True, dry_run=dry_run)
    typer.echo(_render_refresh(report))
    if report.failed:
        raise typer.Exit(EXIT_REFUSED)


def _render_catalog_status(status: CatalogStatus) -> str:
    newest = status.newest_event_at.isoformat() if status.newest_event_at else "none"
    lines = [
        f"available: {'yes' if status.available else 'no'}",
        f"complete: {'yes' if status.complete else 'no'} "
        "(every catalog refreshed recently and answered in full)",
        f"newest event: {newest}",
    ]
    for s in status.sources:
        last = s.last_success_at.isoformat() if s.last_success_at else "never"
        line = (
            f"{s.source.value}: {s.events} event(s); last good refresh {last}; "
            f"failures {s.consecutive_failures}"
        )
        if s.next_attempt_at is not None:
            line += f"; next attempt {s.next_attempt_at.isoformat()}"
        if s.last_error:
            line += f"; last error: {escape_unprintable(s.last_error)}"
        if s.stale:
            line += " [STALE]"
        if s.truncated:
            line += " [CUT SHORT]"
        lines.append(line)
    return "\n".join(lines)


@catalog_app.command("status")
def catalog_status() -> None:
    """Show the catalog cache: events per catalog, the last good refresh, failures, staleness.

    A catalog is STALE when its last good refresh is older than the maximum age (the
    classification then records `catalog: unavailable`). Reads the local cache only."""
    with _guard(), closing(build_container()) as services:
        status = services.catalog.status()
    typer.echo(_render_catalog_status(status))


# ---- llm ----


def _names(models: tuple[str, ...]) -> str:
    return ", ".join(escape_unprintable(m) for m in models) or "none"


def _render_llm_status(status: LlmStatus) -> str:
    reachable = "yes" if status.reachable else f"no ({escape_unprintable(status.error or '')})"
    counts = status.counts
    lines = [
        f"server: {status.base_url}",
        f"reachable: {reachable}",
        f"configured models: {_names(status.configured)}",
        f"installed: {_names(status.installed)}",
        f"missing: {_names(status.missing)}",
        f"global analysis: {'on' if status.analysis_on else 'off'}",
        f"waiting: {status.queued}",
        f"budget this hour: {status.used_this_hour} of {status.hourly_cap}",
        f"judged: {counts.judged}, invalid answers: {counts.invalid}, "
        f"settled from an original: {counts.reused}",
        f"waiting for a retry: {counts.failed_waiting}, parked: {counts.parked}, "
        f"dropped: {counts.dropped}",
    ]
    return "\n".join(lines)


@llm_app.command("status")
def llm_status() -> None:
    """Show whether the model server answers, which models it has, the queue and the hourly budget.

    Asks the server for its model list (nothing is loaded and no item text is sent). A server
    that is down is reported, not an error: the exit status is 0 either way."""
    with _guard(), closing(build_llm()) as built:
        status = built.service.status()
    typer.echo(_render_llm_status(status))


@llm_app.command("requeue-parked")
def llm_requeue_parked(
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            max=1000,
            help="How many parked items to send back at most (default: SOCIAL_TRACKER_LLM_BATCH).",
        ),
    ] = None,
) -> None:
    """Put parked items (given up on after repeated failures) back in the queue now.

    They come back by themselves after SOCIAL_TRACKER_LLM_PARKED_TTL_S (default 24 hours); this
    command does not wait. They queue behind the healthy items, with a clean slate of failures,
    and nothing is sent to the model by this command. Needs no running model server."""
    with _guard(), closing(build_llm()) as built:
        back = built.service.requeue_parked(limit)
    typer.echo(f"{back} parked item(s) back in the queue")


def _render_ping(result: PingResult) -> str:
    label = result.label.value if result.label is not None else "?"
    model = escape_unprintable(result.model or "?")
    return f"ok: {model} answered {label} in {result.elapsed_s:.1f} s"


@llm_app.command("ping")
def llm_ping() -> None:
    """Ask the model one tiny harmless question to see that it answers (this loads the model).

    Run it yourself: nothing else ever calls it. It touches no verdict and uses none of the
    hourly budget. Exit status: 0 when the model answered with a valid verdict, 1 when not."""
    with _guard(), closing(build_llm()) as built:
        result = built.service.ping()
    if not result.ok:
        typer.echo(
            f"error: the language model did not answer ({escape_unprintable(result.error or '?')})",
            err=True,
        )
        raise typer.Exit(EXIT_REFUSED)
    typer.echo(_render_ping(result))


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
    line = (
        f"pass complete: {counts[CollectOutcome.COLLECTED]} collected "
        f"({new} new item(s)), {counts[CollectOutcome.NOT_MODIFIED]} not modified, "
        f"{counts[CollectOutcome.FAILED]} failed, {counts[CollectOutcome.DISCARDED]} discarded, "
        f"{skipped} skipped, {summary.errors} error(s)"
    )
    if isinstance(summary.classification, ClassifyReport):
        return f"{line}\nclassification: {_render_classify(summary.classification)}"
    return line


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
    (their poll interval, 600 s by default), one at a time, then classifies what is pending (CPU
    only) and removes items past the retention period. Stopping lets the collection in flight
    finish (a second Ctrl+C or SIGTERM exits at once with status 130 or 143). Run one worker per
    database.
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

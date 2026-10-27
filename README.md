# Social risk monitor

Tracks social-media signals and surfaces risk. This release (slice 1) provides the foundation:
a registry of **sources** (news RSS feeds, Instagram accounts, live streams, recorded audio), the
**analysis switches** (system-wide and per source), a SQLite store and an append-only **audit
log** in which every change records who made it and why. A local **web UI** (`serve`) manages
the same data from a browser.

The collection and analysis workers are not part of this slice: nothing here fetches or analyses
anything yet, and nothing ever publishes or sends data outside the system. You manage what *will*
be monitored and keep the switches safe: the system starts **paused**, every source starts
**disabled** and every analysis flag starts **off**.

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # creates .venv and installs the project with the dev tools
uv run social-tracker --help
```

## Configuration

Configuration comes from the process environment (no file is read). All variables are optional.

| Variable | Meaning | Default |
| --- | --- | --- |
| `SOCIAL_TRACKER_DB_PATH` | SQLite database file. A leading `~` is expanded and missing parent directories are created. | `data/social_tracker.db` (relative to the working directory) |
| `SOCIAL_TRACKER_ACTOR` | Who the audit log attributes your changes to. `--actor` overrides it. | `cli:<OS user name>` |
| `SOCIAL_TRACKER_BUSY_TIMEOUT_MS` | How long a write waits for another process using the database (1..600000). | `5000` |
| `SOCIAL_TRACKER_RETENTION_DAYS` | How many days collected items are kept, counted from when they were fetched (0..3650). `0` means "purge once classified". | `30` |
| `SOCIAL_TRACKER_ALLOW_PRIVATE_HOSTS` | Comma-separated host names (or IP addresses) the collector may contact although they resolve to a private address. See [Collecting content](#collecting-content). | empty (all private addresses are blocked) |
| `SOCIAL_TRACKER_ALLOWED_PORTS` | Comma-separated destination ports the collector may contact, for feeds and every redirect hop. A value replaces the default list. | `80,443,8080,8443` |
| `SOCIAL_TRACKER_MAX_ITEMS_PER_SOURCE` | Most items one source may hold at a time. Over it, new entries are dropped and counted as `source_quota`. | `100000` |
| `SOCIAL_TRACKER_MAX_BYTES_PER_SOURCE` | Most bytes (title, text and raw entry) one source may hold. Over it, new entries are dropped and counted as `source_quota`. | `536870912` (512 MiB) |
| `SOCIAL_TRACKER_MAX_BYTES_PER_FETCH` | Most bytes of items one fetch may add. Over it, the rest of the feed is dropped and counted as `fetch_quota`. | `16777216` (16 MiB) |

A blank value counts as unset. The `data/` directory is git-ignored.

**The default database depends on the working directory.** `data/social_tracker.db` is resolved
against the directory you run the command from, so the same command run from another directory
opens (and creates) another, empty database. The path is made absolute once, when the command
starts, and `status` prints it on its first line (`database: ...`); check it when the data you
expect is missing, or set `SOCIAL_TRACKER_DB_PATH` to an absolute path for anything that matters.

**Permissions.** A database directory the program creates gets mode `0700` and a database file it
creates gets mode `0600` (the `-wal` and `-shm` files next to it inherit that), and an export file
is written `0600`, because the data is an audit trail and a list of what you monitor. An existing
directory or file keeps whatever mode you gave it. Windows has no such modes: there a new file
inherits the permissions of its directory, so keep the database in a directory only you can read.

Sources never contain credentials. A source that needs one names it through `secret_ref`, shaped
`<scheme>:<name>` with the scheme `env` (an environment variable), `keyring` (an OS keyring
entry) or `file` (a file under the secrets directory). The reference is stored and exported; the
secret itself never is, and the CLI never prints it.

## Quickstart

Every change needs a reason (`--reason`, `-r`) and is attributed to an actor (`--actor`).

```bash
uv run social-tracker sources add city_news --type news_rss --name "City News" \
    --set feed_url=https://example.org/rss.xml -r "first feed"
uv run social-tracker sources list
uv run social-tracker sources enable city_news -r "ready to collect"
uv run social-tracker analysis on --source city_news -r "watch this feed"
uv run social-tracker analysis on -r "start the system"
uv run social-tracker status
```

Analysis really runs for a source only when **all three** hold: the system switch is on, the
source is enabled, and the source's own analysis switch is on. `status` counts those sources.
Collecting content needs only the first two (see [Collecting content](#collecting-content)).

### Commands

| Command | What it does |
| --- | --- |
| `sources list` | One row per source: id, type, enabled, collection state, analysis switch, name. |
| `sources add ID --type T --name N [--set KEY=VALUE]...` | Register a source (disabled). |
| `sources edit ID --set KEY=VALUE...` | Change fields. `id`, `type` and `enabled` cannot be edited. |
| `sources enable ID` / `sources disable ID` | Turn collection eligibility on or off. |
| `sources delete ID` | Remove a source and its runtime state. Audit events are kept. |
| `sources import FILE [--dry-run]` | Load sources from YAML, all or nothing. |
| `sources export [-o, --output FILE] [--force]` | Write every source as YAML to stdout or a file. |
| `analysis on` / `analysis off` `[--source ID]` | Switch analysis for the system or for one source. |
| `status` | Switches, alert threshold, source counts, collection states. |
| `run [--once] [--interval-tick SECONDS]` | Run the collection worker (see [Collecting content](#collecting-content)). |
| `items list [--source ID] [--limit N]` | The collected items, newest first. |
| `items search QUERY [--limit N]` | Search titles and texts by words. |
| `serve [--host H] [--port N] [--allow-non-loopback] [--allowed-host NAME]` | Run the local web UI (see [Web UI](#web-ui)). |

`--set KEY=VALUE` takes a source field. The value is read as JSON when it parses (`600`, `true`,
`null`, `["flood","fire"]`) and as plain text otherwise; wrap text that looks like JSON in quotes
(`sources edit ID --set 'name="2024"'`). `--set field=null` clears an optional field. `sources add`
takes `id`, `type` and `name` through their own arguments and options, not through `--set`.

Results go to stdout. The exit status says what happened:

| Status | Meaning |
| --- | --- |
| `0` | Success. |
| `1` | The operation was refused (invalid input, conflict, unknown source, storage failure...): one `error: ...` line on stderr, nothing on stdout. |
| `2` | Usage mistake (missing option, unknown command). |
| `70` | Unexpected failure, i.e. a bug: `error: unexpected failure (<ExceptionType>); this is a bug` and the frames where it happened on stderr, never the exception message or any variable. Please report it. |

Error messages never repeat the values, keys, reasons or actors you typed, so a secret pasted into
the wrong place is not echoed back. Paths are the exception: `status` and `sources export -o`
print the path they use, with control and hidden characters escaped.

## Source file format

```yaml
sources:
  - id: city_news
    type: news_rss
    name: City News
    feed_url: https://example.org/rss.xml
    poll_interval_s: 600
    keywords: [flood, fire]
    enabled: true
  - id: city_instagram
    type: instagram
    name: City Instagram
    handle: some.handle
    secret_ref: env:IG_MAIN
  - id: radio_live
    type: live_stream
    name: Radio Live
    stream_url: https://stream.example.org/radio.mp3
    chunk_seconds: 30
    language: es
  - id: radio_recordings
    type: recorded_audio
    name: Radio Recordings
    watch_dir: /srv/audio/incoming
```

Common fields: `id` (3-64 characters of `a-z`, `0-9`, `_`), `type`, `name` (up to 100 characters,
unique ignoring case), `credibility` (1-5, default 3), `enabled` (default `false`) and
`secret_ref` (optional). Unknown fields are rejected, so a pasted `token:` or `password:` is an
error rather than something stored.

| Type | Required | Optional (default) |
| --- | --- | --- |
| `news_rss` | `feed_url` (http or https) | `poll_interval_s` (600, at least 120), `keywords` (up to 200) |
| `instagram` | `handle`, `secret_ref` | `include_stories` (true), `include_posts` (true), `poll_interval_s` (1800, at least 900) |
| `live_stream` | `stream_url` (http, https, rtmp or icecast) | `chunk_seconds` (30, between 10 and 60), `language` (`es`, BCP-47 tag) |
| `recorded_audio` | exactly one of `feed_url` or `watch_dir` | `poll_interval_s` (1800, at least 1) |

URLs may not embed credentials (`user:password@host`) or secret-looking query parameters.
`created_at` and `updated_at` may appear in an exported file but are never trusted on import:
the system clock sets them.

### Import replaces entries wholesale

An import is **not** a merge. An entry whose `id` already exists overwrites that source, and every
field the entry omits goes back to its default. In particular an entry without `enabled`
**disables** a source that was enabled, and omitted optional fields (keywords, poll interval...)
are reset. Sources missing from the file are left alone. The whole file is applied or nothing is.

Run `sources import FILE --dry-run` first: it reports what would be added, updated or left
unchanged and writes nothing (it needs no `--reason`). The file must be UTF-8 (a byte order mark is
accepted) and at most 1,000,000 bytes; larger files are refused before they are parsed.

Round-trip: `sources export -o backup.yaml` followed by `sources import backup.yaml` reports every
source as unchanged. `export` refuses to overwrite an existing file unless you pass `--force`.

`export -o` writes a temporary file next to the target, flushes it to disk and only then puts it in
place, so a full disk or a crash never leaves a half-written or empty export, and with `--force` the
old file stays intact until the new one is complete. It never writes through a symbolic link or
over anything that is not a regular file, and the result is readable by you only (`0600`). `import`
likewise only reads regular files (not a pipe, a device or a directory).

## How to load sources

Each recipe below works either as one `sources add` command or as an entry in an import file.

**News RSS feed**

```bash
uv run social-tracker sources add city_news --type news_rss --name "City News" \
    --set feed_url=https://example.org/rss.xml --set poll_interval_s=300 \
    --set 'keywords=["flood","fire"]' -r "register feed"
```

**Instagram account** (needs a secret reference; the token itself lives in the environment, the
keyring or the secrets directory, never in the registry)

```bash
uv run social-tracker sources add city_instagram --type instagram --name "City Instagram" \
    --set handle=some.handle --set secret_ref=env:IG_MAIN -r "register account"
```

**Live stream**

```bash
uv run social-tracker sources add radio_live --type live_stream --name "Radio Live" \
    --set stream_url=https://stream.example.org/radio.mp3 --set chunk_seconds=30 --set language=es \
    -r "register stream"
```

**Recorded audio** (a podcast feed *or* a watched directory, not both)

```bash
uv run social-tracker sources add radio_recordings --type recorded_audio --name "Radio Recordings" \
    --set watch_dir=/srv/audio/incoming -r "register recordings"
uv run social-tracker sources add weekly_podcast --type recorded_audio --name "Weekly Podcast" \
    --set feed_url=https://example.org/podcast.xml -r "register podcast"
```

After adding, `sources enable ID` makes a source eligible for collection and `analysis on --source
ID` opts it into analysis.

## Web UI

`uv run social-tracker serve` starts a local web UI on `http://127.0.0.1:8765` (`--host`,
`--port`). It manages the same sources and switches as the CLI, from the same database
(`SOCIAL_TRACKER_DB_PATH`): a sources list with the collection state, add and edit forms that
adapt to the source type, enable and disable, the system and per-source analysis switches, and a
delete page that asks for confirmation and says that alerts and derived data are kept. A short
guide per source type is at `/howto` and in [docs/howto.md](docs/howto.md). The worker status
panel (GPU and model, stream lag) reads `n/a - no worker running` until the GPU workers exist
(the collection worker has no GPU part), and there is no alerts view yet.

The list has a **Health** column read from the collection state: `ok` while a source is idle or
collecting, `degraded` in backoff (with the failure count and the next retry time), `failed` once
collection stopped (with the failure count), and `n/a` for a disabled source. **Last fetch** is
the time (UTC) of the last successful fetch and **Last error** the short reason of the last
failure (for example `http 503`, `timeout`, `invalid feed: ...`), cleared by the next success;
both read `n/a` while there is nothing to show. The error is a fixed phrase, never a URL, a
response body or a secret.

It is meant for one operator on this machine:

- **Loopback only.** `serve` refuses any other `--host` unless you pass `--allow-non-loopback`,
  because the UI has no login: anyone who can connect can change your sources. Pages are only
  served for `localhost`, `127.0.0.1`, `[::1]` and the host you bind. The `Host` header is parsed
  strictly (`name[:port]` or `[ipv6][:port]`, nothing else) and compared in canonical form, which
  defeats DNS rebinding; a wildcard bind (`0.0.0.0`, `::`) adds no name of its own, so name the
  ones clients use with `--allowed-host NAME` (repeatable).
- **Cleartext.** The UI speaks plain HTTP: beyond loopback its traffic is not encrypted and the
  session cookie has no `Secure` flag. Use `--allow-non-loopback` only on a network you trust, or
  put a TLS proxy you control in front of it.
- **Protected requests.** Every change is a POST that needs a CSRF token tied to your browser
  session (a `SameSite=Strict` cookie), bodies are limited to 16 KiB and must arrive within 10
  seconds (a slower client gets a 408), and responses carry a strict `Content-Security-Policy`,
  `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `X-Content-Type-Options: nosniff`,
  a `Permissions-Policy` that denies every browser feature, and `Cross-Origin-Opener-Policy` and
  `Cross-Origin-Resource-Policy` set to `same-origin`. All text is escaped; a name such as `<script>` is shown as
  text.
- **Offline and read-only toward the world.** htmx is vendored under the package's `static/`
  directory (no CDN), the UI makes no outbound request, and it has no publish, share, post or
  approve action.
- **No secrets.** Instagram takes a `secret_ref` (`env:NAME`, `keyring:NAME`, `file:NAME`), never
  a password. The reference is never shown back; leave the box empty on an edit to keep it.
- **Audited.** Each change is attributed to the actor `web:local` with the reason you give (it
  defaults to `via web UI`; the row buttons use that default).
- **Failures stay readable.** A busy database or a source deleted in another tab shows a short
  message, never a traceback; a switch that fails shows the row as the server holds it, and an
  add, edit or delete that hits a busy database comes back with what you typed (an address is
  echoed only when it carries no login or query string; a secret reference never) and a message to
  try again. Every failure of an in-page (htmx) request is a small fragment announced in the
  status region, never a page pasted into the table; a refused token also reloads the page.
- **Safe to repeat.** Submitting the add form twice creates one source (the repeat with identical
  values is a success, a different one with the same id is a conflict). The edit form carries the
  version it was opened at: saving after someone else changed the source shows the current values
  with a message instead of overwriting them. The address of a stored feed or stream is shown with
  its query string masked (`?***`); leave it as it is to keep the stored value.

Pages work without JavaScript; with it, the row buttons update in place.

## Collecting content

This is an internal observation system: it only READS the feeds you register. Nothing is ever
published, posted or sent, and the only outbound traffic is a `GET` to the URL of an enabled
source. It runs on the CPU only.

```bash
uv run social-tracker run --once
uv run social-tracker items list --limit 10
uv run social-tracker items search "sismo cali"
```

`run` starts the collection worker. Run ONE worker per database (a second one would treat the
first one's collection in flight as interrupted).

- **It starts with analysis off.** Starting a worker switches global analysis OFF, with an audit
  event attributed to `system:worker` and the reason `worker startup`. No other command ever does
  this. Turn it on afterwards with `analysis on`; the running worker notices on its next pass.
  Because of this, `run --once` right after starting collects nothing and says so: it is meant
  for diagnostics, and a real collection needs `run` left running.
- **What is collected.** Each pass (every `--interval-tick` seconds, default 5) collects the
  enabled RSS and Atom sources that are due: never fetched, or `poll_interval_s` seconds (600 by
  default) after the last successful fetch. Sources are collected one at a time, in id order.
  Nothing is fetched while a source is disabled or global analysis is off; the per-source
  analysis switch is about analysis, not collection.
- **What is stored.** One item per entry: title and visible text (markup is parsed and dropped,
  scripts never reach the text), author, link, time, a content hash and the entry as received
  (`raw_payload`). Entries with no visible content are discarded and counted. Text over 100 KiB
  is cut at that size and flagged. An entry is stored once per source: the same guid, or the same
  title and text under another address, is a duplicate. All the items of a fetch are stored in
  one transaction or none are.
- **Failures.** A timeout, a name lookup error, an HTTP error status, an oversize or malformed
  document are recorded on the source (see Last error in the web list) and retried with
  exponential backoff (60 s, 120 s, 240 s, 480 s; never more than one hour). `Retry-After`
  (seconds or an HTTP date) is honoured as a minimum wait. 401, 403 and 429 are failures with a
  clear reason (`http 403 (access denied)`). After 5 consecutive failures the source is `failed`
  and is left alone until the operator disables and enables it again (`sources disable ID`, then
  `sources enable ID`, which clears the failure count). A source disabled or deleted while it is
  being fetched is discarded without storing anything.
- **Stopping.** SIGINT (Ctrl+C) and SIGTERM stop the worker gracefully: the collection in flight
  is finished and committed, no new one starts. A fetch is bounded (a response is read for at
  most 15 s), so the stop completes well inside 30 s. A crash leaves at most a source marked as
  collecting; the next worker start counts it as one failed attempt.
- **Retention.** Items are deleted once they are older than `SOCIAL_TRACKER_RETENTION_DAYS` (30 by
  default), counted from `fetched_at` and checked about once an hour while the worker runs. The
  boundary is strict: an item exactly 30 days old is kept, one microsecond older is removed.
  With `0` only items that reached the `classified` stage are removed (nothing sets that stage
  yet). Alerts are a separate store and are not touched.
- **Safety of the fetch.** Only `http` and `https` addresses without a login are fetched. Every
  host is resolved and ALL its addresses must be public: loopback, private, link-local (cloud
  metadata included), carrier-grade NAT, multicast and reserved addresses are refused, for
  redirects too (at most 5, each one checked again), and the connection goes to the address that
  was checked (so a changing DNS answer cannot redirect it). To reach a feed on a private network,
  list its host name in `SOCIAL_TRACKER_ALLOW_PRIVATE_HOSTS`; every use is logged with the host
  name. A response is limited to 2 MiB (after decompression too), 10 s per network operation and
  15 s overall. Proxy settings of the environment and cookies are ignored. Feeds are parsed with
  `defusedxml` and any document type declaration is refused, which rules out entity bombs and
  external entities.
- **Searching.** `items search` splits your text into plain words (lowercase, accents ignored)
  that must all appear as whole words. Operators such as `AND`, `OR`, `NEAR`, `*`, `-`, quotes
  and `column:` are ordinary text, so no search text can raise an error or change the query.
  Put `--` before a query that starts with `-`. Item text is untrusted: the CLI escapes control
  characters when it prints it.
- **Logs** (on stderr) name source ids, counts and fixed reasons; never URLs, item text or
  secrets.

## Audit log

Each real change writes one event, in the same transaction as the change, with the actor and the
reason. A request that changes nothing (enabling an enabled source, importing known data) writes
no event. Events are append-only and survive the deletion of the source they describe.

## Concurrent use

Several CLI processes may share one database file: writers are serialised, and a writer that
waits longer than the busy timeout fails with `error: database error: SQLITE_BUSY` and exit status
1. Retry the command or raise `SOCIAL_TRACKER_BUSY_TIMEOUT_MS`. Readers (`sources list`, `status`,
`sources export`) never wait for a writer: each reads one consistent snapshot of the database, so
a listing is never a mix of before and after another process's change.

## Development

```bash
uv run ruff check
uv run ruff format --check
uv run pytest -q
uv run lint-imports
```

A few checks drive the UI in a real Chromium (double click, a server restart under an open page,
a busy database, the status announcements). They are opt-in and excluded from the default run:

```bash
set PLAYWRIGHT_BROWSERS_PATH=D:\social_tracker_models\playwright   # any folder you like
uv run playwright install chromium
uv run pytest -m browser
```

Without a browser installed they are skipped.

The architecture is hexagonal. Domain modules are pure, services depend on ports only, and
`social_tracker.bootstrap` is the composition root that wires the SQLite adapters into the
services. Entrypoints such as the CLI never import persistence; `lint-imports` enforces it.
The collector follows the same rules, and its fetcher adapter (`collector/http_fetcher.py`) is the
only module allowed to import an HTTP client or sockets: the web UI has no outbound path at all
and reaches the worker side only through the composition root.

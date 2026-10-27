# Social risk monitor

Tracks social-media signals and surfaces risk. This release provides the foundation and the
first collector: a registry of **sources** (news RSS feeds, Instagram accounts, live streams,
recorded audio), the **analysis switches** (system-wide and per source), a SQLite store, an
append-only **audit log** in which every change records who made it and why, a local **web UI**
(`serve`) that manages the same data from a browser, and a **collection worker** (`run`) that
reads the RSS and Atom feeds you register and stores their entries as **items**.

Only news RSS and Atom sources are collected so far: Instagram, live stream and recorded audio
sources can be registered, but nothing collects them yet. The collected items go through a
CPU-only **classification cascade** (topic filter, heuristic rules, recycled and near-duplicate
check; see [Classifying content](#classifying-content)); the language-model judgment, the
official-catalog cross-check, alerts and the GPU workers are later slices. Nothing ever publishes or sends
data outside the system: the only outbound traffic is a read-only `GET` to the feeds you
registered. The system starts **paused**, every source starts **disabled** and every analysis
flag starts **off**.

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
| `SOCIAL_TRACKER_RECYCLE_DAYS` | Content first SEEN here STRICTLY more than this many days apart is "recycled" when it comes back (1..3650). See [Classifying content](#classifying-content). | `7` |
| `SOCIAL_TRACKER_NEAR_DUPLICATE_THRESHOLD` | Estimated similarity (above 0, up to 1, at most four decimals) from which two texts are near duplicates, inclusive. | `0.8` |
| `SOCIAL_TRACKER_NEAR_DUPLICATE_CANDIDATES` | How many near-duplicate candidates are verified per item, the best-ranked first (1..1000). A lookup that hits the limit is counted in the report of the run. | `500` |
| `SOCIAL_TRACKER_CLASSIFICATION_RETENTION_DAYS` | How long the verdict of an item that no longer exists is kept (1..3650); the hourly retention purge forgets the older ones. | `180` |
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

Nothing is collected until a worker runs. Start it in its own terminal and turn the system on
AFTER it is up:

```text
uv run social-tracker run
uv run social-tracker analysis on -r "start the system"
```

The order matters: **starting a worker switches global analysis off** (audited as
`system:worker`, reason `worker startup`), so an `analysis on` issued before `run` is undone by
it. With the worker already running, `analysis on` is picked up on its next pass. For a
diagnostic one-shot pass use `run --once --no-reset-analysis` (after `analysis on`): it skips the
reset, collects what is due and exits. A plain `run --once` cannot collect anything, because it
resets analysis and then makes its one pass.

### Commands

| Command | What it does |
| --- | --- |
| `sources list` | One row per source: id, type, enabled, collection state, analysis switch, name. |
| `sources add ID --type T --name N [--set KEY=VALUE]...` | Register a source (disabled). |
| `sources edit ID --set KEY=VALUE...` | Change fields. `id`, `type` and `enabled` cannot be edited. |
| `sources enable ID` / `sources disable ID` | Turn collection eligibility on or off. |
| `sources reset ID` | Put a source whose collection `failed` back to work (idle, failures forgotten, stale error cleared). Any other source is left as it is. Also a **Reset** button in the web list. |
| `sources delete ID` | Remove a source and its runtime state. Audit events are kept. |
| `sources import FILE [--dry-run]` | Load sources from YAML, all or nothing. |
| `sources export [-o, --output FILE] [--force]` | Write every source as YAML to stdout or a file. |
| `analysis on` / `analysis off` `[--source ID]` | Switch analysis for the system or for one source. |
| `status` | Switches, alert threshold, source counts, collection states. |
| `run [--once [--no-reset-analysis]] [--interval-tick SECONDS]` | Run the collection worker (see [Collecting content](#collecting-content)). |
| `items list [--source ID] [--limit N] [--ids]` | The collected items, newest first (`--ids` shows each id, which `classify show` takes). |
| `items search QUERY [--limit N] [--ids]` | Search titles and texts by words. |
| `classify run [--limit N]` | Classify the pending items of the eligible sources (CPU only). The worker does it after every collection pass. Exit status `3` when no source is eligible. |
| `classify reclassify --status S [--limit N]` | Send the items at status `out_of_scope` (after a vocabulary change) or `classification_failed` back to the queue; the next run classifies them again. |
| `classify show ITEM_ID` | The verdict of one item: label, confidence, deciding stage, rationale, flags. |
| `classify stats` | Verdicts by label, out of scope by reason, the flags, near copies, duplicates waiting for their original, and the items parked as failed. |
| `topics show` | The topic vocabulary in force, as YAML. |
| `topics set-file PATH` / `topics reset` | Install a vocabulary from a YAML file / go back to the shipped one (new audited version). |
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
| `3` | `classify run` found no eligible source (global analysis off, or no enabled source with its own switch on), so it had nothing to do. It is not a failure: a run with eligible sources and nothing pending exits `0`. |
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
failure (for example `http 503`, `timeout`, `invalid feed: ...`), cleared by the next success,
by enabling the source and by a reset; both read `n/a` while there is nothing to show. The error
is a fixed phrase, never a URL, a response body or a secret. A source whose collection `failed`
shows a **Reset** button (the same as `sources reset ID`); pressing it twice is harmless.

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
uv run social-tracker run --once --no-reset-analysis
uv run social-tracker items list --limit 10
uv run social-tracker items search "sismo cali"
```

`run` starts the collection worker (see the Quickstart for the order of `run` and `analysis on`).

- **One worker per database.** `run`, the long-running one and `--once` alike, takes an exclusive
  lock on a `.worker-lock` file next to the database and refuses to start, with `error: another
  worker is already running on this database`, while another process holds it. The operating
  system releases the lock when the process exits, crashes or is killed, so a leftover file
  blocks nobody. A worker that has just taken the lock knows the previous one is gone, so it
  counts every collection still marked as running as one failed attempt; while it runs, only a
  claim older than the fetch deadline plus a minute is treated that way, so a collection that is
  really in flight is never failed from under its owner.
- **It starts with analysis off.** Starting a worker switches global analysis OFF, with an audit
  event attributed to `system:worker` and the reason `worker startup`. No other command ever
  does this. Turn it on afterwards with `analysis on`; the running worker notices on its next
  pass. `run --once --no-reset-analysis` skips the reset and its audit event, for diagnostics:
  it is accepted only with `--once` (anything else is a usage error, exit status 2), so the
  long-running worker always keeps the safe start.
- **What is collected.** Each pass (every `--interval-tick` seconds, default 5) collects the
  enabled RSS and Atom sources that are due: never fetched, or `poll_interval_s` seconds (600 by
  default) after the last successful fetch (a last fetch dated in the future, because the clock
  moved back, counts as due). Sources are collected one at a time, in id order. Nothing is
  fetched while a source is disabled or global analysis is off; the per-source analysis switch is
  about analysis, not collection.
- **What is stored.** One item per entry: title and visible text (markup is parsed and dropped,
  scripts never reach the text), author, link, time, a content hash and the entry as received
  (`raw_payload`). Bidi controls (such as the right-to-left override) and zero-width characters
  are removed from the title, text and author, so a hidden character can neither reorder what a
  reader sees nor split a word so that no search finds it; joiners that emoji sequences and some
  scripts need are kept, and `raw_payload` keeps the original. An element that hides the rest of
  an entry (an unclosed `<script>`) is not silent: the item gets the flag `content_dropped`,
  and so does one whose closed scripts and styles dropped 1,000 characters or more. Entries with
  no visible content are discarded and counted. Text over 100 KiB is cut at that size and
  flagged. An entry is stored once per source: the same guid, or the same title and text under
  another address, is a duplicate. All the items of a fetch are stored in one transaction or
  none are.
- **What happens next.** After the collection of each pass the worker classifies what is pending,
  on the CPU, in batches of 200 (see [Classifying content](#classifying-content)); a failure
  there costs that step only, and a stop request skips it.
- **Item text is untrusted.** Titles, texts, authors and `raw_payload` come from the internet.
  The CLI escapes control characters when it prints them; any other view of an item (a web page,
  a report, an API) MUST escape all four before showing them and treat them as data, never as
  markup.
- **Limits on what a feed can cost.** Over a limit the extra entries are dropped and counted (the
  fetch still succeeds, and the worker logs the count): at most 16 MiB of items from one fetch
  (`fetch_quota`, `SOCIAL_TRACKER_MAX_BYTES_PER_FETCH`), and at most 100,000 items and 512 MiB
  held per source (`source_quota`, `SOCIAL_TRACKER_MAX_ITEMS_PER_SOURCE` and
  `SOCIAL_TRACKER_MAX_BYTES_PER_SOURCE`; retention frees room as items expire). A feed is also
  parsed with a cap on its number of entries (500 are read, the rest counted as `over_limit`).
- **Failures.** A timeout, a name lookup error, an HTTP error status, an oversize or malformed
  document are recorded on the source (see Last error in the web list) and retried with
  exponential backoff (60 s, 120 s, 240 s, 480 s; never more than one hour). An explicit
  `Retry-After` (seconds or an HTTP date) is believed even beyond that hour, up to 7 days (a
  larger value is clamped, and the worker logs the clamp). 401, 403 and 429 are failures with a
  clear reason (`http 403 (access denied)`). After 5 consecutive failures the source is `failed`
  and is left alone until the operator acts: `sources reset ID` (or the **Reset** button of the
  web list) puts it back to work, and disabling then enabling it does the same and also clears
  the stale error. A source disabled or deleted while it is being fetched is discarded without
  storing anything.
- **Stopping.** SIGINT (Ctrl+C) and SIGTERM stop the worker gracefully: the collection in flight
  is finished and committed, no new one starts. A fetch cannot outlast its deadline (see the
  safety notes below), so the stop completes well inside 30 s. A SECOND signal means "now": the
  worker exits at once with status 130 (SIGINT) or 143 (SIGTERM) after closing the database,
  abandoning the collection in flight; the next worker start counts it as one failed attempt. A
  crash does the same.
- **Retention.** Items are deleted once they are older than `SOCIAL_TRACKER_RETENTION_DAYS` (30 by
  default), counted from `fetched_at` and checked about once an hour while the worker runs. The
  boundary is strict: an item exactly 30 days old is kept, one microsecond older is removed.
  A purged item does not come back when the feed still lists it: an entry published before the
  retention horizon (now minus the retention) is skipped at ingest and counted as
  `older_than_retention`, and for entries with no publication date, which are dated by when they
  were first seen, a small tombstone (the dedup key, no text) outlives the item (the oldest are
  forgotten beyond 50,000 per source). With `0` the items the classifier has processed are
  removed, once, as soon as the next purge runs: `classified` ones at once, `out_of_scope` ones
  only after their first 24 hours (so a vocabulary change can bring them back, see
  `classify reclassify`); they leave a tombstone, no horizon applies, and the items still waiting
  for classification or parked as `classification_failed` stay. Their verdicts and near-duplicate
  signatures are kept (they hold no text; see [what is stored](#classifying-content)), so a copy
  that shows up later is still recognised. Alerts are a separate store and are not touched.
- **Safety of the fetch.** Only `http` and `https` addresses without a login are fetched, and
  only on the ports 80, 443, 8080 and 8443 (`SOCIAL_TRACKER_ALLOWED_PORTS` replaces the list),
  redirects included. Every host is resolved and ALL its addresses must be public: loopback,
  private, link-local (cloud metadata included), carrier-grade NAT, multicast, reserved and
  site-local addresses are refused, IPv6 forms that wrap an IPv4 address (mapped, compatible,
  translated, NAT64, 6to4) are judged by the IPv4 address inside, and the connection goes to the
  address that was checked (so a changing DNS answer cannot redirect it). Redirects (at most 5)
  are followed by hand, each one checked again; a redirect from `https` to `http` is refused, and
  the conditional-request validators (`If-None-Match`, `If-Modified-Since`) are sent only to the
  host and port the fetch started at. To reach a feed on a private network, list its host name
  in `SOCIAL_TRACKER_ALLOW_PRIVATE_HOSTS`; every use is logged with the host name. A response
  is limited to 2 MiB (after decompression too) and 100 KiB of headers. A fetch has 10 s per
  network operation and 15 s overall: the 15 s is a wall-clock deadline enforced inside every
  connect, TLS handshake, read, write and name lookup, shared by the redirects and all the
  addresses of a host, so a server that sends one byte at a time cannot hold the worker past it
  (a lookup the system never answers is abandoned in a helper thread; at most 8 may hang at
  once). Proxy settings of the environment and cookies are ignored. Feeds are parsed with
  `defusedxml` and any document type declaration is refused, which rules out entity bombs and
  external entities.
- **Searching.** `items search` splits your text into plain words (lowercase, accents ignored)
  that must all appear as whole words. Operators such as `AND`, `OR`, `NEAR`, `*`, `-`, quotes
  and `column:` are ordinary text, so no search text can raise an error or change the query.
  Put `--` before a query that starts with `-`.
- **Logs** (on stderr) name source ids, counts and fixed reasons; never URLs, item text or
  secrets.

## Classifying content

This is internal observation only: a verdict is stored locally and shown to you, never published
or sent anywhere. The cascade runs on the CPU, needs no model and no network, and is
deterministic: the same item, history and vocabulary always give the same verdict. Within one run
the order the items were stored in does not matter (they are taken in the order they were first
SEEN here); across runs the item classified first is the original of its later copies. The
Servicio Geológico Colombiano may be NAMED in a text, but it is never a data source of this
application.

```bash
uv run social-tracker classify stats
uv run social-tracker topics show
```

The worker does `classify run` after every collection pass; run it yourself for a one-off pass.
Its exit status is `0` when it ran (also when nothing was pending), `1` when a storage error
stopped it (run it again: it resumes) and `3` when no source is eligible, so there was nothing it
could do. `classify show ITEM_ID` prints one verdict (`items list --ids` shows the ids).

**What it looks at.** The primary focus is (a) earthquakes and aftershocks, damage, rumors and the
people talking about them, and (b) what is said about the Secretaría de Gestión del Riesgo de
Emergencias y Desastres of Cali (formerly the DAGRD): statements attributed to it (true or false),
rumors and complaints about its management, alerts or evacuation orders claimed to come from it.
Other natural disasters (floods, landslides, volcano, forest fires, storms) are secondary. Crime
and public order unrelated to disasters are out of scope.

**The stages**, in order (the worker runs them after each collection pass, 200 items at a time,
the one seen first first; `classify run` does the same on demand):

1. **Topic filter.** An item with none of the vocabulary below is `out_of_scope` and counted by
   reason (`no_topic_match`, `crime_public_order`, `non_spanish` for English text); an item with
   no words at all is `unclassified`. The title and the text are read together. A text longer
   than 20,000 characters is read at both ends (the first 10,000 and the last 10,000) and flagged
   `truncated`; a truncated text with no topic in the parts that were read is NOT called off topic:
   it is kept as `unclassified` for review, last in the queue, so that padding cannot hide a rumor.
2. **Rules.** Weighted heuristic signals (table below) become a preliminary label and a
   confidence.
3. **Recycled and near-duplicate check.** See below.
4. The official-catalog cross-check and the language-model judgment are later units. The verdict
   already records `catalog: unavailable` for earthquake items (the check runs before the item's
   write transaction opens, so a slow catalog never holds the database), and marks what the LLM
   will have to settle (`needs_llm`, with a `priority`).

**Labels:** `unintentional_rumor`, `deliberate_disinformation`, `recycled_content`, `no_risk`,
`unclassified`, `out_of_scope`. A verdict has a label, a confidence from 0 to 1, the stage that
decided, a rationale of at most 300 characters (signal names, topics and matched vocabulary
terms; item text is never quoted), the evidence (`signals`), and the flags `low_confidence`,
`needs_llm`, `truncated`, `duplicate_of`, `near_of`, `recycled_of` and `awaiting_original`.

**Hostile spelling.** Before anything is matched the text is folded: case, accents, punctuation,
full-width forms and invisible characters are removed, and Cyrillic and Greek letters that look
like Latin ones are replaced by them (`tеmblor` with a Cyrillic `е` is `temblor`), as are the
blank "letters" (Hangul filler, Braille blank). A second, decoded view joins a word spelled letter
by letter (`t.e.m.b.l.o.r`, `t e m b l o r`, up to 15 letters) and reads a single digit inside a
word as a letter (`t3mblor`, `sism0`); the stages look at both views. The trick itself is a signal
(`obfuscated_text`): it raises suspicion and the queue priority, can keep an otherwise clean report
from settling, and never decides a label alone. Several spelled-out words in a row are joined into
one blob and are not recognised (the signal still fires).

**How the rules decide.** Each signal argues for one label with a weight; the evidence for a label
is `1 - product(1 - weight)` over its signals, so clues add up but never pass 1. The label with the
most evidence wins (a tie goes to the more serious one). Evidence under 0.4 decides nothing: the
item is `unclassified` and goes to the LLM. Two labels that are both at least 0.3 and closer than
0.15 are *conflicting*; the time of a prediction restates the prediction and is not counted as a
second opinion for that.

| Signal | Argues for | Weight | Fires when |
| --- | --- | --- | --- |
| `prediction_claim` | unintentional_rumor | 0.45 | a future quake is announced ("mañana habrá un temblor", "va a temblar", "pronostican un terremoto", "un sismo de magnitud 7 sacudirá Cali", "el temblor que se viene"), unless negated shortly before ("no habrá", "es falso") |
| `prediction_specific_time` | deliberate_disinformation | 0.35 | the prediction comes with a date, a weekday, an hour or "mañana"/"hoy"/"esta noche" |
| `pseudo_authority` | unintentional_rumor | 0.30 | "lo dijo la NASA", "según un científico" (anonymous authorities) |
| `virality_phrases` | unintentional_rumor | 0.20 | "compartan", "reenvía", "pásalo", "urgente", unless negated right before ("no compartan") |
| `censorship_bait` | deliberate_disinformation | 0.30 | "antes de que lo borren", "no quieren que sepas" |
| `all_caps` | unintentional_rumor | 0.15 | at least 20 letters, 70 % or more uppercase |
| `repeated_exclamation` | unintentional_rumor | 0.10 | `!!`, `¡¡` or three `!` |
| `unverifiable_sources` | unintentional_rumor | 0.25 | "fuentes confiables", "dicen que", "me dijo un amigo" |
| `official_attribution_unlinked` | deliberate_disinformation | 0.45 | an official body (the secretariat, UNGRD, ...) is named as the speaker of ANY claim and there is no link to a configured official domain: the full weight with an order or alarm word ("evacuen", "salgan", "peligro"), 55 % of it (0.25) without one. Never when the source itself, or the address of the item, is on an official domain |
| `official_link` | no_risk | 0.50 | a link to (or an item served from) a configured official domain |
| `evacuation_claim` | unintentional_rumor | 0.20 | evacuation, shelter and closure claims ("evacuen", "salgan de", "habilitan albergues", "cierre de vías", "suspendidas las clases") |
| `casualty_claim` | unintentional_rumor | 0.25 | casualty counts or collapse claims ("300 muertos", "el edificio colapsó"); 60 % of that next to a report or an official link |
| `recycled_marker` | recycled_content | 0.50 | "video de 2016", "video viejo", "hace años", "de hace 10 años", "fotos del archivo", "imágenes de archivo" |
| `old_year_reference` | recycled_content | 0.25 | a year at least two years before the year the item was first seen |
| `news_report` | no_risk | 0.60 | a report verb with a quake and a magnitude ("el SGC reportó un sismo de magnitud 4.2", "un temblor de 4.5 grados") or an agency wording ("el Servicio Geológico Colombiano informó un temblor", "la Red Sismológica Nacional registró un sismo") |
| `obfuscated_text` | unclassified | 0.30 | mixed scripts in one word, invisible filler characters, a word spelled letter by letter or a digit inside a word: suspicion only, no label |
| `catalog_mismatch` | unintentional_rumor | 0.30 | the official-catalog cross-check contradicts the claimed magnitude (difference of 1.0 or more against the event nearest in time) or finds no event in the time window; set by the check, not read from the text. Too weak to decide a label alone: it adds to other rumor evidence and sends a clean-looking report to review |

A link counts only if it is a real one: it must start with `http://`, `https://` or `www.` and its
host (up to the first `/`, `?`, `#` or `:`) must be the configured domain or a subdomain of it, on a
label boundary. A login in the address (`https://official-domain.example@evil.example/`), the
domain only in a path or query string, a look-alike (`evil-official-domain.example`,
`official-domain.example.evil.com`), a punycode or non-Latin look-alike and a bare mention of the
domain are not official links. Case and a trailing dot do not matter.

Reporting a real quake is not predicting one: "el SGC reportó un sismo de magnitud 4.2" gets
`news_report` and none of the prediction signals. The weights are starting points chosen by hand,
not measured values (see Calibrating).

**What can settle as harmless.** `no_risk` is settled (never queued) when nothing argues for risk:
- an on-topic Spanish text of at least 6 words with no signal at all (chatter, a complaint about
  the secretariat, a news item about a landslide) is `no_risk` at 0.6; a shorter text, or a pile
  of tokens without a Spanish function word, is inconclusive instead;
- strong benign evidence can reach 0.8 and more: a report with an official link
  (`news_report` 0.60 and `official_link` 0.50 give 0.80 at neutral credibility);
- benign evidence alone under 0.5 (a lone link) is not enough and is inconclusive.

**Benign evidence cannot launder a risk signal.** If a risk signal (anything but the stylistic
`all_caps`, `repeated_exclamation` and `old_year_reference`) or an obfuscation is present next to
benign evidence, `no_risk` cannot win however strong that evidence is: "temblor: 300 muertos,
edificios colapsaron, evacuen se registro un sismo de magnitud 5" stays a suspected rumor, is marked
conflicting and goes to the LLM with a raised priority. The price is that a real report that cites
an official body WITHOUT a link (`official_attribution_unlinked` fires) is queued too, until the
catalog cross-check of a later unit can confirm it.

**Credibility is a prior, never a label.** The source's credibility (1 to 5, 3 by default) moves
the confidence of a label the evidence already chose by 0.03 per step away from 3: a more credible
source lowers a risk label's confidence and raises `no_risk`'s. With no evidence there is no label
at any credibility.

**Low confidence and the LLM queue.** A confidence strictly under the alert threshold (`status`
shows it; 0.5 by default) sets `low_confidence`. An in-scope item is marked `needs_llm` when it is
inconclusive, has conflicting signals, or has a risk label with a confidence under 0.8 (0.8 is
settled, 0.79 is not), unless it duplicates an item that was already judged. There is no dead zone
under 0.4: weak evidence is "inconclusive" and queued too, so the queue only ever shrinks as
evidence grows. The credibility prior can move a confidence a little under 0.4 without changing
that. The `priority` ranks the queue (strong evidence first, then how clearly on topic, with a bump
for the secretariat and for a laundering attempt). The hourly cap on LLM judgments is applied by
the stage that consumes the queue.

**Recycled and near-duplicate content.** Time is measured from when THIS application first saw
each item (`fetched_at`), never from the publication date, which the feed writes and can forge
(a backdated or future date changes nothing, and a forged "original" cannot make the real report
look recycled). Each in-scope item that is not itself an exact copy or recycled leaves a signature:
the content hash, a hash of the folded text, a hash of the numbers in it, its word count and a
MinHash of its 4-character shingles (80 values, no text). Lookups use indexes and never scan the
table; the cost of one grows with the number of stored signatures that share a band with the item,
and at most `SOCIAL_TRACKER_NEAR_DUPLICATE_CANDIDATES` (500) of them are verified, the ones
sharing the most bands first (a lookup that hits the limit is counted in the report).
- **Identical content** (same content hash or same folded text, looked up on its own and never
  limited by the candidate cap) within `SOCIAL_TRACKER_RECYCLE_DAYS` (7 days exactly is still
  inside) is a *duplicate*: it reuses the label and evidence of the earlier verdict with the
  credibility prior of its OWN source, `classify show` prints `duplicate of`, and it is not queued
  again. If the original is itself still waiting for the LLM the duplicate is flagged
  `awaiting_original` (a provisional verdict; the link is what the LLM stage follows to settle both).
  Further apart than that, in either direction (an old item arriving after a new one counts the
  same), it is `recycled_content` and references the other (`recycled of`) at 0.9, moved by the
  credibility prior.
- **A near copy** (estimated similarity at least `SOCIAL_TRACKER_NEAR_DUPLICATE_THRESHOLD`, 0.8,
  and BOTH texts of at least 12 words and with exactly the same numbers) never inherits a verdict:
  a magnitude, an hour or one digit changed makes it a different report. Within the window its
  rules run on its own text and it only links to the original (`near copy of`); beyond it it is
  `recycled_content` at 0.9 times the similarity.
- **Short texts** (fewer than 12 words) match only identical content inside the window: "Temblor en
  Cali" from a second source weeks later is not recycled, and a series of templated bulletins
  ("Sismo de magnitud X con epicentro en Y a las HH:MM") is not mistaken for copies.
- The similarity is an estimate with a standard error of `sqrt(J (1 - J) / 80)` (0.045 at 0.8).
  Measured on 3,000 random pairs of texts with 1 to 25 % of their words replaced, by TRUE
  similarity J: at about 0.7 the pair is a candidate 93 % of the time and accepted 5 % of the
  time; at about 0.8 it is a candidate 99.8 % of the time and accepted 58 % of the time (the
  threshold itself, so the noise decides); at about 0.9 it is accepted 98.5 % of the time. A
  one-word edit of a 20-word caption keeps about 0.87 and is caught in more than 95 % of the
  cases. `tests/classification/test_similarity.py` pins bounds around these numbers.

**The vocabulary** is editable data. `topics show` prints it; edit a copy and install it with
`topics set-file FILE -r "why"` (a new audited version; the same vocabulary again changes nothing)
or return to the shipped one with `topics reset -r "why"`. Matching is by whole words and phrases,
ignoring case, accents, punctuation, full-width forms and hidden characters, so `¡SÍSMO!` is
`sismo`, while `sismógrafo` is NOT `sismo` (list every form you want: singular, plural, verb
forms such as `tiembla`, `temblaba`, `está temblando`, adjective). Two ambiguous earthquake words
count only with context: `magnitud` needs a number within 3 words after it or an earthquake term
fewer than 8 words away, and `réplica`/`réplicas` an earthquake term fewer than 8 words away ("el alcalde replica a las
críticas" and "la magnitud del problema" are not quakes). The file is UTF-8, at most 64 KiB, and
refused when it has an unknown key, a duplicate (also across accents and case), a term in two
groups, a list over 500 terms, a term over 80 characters or 12 words, or an unknown or
out-of-range weight. Groups: `earthquake`, `secretariat` (the official name and its variants),
`other_disaster`, `places` (context only: a place alone never puts an item in scope), `exclude`
(crime terms: out of scope unless a disaster term is present), `official_bodies` (a bare "la
secretaría" counts, but "la secretaría de Hacienda" is another body), `official_domains`,
`official_channel_hints` and `weights` (per-signal overrides; 0 switches a signal off). Terms are
stored folded (lowercase, no accents).

Items already classified are not redone when the vocabulary changes. To judge again what the old
vocabulary called off topic, send it back: `classify reclassify --status out_of_scope [--limit N]`
forgets those verdicts and puts the items in the queue; the next `classify run` (or worker pass)
classifies them with the vocabulary in force. `--status classification_failed` retries items that
were parked (below). With a retention of 0 days an out-of-scope item is kept for its first 24 hours
so that this is possible.

**`official_domains` and `official_channel_hints` are placeholders: verify before relying on
them.** The shipped values (`official-domain.example`, `@official_channel_placeholder`) are not
real addresses and nobody has checked them. Put the domains you have verified yourself in
`official_domains`: each needs at least two labels and must not be a public suffix (`com`,
`gov.co`, `com.co`... are refused: a link "to" them proves nothing). A link to one of them lowers
the suspicion of an unverified attribution. The channel hints are informational only in this
version and drive no decision.

**Safety and behavior.** An item is classified only while global analysis is on, its source is
enabled and the source's analysis switch is on; this is checked again for every item, so a source
disabled in the middle of a run stops at the next item and its pending items stay untouched. Each
item is classified atomically and at most once (two runs racing over the same items never store
two verdicts); a storage error ends a run and the next one resumes. Any other error belongs to one
item: it is counted and the run goes on, and after three consecutive errors the item is parked as
`classification_failed` (it is skipped, `classify stats` counts it and the item is never purged by
a retention of 0) until `classify reclassify --status classification_failed` retries it. Changing
the vocabulary or the thresholds affects the items classified afterwards. Item text is untrusted
data: an instruction written inside an item is just text.

**What is stored, and for how long.** A verdict holds the label, the confidence, the signal names
and weights, the topics and a rationale made of signal names and vocabulary terms: never the text of
the item. A signature holds a SHA-256 of the title and text, a SHA-256 of the folded text, a
SHA-256 of its numbers, the word count and the MinHash: no text either, but a hash lets anyone who
has a candidate text confirm that it was seen here, and a MinHash reveals which texts are similar,
so treat the database as sensitive as the items. Signatures are forgotten 730 days after the item
was first seen; the verdict of an item that no longer exists is forgotten after
`SOCIAL_TRACKER_CLASSIFICATION_RETENTION_DAYS` (180) days, by the hourly retention purge; the
verdict of an item still stored is kept as long as the item.

**Calibrating.** The thresholds and weights are starting points, not measurements on real feeds.
What exists so far, all of it synthetic: a seeded corpus of 2,000 items whose mix is an assumption
(`tests/classification/corpus.py`), in which 13.0 % of the items (21.9 % of the in-scope ones) are
queued for the LLM, a corpus of 2,000 templated bulletins over 83 days in which no item is flagged
as a copy, and 65 labeled Spanish examples (`tests/data/gold_examples.yaml`, one of them recorded as a known
gap) that state what the cascade should do and are the seed of the gold set. The escalation rate depends on the real mix
(texts that cite an official body without a link are always queued for now), so read `classify
stats` and a sample of `classify show` on your sources, then adjust `weights` (and the term lists)
with `topics set-file`, and the thresholds with their environment variables. A hand-labelled gold
set and a tool to measure precision and recall against it come in a later slice; until then treat
the confidences as relative, not as probabilities.

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

Without a browser installed they are skipped. A test that hangs fails after 120 seconds
(`pytest-timeout`) with a dump of the stacks instead of blocking the run.

Dependencies are checked against the public advisory database with `pip-audit`:

```text
uv export --frozen --no-hashes -o requirements-audit.txt
uvx pip-audit -r requirements-audit.txt --no-deps --disable-pip
```

(The exported file is a throwaway: do not commit it.) Last run: 2026-10-27, on the locked
versions, `No known vulnerabilities found`; the lock file it audited:

```text
uv.lock sha256 96c8234a77c9da7b4771ad43d2e414edd861e37e205103d71b8955fed3d5c059
```

The line is informational: it says which dependency set the result belongs to. After any change
to `uv.lock` run the audit again and update the date and the hash (a test fails while they differ).

The architecture is hexagonal. Domain modules are pure, services depend on ports only, and
`social_tracker.bootstrap` is the composition root that wires the SQLite adapters into the
services. Entrypoints such as the CLI never import persistence; `lint-imports` enforces it.
The collector follows the same rules, and its fetcher adapter (`collector/http_fetcher.py`) is the
only module allowed to import an HTTP client or sockets: the web UI has no outbound path at all
and reaches the worker side only through the composition root. The classification core
(every module of `classification/` except `service`, `ports` and `yaml_io`, whichever is added
later) is pure and deterministic: an import contract forbids it storage, file, logging, thread and
network libraries, YAML and randomness, and a source check forbids calls to the current time.

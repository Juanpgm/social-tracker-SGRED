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
check, and a cross-check against the USGS and EMSC catalogs; see
[Classifying content](#classifying-content)); the language-model judgment, alerts and the GPU
workers are later slices. Nothing ever publishes or sends data outside the system: the only
outbound traffic is a read-only `GET` to the feeds you registered and to the two catalogs. The system starts **paused**, every source starts **disabled** and every analysis
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
| `SOCIAL_TRACKER_ALERT_RETENTION_DAYS` | How many days an alert is kept after its latest occurrence (0..3650); strictly older ones go. `0` keeps an alert only until it has been looked at (seen, dismissed) or withdrawn. See [Alerts](#alerts). | `180` |
| `SOCIAL_TRACKER_ALERT_DEDUP_HOURS` | Copies seen within this many hours of each other (inclusive) are one alert (1..720). | `24` |
| `SOCIAL_TRACKER_ALERT_CRITICAL_WINDOW_HOURS` | How fresh an event must be (hours between its publication and the first sighting, inclusive) for a catalog contradiction to be critical (1..720). | `24` |
| `SOCIAL_TRACKER_ALERT_CRITICAL_DELIBERATE` | Confidence from which `deliberate_disinformation` is critical (above 0, up to 1, at most four decimals). | `0.85` |
| `SOCIAL_TRACKER_ALERT_CRITICAL_CATALOG` | Confidence from which a catalog contradiction about a fresh event is critical (above 0, up to 1). | `0.8` |
| `SOCIAL_TRACKER_ALERT_HIGH` | Confidence from which an alert is `high` (0..1; not under the medium one). | `0.7` |
| `SOCIAL_TRACKER_ALERT_MEDIUM` | Confidence from which an alert is `medium`, below it `low` (0..1). | `0.5` |
| `SOCIAL_TRACKER_ALLOW_PRIVATE_HOSTS` | Comma-separated host names (or IP addresses) the collector may contact although they resolve to a private address. See [Collecting content](#collecting-content). | empty (all private addresses are blocked) |
| `SOCIAL_TRACKER_ALLOWED_PORTS` | Comma-separated destination ports the collector may contact, for feeds and every redirect hop. A value replaces the default list. | `80,443,8080,8443` |
| `SOCIAL_TRACKER_CATALOG_CENTER` | `latitude,longitude` of the region the official-catalog cross-check cares about. | `3.4516,-76.5320` (Cali) |
| `SOCIAL_TRACKER_CATALOG_RADIUS_KM` | Radius of that region in km (above 0, up to 20000). | `300` |
| `SOCIAL_TRACKER_CATALOG_MIN_MAGNITUDE` | Smallest event kept inside the region (-2..10). | `2.5` |
| `SOCIAL_TRACKER_CATALOG_GLOBAL_MIN_MAGNITUDE` | Smallest event kept anywhere in the world, for earthquakes felt elsewhere (-2..10). | `5.5` |
| `SOCIAL_TRACKER_CATALOG_INTERVAL_S` | How often the worker refreshes the catalogs, in seconds (60..86400). | `600` |
| `SOCIAL_TRACKER_CATALOG_MAX_AGE_S` | A cache whose last good refresh is older than this is stale and the check says `unavailable` (from the interval up to 604800). | `7200` |
| `SOCIAL_TRACKER_CATALOG_INGEST_GRACE_S` | How long the catalogs may take to publish an earthquake, in seconds (0..86400). A claim newer than the last refresh minus this is confirmed if the cache has it and otherwise `unavailable`, never a mismatch. | `1800` |
| `SOCIAL_TRACKER_LLM_BASE_URL` | Address of the model server (Ollama). Only `http://127.0.0.1` or `http://localhost`, with an optional port, is accepted: anything else stops the start. See [The language model](#the-language-model-stage-5). | `http://127.0.0.1:11434` |
| `SOCIAL_TRACKER_LLM_MODEL` | Model that judges (a name known to the server). | `st-gemma4-e4b` |
| `SOCIAL_TRACKER_LLM_FALLBACK_MODEL` | Model tried once when the first is not installed or cannot be loaded; `none` turns the fallback off. | `st-qwen35-4b` |
| `SOCIAL_TRACKER_LLM_KEEP_ALIVE_S` | Seconds the server keeps the model loaded after a call; when they pass the GPU is freed (1..86400). | `300` |
| `SOCIAL_TRACKER_LLM_TIMEOUT_S` | Hard deadline of one call to a model that is already loaded, response headers and answer included, in seconds (1..600). | `60` |
| `SOCIAL_TRACKER_LLM_LOAD_TIMEOUT_S` | Hard deadline of a call that has to LOAD the model first (it is not resident: the first call, or one after the keep-alive ran out), in seconds (1..1800; never shorter than the call deadline). A timeout while loading is the server's, not the item's. | `180` |
| `SOCIAL_TRACKER_LLM_NUM_CTX` | Context window sent to the server (2560..262144); the item text is cut, deterministically and flagged, so that the prompt fits next to the room for the answer. | `8192` |
| `SOCIAL_TRACKER_LLM_PASS_BUDGET_S` | Wall-clock budget of one escalation pass, in seconds (1..86400), on top of the item count. | `120` |
| `SOCIAL_TRACKER_LLM_HOURLY_CAP` | Judgments started per UTC hour (1..100000); the rest waits. | `200` |
| `SOCIAL_TRACKER_LLM_QUEUE_MAX` | Items that may wait for the model (1..1000000). Beyond it the excess of a source over its fair share goes first, then the LOWEST priority (the oldest among equals); dropped items keep their CPU verdict and are counted. | `2000` |
| `SOCIAL_TRACKER_LLM_SOURCE_SHARE` | The share of the queue bound one source may hold when the queue overflows (above 0 up to 1, at most four decimals). | `0.25` |
| `SOCIAL_TRACKER_LLM_BATCH` | Items judged per worker pass (1..1000). | `20` |
| `SOCIAL_TRACKER_LLM_MAX_FAILURES` | Failures that may be the item's own (a timeout of a loaded model, a body that is not an answer) before the item is parked (1..100). They count only while the model is answering other items; a refused request, a 429, a 5xx, a slow load or a dead server is the server's and never counts. | `3` |
| `SOCIAL_TRACKER_LLM_PARKED_TTL_S` | A parked item goes back to the queue after this many seconds (60..2592000); `llm requeue-parked` does it at once. | `86400` |
| `SOCIAL_TRACKER_LLM_RETRY_BASE_S` | Wait after the first failed call, in seconds; it doubles per failure (1..86400). | `60` |
| `SOCIAL_TRACKER_LLM_RETRY_CAP_S` | Longest wait between two tries of the same item (at least the base, up to 604800). | `3600` |
| `SOCIAL_TRACKER_LLM_PROMPT_CHARS` | An item text longer than this is read at both ends and flagged in the request (100..100000). | `6000` |
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
| `catalog refresh [--dry-run]` | Fetch the USGS and EMSC catalogs now and store their events (one-shot; `--dry-run` fetches and reports, stores nothing). Only while global analysis is on: exit status `3` otherwise. |
| `catalog status` | The catalog cache: events per catalog, last good refresh, failures, `[STALE]` and `[CUT SHORT]` marks. |
| `classify run [--limit N]` | Classify the pending items of the eligible sources (CPU only). The worker does it after every collection pass. Exit status `3` when no source is eligible. |
| `classify escalate [--limit N] [--dry-run]` | Send the items the CPU stages could not settle to the local language model (one-shot; the worker does it after every pass). `--dry-run` says how many would be sent (not counting those that settle without a call) and how many a full queue would drop, and changes nothing. Exit status `3` when it could do nothing (no source is eligible, or nothing could be judged because the model server is unreachable, is not Ollama, is resting after a failure, another process holds the model, or the run was stopped) and `1` when a storage error stopped it or the server failed after some items were judged. |
| `llm requeue-parked [--limit N]` | Put parked items (given up on after repeated failures of their own) back in the queue now, instead of after `SOCIAL_TRACKER_LLM_PARKED_TTL_S`. Sends nothing to the model and needs no server. |
| `llm status` | Whether the address answers like Ollama (a loopback port can be held by any local process: nothing else is asked of something that does not identify itself), which models it has and which configured ones are missing, the queue, the hourly budget and the counts of the stage. A server that is down or is not Ollama is reported, exit status `0`. |
| `llm ping` | One tiny harmless question to the model, to see that it answers (it loads the model). Run it yourself; nothing else ever calls it. Exit status `1` when it did not answer. |
| `classify reclassify --status S [--limit N]` | Send the items at status `out_of_scope` (after a vocabulary change) or `classification_failed` back to the queue; the next run classifies them again. |
| `classify show ITEM_ID` | The verdict of one item: label, confidence, deciding stage, rationale, flags, when and why it was settled by the language-model stage (`llm`, `duplicate`, `invalid` or `parked`) and, if the model saw it, which model and prompt version. |
| `classify stats` | Verdicts by label, out of scope by reason, the flags, near copies, duplicates waiting for their original (and, apart, the copies whose original is gone, which the next pass releases), the items parked as failed and the counts of the language-model stage (judged, invalid answers, waiting for a retry, parked, dropped). |
| `classify calibrate [--file F] [--with-llm]` | Measure the CPU cascade against labeled texts (the local gold set, or `--file`): accuracy, precision and recall per label, confusion matrix, escalation rate and a sweep of the alert threshold. A report only; `--with-llm` (never the default) also puts the escalated texts to the local model. Exit status `3` when there is nothing to measure. See [Gold set and calibration](#gold-set-and-calibration). |
| `alerts list [--severity S] [--label L] [--source ID] [--status S] [--tag T] [--since D] [--until D] [--limit N] [--offset N]` | The alerts, newest first (withdrawn ones only with `--status withdrawn`). Dates are days (`2026-03-01`, the `--until` day inclusive) or ISO moments, UTC. |
| `alerts show ALERT_ID` | One alert: what it says, its evidence, the items behind it and its audit trail. |
| `alerts seen ALERT_ID` / `alerts dismiss ALERT_ID` / `alerts reopen ALERT_ID` `--reason R [--actor A]` | Mark an alert (audited). `reopen` brings a dismissed alert back as seen. A repeat changes nothing. |
| `alerts refresh` / `alerts rebuild` | Turn the new and changed verdicts into alerts now (the worker does it every pass) / evaluate every verdict again with the threshold in force. Both are idempotent; neither deletes an alert. |
| `alerts stats` | Alerts by status and by severity. |
| `alerts threshold show` | The minimum confidence from which a risk verdict raises an alert. |
| `alerts threshold set VALUE --reason R [--actor A]` | Set that threshold (audited). Existing alerts follow it only on the next `alerts rebuild`. |
| `gold add --label L [--text-file F] [--item ITEM_ID] [--note N] [--credibility N] [--url U] [--actor A]` | Label a text for the gold set; give exactly one of `--text-file` and `--item`. |
| `gold label [--text-file F] [--item ITEM_ID] [--label L] [--note N] [--credibility N] [--url U] [--actor A]` | The same, asking for the label (and the note) on the terminal when they are not given. |
| `gold list [--label L] [--limit N] [--offset N]` | The labeled texts in the order they were added. |
| `gold show ID` / `gold remove ID` | One labeled text whole / forget one. |
| `gold export [-o, --output FILE] [--force] [--label L]` | Write the gold set as YAML to stdout or a file (it can hold real text: do not commit it). |
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
| `1` | The operation was refused (invalid input, conflict, unknown source, storage failure...): one `error: ...` line on stderr, nothing on stdout. `classify escalate` also uses it when a storage error stopped the run or the server failed after some items were judged, `llm ping` when the model did not answer, and `alerts refresh` / `alerts rebuild` when some items could not be evaluated (their result lines, if any, are still printed). |
| `2` | Usage mistake (missing option, unknown command). |
| `3` | `classify run` or `classify escalate` found no eligible source (global analysis off, or no enabled source with its own switch on), so it had nothing to do; `classify escalate` also uses it when it could judge nothing because the model server was unreachable (or is not Ollama, or is resting after a failure), another process held the model, or the run was stopped; or `catalog refresh` ran while global analysis is off; `classify calibrate` also uses it when there is nothing to measure (an empty gold set, or a file with no labeled example). It is not a failure: a run with eligible sources and nothing pending exits `0`. |
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
(the collection worker has no GPU part). Next to it a read-only **Language model** panel shows
the stored figures of the escalation (configured models, verdicts waiting, judgments started this
hour of the cap, counts); it never asks the model server anything, so whether the server answers
is `social-tracker llm status`.

The **Alerts** page (`/alerts`, linked from every page, with a badge counting the new ones) lists
the alerts newest first, 25 to a page: filters for severity, label, status, source, tag and a
date range (a plain GET form, so a filtered page can be bookmarked), the occurrence count, the
excerpt and **Mark seen**, **Dismiss** and **Reopen** buttons that update the row in place. The
detail page shows the evidence, the items behind the alert and its trail; it links nowhere
outside this machine. A mark that cannot be applied (another tab already changed the alert, a busy
database) redraws the row as it really is with a message; pressing a button twice is harmless.
Every stored string is escaped, and there is no publish, share, mail, draft or approve action
anywhere: alerts are stored and shown here, on this machine only.

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
source and to the two official earthquake catalogs (see [Classifying
content](#classifying-content)). It runs on the CPU only.

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
or sent anywhere. The first four stages of the cascade run on the CPU, need no model and no
network, and are deterministic: the same item, history and vocabulary always give the same verdict. Within one run
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
4. **Official-catalog cross-check** (earthquake items only), described below. It reads a local
   cache and runs before the item's write transaction opens, so a slow catalog never holds the
   database. The verdict marks what the language model will have to settle (`needs_llm`, with a
   `priority`).
5. **Language-model judgment** (optional, after the stages above, on this machine only), described
   in [The language model](#the-language-model-stage-5).

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

| `catalog_mismatch` | unintentional_rumor | 0.30 | the official-catalog cross-check contradicts the claimed magnitude (no claimed magnitude within 1.0 of any compared event) or finds no event in the time window of a complete cache; set by the check, not read from the text. Too weak to decide a label alone: it adds to other rumor evidence and sends a clean-looking report to review |

**The official-catalog cross-check** (stage 4). For an in-scope earthquake item the cascade
compares what the text claims with what two official catalogs recorded: the USGS and the EMSC
FDSN event services (`earthquake.usgs.gov`, `www.seismicportal.eu`), and no other host. The
Servicio Geológico Colombiano is excluded: it is never fetched and has no configuration. The
cascade itself never touches the network: the worker refreshes a local cache of events (every
10 minutes by default, `catalog refresh` does it on demand) and the check reads only that cache,
before the item's write transaction opens.

- **What is read from the text:** a magnitude (`magnitud 7.5`, `M6.2`, `temblor de 5,8`,
  `6.1 grados Richter`, several magnitudes such as `sismo de 5.5 y luego otro de 4.0`, and ranges
  of at most one degree; a number above 10 or with a unit after it, such as `50 m2`, is not a
  magnitude and a wider range such as `magnitud 4 a 7` is dropped), a time (`hace un momento`,
  `hace 5 minutos`, `a las 3:15 pm`, `a las 3 de la tarde`, read as Colombia time unless it says
  UTC; only when it sits in a sentence about the earthquake; a 12-hour time with no am/pm and no
  part of the day, `12 de la mañana`, a date other than the publication day, `ayer`, `hace 3
  días` or `hace años` claims no time or is not checked) and a place. A hypothetical, future or
  vague text (`simulacro`, `habrá`, `podría`, `mañana`, `guía`, `qué hacer ante`, `esta
  madrugada`, `este lunes`) claims nothing. Only numbers, an instant and a plain lowercase place
  name are kept; nothing of the text is interpreted.
- **What is compared:** the events within 6 hours of the claimed instant (or of the publication
  when the text names none), either within 300 km of Cali (centre and radius are configurable)
  or of magnitude 5.5 or more anywhere (an earthquake felt elsewhere). A claim that names a place
  outside the region (anything but Colombia, Cali, the Valle del Cauca and the other places of
  the default list) is compared only with the events of magnitude 5.5 or more, and a claim below
  what the cache holds (2.5 in the region, 5.5 elsewhere) cannot be verified. With a named instant
  the events nearest that instant are compared (those within two minutes of the nearest count
  together, so two catalogs that disagree about one earthquake do not contradict a faithful
  claim); with none, every event of the window is, so an aftershock does not hide the mainshock.
  The same earthquake in both catalogs counts once, and the claim is compared with the closest
  of its magnitudes.
- **The outcome**, recorded as `catalog:` in the verdict: `consistent` (every claimed magnitude is
  within 1.0 of a compared event), `mismatch` (no claimed magnitude is, or no event in the
  window while both catalogs are fresh and answered in full), `unavailable` (no catalog
  refreshed successfully within the maximum age, or the cache is empty, or the event is absent
  while a catalog is stale or its last answer was cut short, or the claim is newer than the last
  refresh minus the ingest grace so that the catalogs may not have published it yet, or it
  names a place outside the region and no large event was recorded, or only some of the claimed
  magnitudes match), `no_claim` (no magnitude, or a time the cache cannot answer) and
  `not_checked` (out of scope). `unavailable` never blocks a classification and never raises
  risk by itself.
- **What it changes:** a `mismatch` adds the `catalog_mismatch` signal (unintentional_rumor,
  0.30): it adds to the other rumor evidence and sends a clean-looking report to review, and it
  cannot decide a label alone. A `consistent` claim lowers an `official_attribution_unlinked`
  signal to half its weight; it never creates a label or a signal by itself.
- **The cache:** `catalog status` shows events per catalog, the last good refresh, failures and
  staleness. A failed catalog (timeout, DNS, 4xx/5xx, `Retry-After`, an answer that is not what
  was asked for, a redirect off the two hosts) is never an error of the run: it is retried with
  exponential backoff (60 s doubling, at most an hour) and the other catalog is still stored. An
  empty answer (`204 No Content`) is a quiet day, not a failure. Only earthquakes are kept
  (quarry blasts and explosions are dropped and counted). An answer that reaches the request
  limit may be cut short: it is stored but `catalog status` marks it `[CUT SHORT]` and the
  cache is not complete until a shorter answer. Events a complete, non-empty answer no longer
  lists are removed. Events older than 14 days are purged by the hourly housekeeping. The worker
  refreshes the catalogs after the classification of each pass and stops between requests when
  a stop is requested. Like the sources, the catalogs
  are fetched only while global analysis is on: `analysis off` stops the refresh, and `catalog
  refresh` refuses to run (exit status `3`). The cache keeps the origin time, position, magnitude
  and place of each event; the place is catalog text, stored as data and escaped where shown.

```bash
uv run social-tracker catalog status
```

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
catalog cross-check can confirm it (a `consistent` claim halves that signal).

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
the stage that consumes the queue (see [The language model](#the-language-model-stage-5)).

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
  A verdict the language model gave is shared only with a copy whose stored text is the original's
  TO THE BYTE: a copy that differs by punctuation, case, spacing, a hidden character, a look-alike
  letter or another address is not given the model's verdict; it is linked as a near copy and judged
  for itself.
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

### The language model (stage 5)

The cascade settles most items on the CPU. The ones it cannot settle (`needs_llm`: inconclusive,
conflicting, or a risk label under 0.8; 0.8 is settled, 0.79 is not; never a copy of an item that
was already judged) go to a language model that runs **on this machine**, through
[Ollama](https://ollama.com). It is optional: without a server the verdicts of the CPU stages
simply stay as they are. See `docs/howto.md` for how to run Ollama and which models to install.

- **What is sent, and where.** The text of the item and the evidence the earlier stages found
  (signals, topics, the catalog outcome, the credibility of the source as a weak prior) go to the
  address in `SOCIAL_TRACKER_LLM_BASE_URL` and nowhere else. That address is checked when the
  application starts: only `http://127.0.0.1` or `http://localhost` (optionally with a port) is
  accepted, and the connection is always made to the literal address 127.0.0.1. It is the only
  plain-http address of the application. No redirect is followed, no proxy variable is honoured,
  every call has a hard deadline (`SOCIAL_TRACKER_LLM_TIMEOUT_S`, 60 s) and an answer larger than
  256 KiB is refused. An import-linter contract keeps every other module from opening a
  connection.
- **The call.** `POST /api/chat` with `stream: false`, the answer schema as `format`, temperature
  0, seed 0 (a retry uses another), an explicit `num_ctx` and `num_predict`, and `keep_alive` of
  300 s: the server unloads the model, freeing the GPU, 300 s after the last call. `think: false`
  is sent for every model (a `thinking` field in an answer is never read); if the server rejects
  the field for a model, that model is asked once more without it and remembered. The default
  model is `st-gemma4-e4b`; if it is not installed or cannot be loaded, `st-qwen35-4b` is tried
  once. The whole exchange, response HEADERS included, runs under one wall-clock deadline that is
  enforced inside every connect, read and write (the layer the feed fetcher uses): a peer that
  drips a byte at a time cannot hold the worker. A call to a model that is probably not resident
  gets the longer `SOCIAL_TRACKER_LLM_LOAD_TIMEOUT_S`; the lease that keeps two processes apart
  outlives the longest possible judgment (a cold load, then the retry).
- **Who is listening.** A loopback port can be held by any local process, and the item text goes
  to whatever answers there. Before any item is sent, and every five minutes, the application
  asks the address who it is (`Ollama is running` at `/`, or a version at `/api/version`) and
  sends nothing to something that does not say so (`llm_not_ollama`, which rests the judge like a
  failure; `llm status` reports it). This is a check, not authentication: pick a port you
  control, and do not run other things on the address you configure.
- **The item text is data.** It travels as one JSON string between two markers that carry a
  digest of the text itself, after an instruction that says it is untrusted, must never be
  followed, that there are no tools and that the only output is the schema. Text such as "ignore
  previous instructions", a fake JSON verdict, a delimiter or a role marker cannot end the block,
  start a line of its own or change the schema. The prompt is a request, not a boundary: the answer
  is validated again.
- **The prompt spells the answer out.** The instruction names the four fields (`label`,
  `confidence` as a fraction from 0.0 to 1.0, never a percentage, `rationale`, `evidence`), their
  limits and the meaning of each label; the schema in `format` is not the only place the model
  reads it. A percentage is NOT rescaled: `85` is invalid. The item text is cut,
  deterministically and flagged, so that the prompt (and the correction of a retry) fits in
  `SOCIAL_TRACKER_LLM_NUM_CTX` next to the room for the answer (about 2.5 characters per token,
  deliberately conservative).
- **The answer is validated strictly.** Exactly one JSON object with one of the four labels
  (`unintentional_rumor`, `deliberate_disinformation`, `recycled_content`, `no_risk`), a finite
  confidence from 0 to 1, a rationale and up to five evidence strings, nothing else. Unknown
  labels, out-of-range or non-numeric confidences, extra or missing fields, duplicate keys, prose
  or a code fence around the JSON, truncated or empty output are invalid. Invalid output is asked
  for **once more**, and the second try is not a byte-for-byte repeat: it carries another seed and
  a corrective message that names the CLASS of the problem (`not_json`, `schema`, `cut`...) and
  never quotes what the model wrote; after an answer cut by the token limit it also has more room
  to answer. Two invalid answers in a row leave the verdict of the earlier stage as it was (the
  judgment is recorded as `invalid`, the item leaves the queue flagged `low_confidence`), but only
  while the judge has been answering OTHER items in the same pass or the last hour: a judge that
  answers nobody validly is sick, so the item stays queued and the judge rests. A rationale is
  cleaned of control and hidden characters and cut to 300 characters; nothing else of the answer
  can change what the application does. A judged label under 0.5 is applied but flagged
  `low_confidence`; the credibility of the source is recorded and never moves a judged confidence.
- **A valid answer can still be suspicious.** The confidence of a model is capped at 0.95. A
  `no_risk` is NOT final when the rules had found something it clears: an obfuscated text
  (`obfuscated_text`), an injection-style signal, a risk signal or a risk verdict of confidence
  0.6 or more, or a claim the catalog contradicts. Such a `no_risk` is applied but capped at 0.7,
  flagged `low_confidence`, noted for review in its rationale and counted (`contradicted` in the
  report, `llm_contradicts_prior` in the log); the label of the model is never above what the
  earlier stages' evidence allows it to claim without a person looking.
- **Order and budget.** Highest `priority` first, the oldest first among equals, at most
  `SOCIAL_TRACKER_LLM_BATCH` (20) per worker pass and at most `SOCIAL_TRACKER_LLM_PASS_BUDGET_S`
  (120 s) of wall-clock time per pass, and at most `SOCIAL_TRACKER_LLM_HOURLY_CAP` (200)
  judgments per UTC hour; an invalid answer's one retry belongs to the same judgment, and a call
  that failed before the model gave anything (nothing listening, a refusal, a 5xx, a slow load) is
  refunded, so a server that is down does not burn the budget. The stop flag is checked between
  items and between the two calls of a judgment. The queue is bounded by
  `SOCIAL_TRACKER_LLM_QUEUE_MAX`; it counts what could be served now (not items waiting out a
  backoff, not items of a source that is switched off). Beyond it, a source that holds more than
  `SOCIAL_TRACKER_LLM_SOURCE_SHARE` (25 %) of the bound loses its own excess first, so one source
  cannot crowd out the others; if the queue is still over, the LOWEST priority goes first and the
  oldest among equals (an older high-priority item is never dropped before a newer low-priority
  one). Dropped items keep the CPU verdict, are flagged `low_confidence`, settle as `parked` and
  are counted in `classify stats` and `llm status`. `classify escalate --dry-run` reports how many
  it would drop and drops none. Items waiting for the model are not purged by the retention purge
  until they are twice as old as the retention (twice the grace, with a retention of 0), and what
  the model said about a purged item (the audit record with its raw output) is deleted with it.
- **When the server fails.** There are two kinds of failure, and only one of them is ever held
  against an item. The SERVER's: nothing listening, something that is not Ollama, a missing model,
  a refused or rate-limited request (400, 401, 403, 429...), a 5xx (an out-of-memory runner
  included) and a timeout while the model was being loaded. They never count against the item:
  the reservation of the hour is given back, the item goes behind the others, the whole judge
  rests (60 s, 120 s, 240 s... up to an hour; it starts again from 60 s with the first success)
  and the batch ends. The ITEM's: a timeout of a model that was already loaded and a body that is
  not an answer. They count against the item (`classify show` prints the next attempt) and the
  item is parked after `SOCIAL_TRACKER_LLM_MAX_FAILURES` (3) of them, but only while the model is
  answering OTHER items in the same pass or the last hour; otherwise they are the server's. Two
  such failed items in a row end the batch. Because every server failure rests the judge, a
  server that crashes on every item is probed about once per rest and parks nothing; the price is
  that a single item that crashes the runner (a 5xx) is retried with a growing delay (up to an
  hour) and is never parked for it. A parked item keeps its CPU verdict (settled as `parked`) and
  comes back to the queue, behind the healthy ones and with a clean slate, after
  `SOCIAL_TRACKER_LLM_PARKED_TTL_S` (24 h) or when `llm requeue-parked` says so.
- **Copies.** An exact copy that waited for its original (`awaiting_original`) takes the judged
  label and confidence when the original is judged only if its text is the original's to the byte;
  it keeps its OWN source credibility on its record and says so in its rationale, and it costs no
  call. A copy that differs by a character is queued and judged for itself (linked as a near
  copy). If the original gets no valid judgment (invalid, parked, dropped) the copy keeps the
  provisional verdict it had, flagged `low_confidence` like the original, and stops waiting. A
  copy never waits for ever: if its original is purged, pruned, settled without it or belongs to a
  source that is off, the next pass (or the pruning itself) releases it into the queue as a normal
  item. A copy whose own source was switched off meanwhile is left alone (PC-2).
- **One at a time, and never inside a transaction.** A lock in the process and a lease in the
  database keep two workers (or a worker and `classify escalate`) from loading the GPU together
  (a lease left by a crashed process expires). The call is made with no transaction open; the
  answer is then written in one short transaction that first checks that the item still exists,
  is still classified, its source may still be analysed and its verdict still waits. A source
  switched off, an item purged or a verdict settled meanwhile discard the answer instead of
  writing it, so a run that is repeated or races another never judges an item twice. Global
  analysis off sends nothing.
- **What is kept.** `classify show` prints which model and prompt version judged an item and when
  and why its verdict was settled (`settled_at`, `settle_reason`: `llm`, `duplicate`, `invalid`,
  `parked`), which is what a later alert stage reads. The
  judgment (label, confidence, rationale, evidence, model, prompt version, the verdict it replaced
  and the raw answer, cut to 2,048 characters) is stored for the audit. Logs name counts, ids and
  codes; never item text, rationales or model output.

## Audit log

Each real change writes one event, in the same transaction as the change, with the actor and the
reason. A request that changes nothing (enabling an enabled source, importing known data) writes
no event. Events are append-only and survive the deletion of the source they describe.

## Alerts

An alert says that a stored verdict deserves a person's attention. It is **stored and shown on
this machine only**: there is no publication, webhook, e-mail, draft of a debunk or approval flow
anywhere in the application, and the import rules keep it so (the alerts module opens no
connection and imports no network or mail client).

**What alerts.** A verdict raises an alert when its label is `unintentional_rumor`,
`deliberate_disinformation` or `recycled_content` and its confidence is **at least** the alert
threshold (default `0.5`, inclusive; `alerts threshold set` changes it, audited with an actor and
a reason). `no_risk`, `unclassified` and `out_of_scope` never alert, whatever their confidence.
The step runs in the worker after the classification and the language model of every pass, and
by hand with `alerts refresh` (what is new or changed) or `alerts rebuild` (everything, with the
threshold in force). It reads only what is stored; running it twice changes nothing.

| Severity | When (every number is configurable, see the variables above) |
| --- | --- |
| `critical` | `deliberate_disinformation` at 0.85 or more; or the official catalogs contradict the claim (`catalog: mismatch`) about an event at most 24 hours old (inclusive) at 0.8 or more. |
| `high` | 0.7 or more. |
| `medium` | 0.5 or more. |
| `low` | Anything else that qualifies (only possible when the threshold is under 0.5). |

The age of an event is the time between the item's publication date and the moment this
application FIRST saw it, never under zero: the first sighting is the one clock a feed cannot
forge.

**What an alert holds.** Its label, severity, confidence and rationale, the source (id and a
snapshot of its name, so deleting the source keeps the name and the pages say `source removed`),
the item it came from, an excerpt, its times and the evidence: the deciding stage, what the
catalogs said, the signals, whether the verdict is still provisional (it waits for the model) and,
when the model judged it, which model and prompt version. **Tags** make the notable evidence
filterable: the topics the filter found (`earthquake`, `secretariat`, `other_disaster`) and
`secretariat_attribution` (a claim attributed to the Secretaria de Gestion del Riesgo de
Emergencias y Desastres de Cali, formerly DAGRD, with no link to it), `catalog_contradiction`,
`prediction` and `obfuscated_text`.

**The excerpt** is at most 500 characters (a longer text is cut and ends with an ellipsis).
Control characters, hidden characters (zero width, right-to-left overrides, line separators, lone
surrogates) are removed, a link keeps only its scheme, host and path (credentials, query string
and fragment go: that is where tokens live), and anything that looks like a cookie, an
authorization header, a token, a password or a long hex key is replaced by `[redacted]`. It is
stored as plain text and escaped wherever it is drawn.

**Copies are one alert.** The same content, or a near copy that points to an original, seen within
24 hours (inclusive, either way round) of an alert's items joins that alert: the occurrence count
goes up and the sources are linked. An item that bridges two alerts makes them one (the oldest
stays, and the one that needs attention most decides the status). The severity of an alert is the
highest of its active items, so merging a copy can raise it and never lower it.

**One item, one alert.** An item is a member of at most one alert however often it is reprocessed,
reclassified or settled by the model, and two workers racing on it store one. When its verdict
changes (the model upgrades or downgrades it, a copy takes its original's) the alert follows. When
no item of an alert qualifies any more (the verdict fell under the threshold or became `no_risk`)
the alert is **withdrawn** with the reason, not deleted: it leaves the default list (`alerts list
--status withdrawn`, or the Alerts page filter, shows it), and it comes back by itself, keeping the
status the operator gave it, if a verdict qualifies again. A raised threshold withdraws on
`alerts rebuild`, not on a plain refresh.

**Statuses.** `new`, `seen`, `dismissed`. The moves are `new` to `seen`, `new` to `dismissed`,
`seen` to `dismissed` and `dismissed` to `seen` (`alerts reopen`); anything else is refused, and
asking for the status an alert already has changes nothing and writes nothing. Every change keeps
an audit entry with the actor and the reason (the web UI uses the actor `web:local`).

**Retention.** An alert is kept `SOCIAL_TRACKER_ALERT_RETENTION_DAYS` (180) days after its latest
occurrence and the hourly purge forgets the older ones (strictly older: at exactly 180 days it is
still kept). With `0` an alert is kept only until it has been seen, dismissed or withdrawn. The
excerpt and the evidence outlive the item they came from (the item is purged after
`SOCIAL_TRACKER_RETENTION_DAYS`), and a forgotten alert is not raised again by the verdict of an
item that still exists.

```bash
uv run social-tracker alerts threshold show
uv run social-tracker alerts threshold set 0.6 --reason "fewer low-confidence alerts"
uv run social-tracker alerts refresh
uv run social-tracker alerts list --severity critical --since 2026-03-01
uv run social-tracker alerts stats
```

## Gold set and calibration

The cascade is only as good as it is measured. The **gold set** is a list of texts a person
labeled with the answer they want from the classifier (one of the six labels). It lives in the
local database next to the collected items and **never in the repository**: the texts may be real
content from real sources, so `gold export` writes them to a file only when you ask, and the file
should stay out of version control. The synthetic seed in the repository
(`tests/data/gold_examples.yaml`, the same 65 examples the tests use) holds no real content.

```text
uv run social-tracker gold add --label unintentional_rumor --text-file chain.txt --note "prediction chain"
uv run social-tracker gold add --label no_risk --item ITEM_ID --credibility 4
uv run social-tracker gold label --text-file other.txt
uv run social-tracker gold list
uv run social-tracker gold export -o my-gold.yaml
uv run social-tracker classify calibrate
uv run social-tracker classify calibrate --file tests/data/gold_examples.yaml
```

A text comes from a file (UTF-8, at most 100 KiB, at most 20,000 characters kept) or from a
collected item (`items list --ids`). It is cleaned like an item text, and the same text is the
same example: adding it again with the same label changes nothing, with another label is refused
until you `gold remove` it. The set holds at most 5,000 examples.

`classify calibrate` runs the CPU cascade (the topic filter and the rules) over the set and prints
a report: accuracy; precision, recall and F1 per label; the confusion matrix (rows are the label
you expected, columns the one the cascade gave); the share of texts that would go to the language
model; and a sweep of the alert threshold from 0.30 to 0.95 in steps of 0.05 (the one in force is
marked) with the alerts each value would raise, how many were right and how many risky texts it
would miss. The threshold with the best F1 (then the best precision, then the lowest) is
**suggested; it is a report only and nothing is ever applied**: no verdict, setting or file
changes, and you decide whether to run `alerts threshold set`. The cascade is run without history
(no duplicate or recycled check) and without the catalogs, and a text that rules leave between
0.4 and 0.8 is exactly one the model would settle later, so read the sweep as the behaviour of
the CPU stages alone. An example file may name several acceptable labels; an entry that names none
is left out and counted.

`--with-llm` (never the default) also puts the escalated texts, one at a time, to the model server
on this machine (the address is `SOCIAL_TRACKER_LLM_BASE_URL`, loopback only, and the server is
identified before any text is sent). The answer replaces the cascade's for that text in the report
only; nothing is stored, no hourly budget is used, and after three failures in a row it stops
asking and scores the rest with the cascade alone.

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
uv.lock sha256 5569fb3cbe316e1d0b62c016faf3e66ad42fbab7c46348d9e65c29596adef06f
```

The line is informational: it says which dependency set the result belongs to. After any change
to `uv.lock` run the audit again and update the date and the hash (a test fails while they differ).

The architecture is hexagonal. Domain modules are pure, services depend on ports only, and
`social_tracker.bootstrap` is the composition root that wires the SQLite adapters into the
services. Entrypoints such as the CLI never import persistence; `lint-imports` enforces it.
The collector follows the same rules, and its fetcher adapter (`collector/http_fetcher.py`) is the
only module allowed to import an HTTP client or sockets (the catalog refresher reuses it, with a
host allowlist, and has no I/O of its own): the web UI has no outbound path at all
and reaches the worker side only through the composition root. The classification core
(every module of `classification/` except `service`, `ports` and `yaml_io`, whichever is added
later) is pure and deterministic: an import contract forbids it storage, file, logging, thread and
network libraries, YAML and randomness, and a source check forbids calls to the current time.

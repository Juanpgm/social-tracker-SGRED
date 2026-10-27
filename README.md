# Social risk monitor

Tracks social-media signals and surfaces risk. This release (slice 1) provides the foundation:
a registry of **sources** (news RSS feeds, Instagram accounts, live streams, recorded audio), the
**analysis switches** (system-wide and per source), a SQLite store and an append-only **audit
log** in which every change records who made it and why.

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
| `SOCIAL_TRACKER_DB_PATH` | SQLite database file. Missing parent directories are created. | `data/social_tracker.db` (relative to the working directory) |
| `SOCIAL_TRACKER_ACTOR` | Who the audit log attributes your changes to. `--actor` overrides it. | `cli:<OS user name>` |
| `SOCIAL_TRACKER_BUSY_TIMEOUT_MS` | How long a write waits for another process using the database (1..600000). | `5000` |

A blank value counts as unset. The `data/` directory is git-ignored.

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

### Commands

| Command | What it does |
| --- | --- |
| `sources list` | One row per source: id, type, enabled, collection state, analysis switch, name. |
| `sources add ID --type T --name N [--set KEY=VALUE]...` | Register a source (disabled). |
| `sources edit ID --set KEY=VALUE...` | Change fields. `id`, `type` and `enabled` cannot be edited. |
| `sources enable ID` / `sources disable ID` | Turn collection eligibility on or off. |
| `sources delete ID` | Remove a source and its runtime state. Audit events are kept. |
| `sources import FILE [--dry-run]` | Load sources from YAML, all or nothing. |
| `sources export [-o FILE] [--force]` | Write every source as YAML to stdout or a file. |
| `analysis on` / `analysis off` `[--source ID]` | Switch analysis for the system or for one source. |
| `status` | Switches, alert threshold, source counts, collection states. |

`--set KEY=VALUE` takes a source field. The value is read as JSON when it parses (`600`, `true`,
`null`, `["flood","fire"]`) and as plain text otherwise; wrap text that looks like JSON in quotes
(`sources edit ID --set 'name="2024"'`). `--set field=null` clears an optional field. `sources add`
takes `id`, `type` and `name` through their own arguments and options, not through `--set`.

Results go to stdout. A refused operation prints one `error: ...` line on stderr and exits with
status 1; a usage mistake (missing option, unknown command) exits with status 2. Error messages
never repeat values you typed, so a secret pasted into the wrong place is not echoed back.

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
    stream_url: https://example.org/live.m3u8
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
    --set stream_url=https://example.org/live.m3u8 --set chunk_seconds=30 --set language=es \
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

## Audit log

Each real change writes one event, in the same transaction as the change, with the actor and the
reason. A request that changes nothing (enabling an enabled source, importing known data) writes
no event. Events are append-only and survive the deletion of the source they describe.

## Concurrent use

Several CLI processes may share one database file: writers are serialised, and a writer that
waits longer than the busy timeout fails with `error: database error: SQLITE_BUSY` and exit status
1. Retry the command or raise `SOCIAL_TRACKER_BUSY_TIMEOUT_MS`.

## Development

```bash
uv run ruff check
uv run ruff format --check
uv run pytest -q
uv run lint-imports
```

The architecture is hexagonal. Domain modules are pure, services depend on ports only, and
`social_tracker.bootstrap` is the composition root that wires the SQLite adapters into the
services. Entrypoints such as the CLI never import persistence; `lint-imports` enforces it.

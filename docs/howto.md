# How to register a source

The web UI (`social-tracker serve`) and the CLI register the same sources. Pick the type that
matches where the content comes from. A new source is disabled: nothing is collected until you
enable it, and nothing is ever published or sent anywhere.

Each section says what to enter, gives an example and states the limits. The UI shows the same
short guide at `/howto`.

## `news_rss`: news RSS feed

**What to enter.** The address of an RSS or Atom feed (http or https) and a name. Optionally a list
of keywords to focus on.

**Example.** `https://example.org/rss.xml`

**Limits.** The feed is checked at most every 120 seconds (default 600). Addresses that embed a
login or a secret in the query string are refused, and so are `file:` addresses.

**Collecting it.** Enabling the source only makes it eligible. Content is collected while a
worker runs (`social-tracker run`) and global analysis is on: a starting worker always switches
global analysis off, so turn it on afterwards with `social-tracker analysis on` (one worker may
run per database). The list in the web UI then shows the time of the last successful fetch and
the reason of the last failure; after five failures in a row the source is marked failed and its
row offers a Reset button (`social-tracker sources reset ID` does the same). A
feed on a private network (an address that resolves to a private IP) is refused unless its host
name is listed in `SOCIAL_TRACKER_ALLOW_PRIVATE_HOSTS`. Collected items are read with
`social-tracker items list` and `social-tracker items search`, and are deleted after
`SOCIAL_TRACKER_RETENTION_DAYS` days (the README gives the default).

## `instagram`: Instagram account

**What to enter.** The account handle (letters, digits, dots, underscores) and a secret reference
that names where the login is kept: `env:NAME` (an environment variable), `keyring:NAME` or
`file:NAME`. Never type the password into the form or into a source file.

**Example.** Handle `some.handle`, reference `env:IG_MAIN`.

**Limits.** Collection from Instagram is best-effort: it can stop working without notice and is
checked at most every 900 seconds (default 1800). Use a dedicated account, not a personal one. Posts and stories
only: no Lives. The credentials stay on this machine, in the place the reference points to, and are
never tracked by git or shown in the interface.

## `live_stream`: live audio stream

**What to enter.** The address of a direct audio stream (http, https, rtmp or icecast), such as the
Icecast mount point of a radio station. A web page that plays the stream is not enough: the address
must serve the audio itself. Playlists such as HLS (`.m3u8`) are not a direct stream.

**Example.** `https://stream.example.org/radio.mp3`

**Limits.** Audio is cut into chunks of 10 to 60 seconds (default 30). The language is a tag such as
`es` (default `es`). Video-only or login-protected streams do not work.

## `recorded_audio`: recorded audio

**What to enter.** Either the address of a podcast-style feed or the path of a local directory that
holds audio files. Give one of them, not both.

**Example.** `https://example.org/podcast.xml` or `/srv/audio/incoming`

**Limits.** A feed is checked at most every 1 second (default 1800). Files are processed in batches, not in real time. Common audio formats such as MP3, M4A,
WAV, OGG and FLAC are expected; a file that cannot be decoded is skipped.

## Official earthquake catalogs

Not a source: the classifier also checks the magnitude and time an earthquake item claims against
the USGS and EMSC catalogs, and nothing else is fetched for it. The worker refreshes a local cache
of their events while global analysis is on (`social-tracker analysis on`); `social-tracker catalog
refresh` does it once on demand and `social-tracker catalog status` shows the cache and whether it
is stale. A `mismatch` adds evidence of risk and a `consistent` claim lowers an unlinked
attribution; when the catalogs cannot be reached the verdict says `catalog: unavailable` and
nothing else changes. The region (default Cali, 300 km) and the refresh interval are set with the
`SOCIAL_TRACKER_CATALOG_*` variables listed in the README.

## Local language model (Ollama)

Not a source: the items the rules cannot settle (`classify stats` shows how many wait) are judged
by a language model that runs on THIS machine. It is optional; without it the verdicts of the CPU
stages stay as they are.

**Privacy.** The text of an item is sent only to the address in `SOCIAL_TRACKER_LLM_BASE_URL`, and
that address must be `http://127.0.0.1` or `http://localhost` (with an optional port); the
application refuses to start with any other. Nothing is sent to a cloud service, no proxy is
used and no redirect is followed. A loopback port can be held by any local program, so the
application first asks the address who it is (Ollama answers `Ollama is running`) and sends
nothing to something that does not say so; keep your own port for Ollama and run nothing else
there.

**Run Ollama.** Install it from [ollama.com](https://ollama.com) and start the server (it listens
on `127.0.0.1:11434` by default). The models are large: on Windows, if `C:` is short of space,
move them to another drive by setting the environment variable `OLLAMA_MODELS` (for example to
`D:\ollama\models`) BEFORE the server starts, then restart it. Models already pulled stay where
they were unless you move the folder.

**Models.** The default judge is `st-gemma4-e4b`; if it is not installed or cannot be loaded,
`st-qwen35-4b` is tried once (every model is told not to think; a model that rejects that is asked
again without it). The first call after the server unloaded the model has to load it again and gets
a longer deadline (`SOCIAL_TRACKER_LLM_LOAD_TIMEOUT_S`, 180 seconds). Both names, and the rest of the
settings, are the `SOCIAL_TRACKER_LLM_*` variables of the README. The server frees the GPU 300
seconds after the last call (`SOCIAL_TRACKER_LLM_KEEP_ALIVE_S`).

**Check it.**

- `social-tracker llm status` says whether the server answers, which models it has and which
  configured ones are missing, how many items wait, and how much of the hourly budget (200 judgments
  per UTC hour) is used. It loads no model.
- `social-tracker llm ping` asks the model one tiny harmless question. Run it yourself when you want
  to see the model answer (it loads the model onto the GPU); nothing else ever calls it.
- `social-tracker classify escalate --dry-run` says how many items would be sent;
  `classify escalate` sends them once (the worker does it after every pass while global analysis is
  on).

**If the server is down** nothing breaks: the CPU verdicts stay, the whole judge rests and the item
it was asked about goes behind the others, for 60 seconds, then 120, doubling up to one hour; the
hourly budget is not used up by a server that does not answer. Only a failure that may be the
item's own (and only while the model is answering other items) counts towards parking it;
a parked item returns to the queue after a day, or when you run `social-tracker llm requeue-parked`.
`classify show ITEM_ID` prints when, and which model and prompt version judged an item.

## Alerts

An alert is a stored verdict that deserves a person's attention. Alerts live in the local database
and are shown by the web UI (`/alerts`) and `social-tracker alerts list`; **nothing is ever sent
anywhere**: there are no external notifications, no webhooks, no e-mail, no draft replies and no
approval flow.

**When one is raised.** A verdict that is `unintentional_rumor`, `deliberate_disinformation` or
`recycled_content` with a confidence of at least the alert threshold (0.5 unless you changed it
with `social-tracker alerts threshold set VALUE --reason "..."`). The worker does it after every
pass; `social-tracker alerts refresh` does it now, and `social-tracker alerts rebuild` evaluates
everything again with the threshold in force (it withdraws alerts that no longer qualify and
never deletes one).

**Severity.** `critical` for deliberate disinformation at 0.85 or more, or for a claim the
official catalogs contradict about an event at most 24 hours old at 0.8 or more; `high` from 0.7,
`medium` from 0.5, `low` otherwise. Copies seen within 24 hours are one alert with an occurrence
count; its severity never lowers on a merge and follows the model's final verdict when the model
settles a verdict that was waiting for it. If the severity rises after you marked an alert `seen`
or `dismissed`, it goes back to `new` so that the worse news is not hidden. A publication date
that is missing or from the future counts as an unknown age, never a fresh one.

**Statuses.** A new alert is `new`. Mark it `seen` or `dismiss` it, and `reopen` a dismissed one
(`social-tracker alerts seen|dismiss|reopen ALERT_ID --reason "..."`, or the buttons on the page).
Each change is audited with who and why. Doing the same thing twice changes nothing. An alert
whose verdict later stops qualifying (the model cleared it, or it fell under the threshold) is
withdrawn with a reason instead of deleted, and comes back by itself if the verdict returns.

**What you see.** The label, severity, confidence, the source (an alert keeps the name of a source
you deleted and says `source removed`), a short plain-text excerpt (at most 500 characters, with
links stripped of their query strings and anything that looks like a cookie or a token redacted),
the evidence (the stage that decided, what the catalogs said, the signals, the model and prompt
version when the model judged it) and the tags, among them `secretariat_attribution` for a claim
attributed to the Secretaria de Gestion del Riesgo de Emergencias y Desastres de Cali (formerly
DAGRD) with no link to it. Alerts are kept 180 days after their latest occurrence
(`SOCIAL_TRACKER_ALERT_RETENTION_DAYS`).

**Dates and the page.** Dates you type in the filters (and in `alerts list --since/--until`) are read
in your time zone, `SOCIAL_TRACKER_TIMEZONE` (America/Bogota unless you set it), unless you add an
offset such as `-05:00`; times are drawn in it too. The top of the Alerts page says whether
analysis is on, when the worker last passed and the alerts were last refreshed, and the threshold;
if analysis is off nothing is being analysed and no new alert will appear.

## Labeling texts and calibrating

To see how well the classifier does, label some texts yourself: the **gold set**. It is stored in
the local database, never in the repository, and may hold real text, so do not commit an export.

- `social-tracker gold add --label no_risk --text-file text.txt --note "official bulletin"` labels
  a text from a file; `--item ITEM_ID` takes a collected item instead (see
  `social-tracker items list --ids`). `social-tracker gold label --text-file text.txt` asks for the
  label and the note on the terminal.
- `social-tracker gold list`, `gold show ID`, `gold remove ID` and `gold export -o gold.yaml`
  (it refuses a path inside a git work tree unless you pass `--allow-repo-path`: the file may hold
  real text and must stay out of version control).
- `social-tracker classify calibrate` measures the CPU cascade against the set: accuracy,
  precision and recall per label, the confusion matrix, how many texts would go to the model and
  which alert threshold would have worked best. It only reports: it changes nothing. Add
  `--file tests/data/gold_examples.yaml` to run it on the synthetic examples of the repository,
  and `--with-llm` (never the default) to also ask the local model about the texts the cascade
  would escalate.

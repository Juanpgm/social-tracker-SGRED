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

"""Links and the official domains (S3): parsed like a browser reads them, never matched as text."""

import pytest

from social_tracker.classification.links import is_official_host, links_in, official_hosts
from tests.classification.support import run_rules

DOMAINS = ("official-domain.example",)
BASE = "Según la Secretaría de Gestión del Riesgo: sismo de magnitud 4.2 en Cali. "


def official(text: str) -> bool:
    return official_hosts(links_in(text), DOMAINS)


@pytest.mark.parametrize(
    "link",
    [
        "https://official-domain.example/sismos",
        "http://official-domain.example",
        "https://www.official-domain.example/a?b=c#d",
        "www.official-domain.example/sismos",
        "https://news.official-domain.example/x",
        "HTTPS://OFFICIAL-DOMAIN.EXAMPLE/SISMOS",  # uppercase
        "https://Official-Domain.Example./sismos",  # trailing dot of an absolute name
        "https://official-domain.example:8443/sismos",  # a port
        "(https://official-domain.example)",  # punctuation around the link
        "https://official-domain.example, gracias",
        "https://official-domain.example.",  # end of a sentence
        "https://official-domain.example?x=1",
        "https://official-domain.example#top",
    ],
)
def test_a_real_link_to_an_official_domain_is_recognised(link):
    assert official(f"Mira {link}")


@pytest.mark.parametrize(
    "link",
    [
        # login (userinfo) tricks: the real host is after the "@"
        "https://official-domain.example@evil.example/sismos",
        "https://official-domain.example" + ":443" + "@evil.example/",  # assembled: see hygiene
        "https://" + "user" + ":" + "pw" + "@official-domain.example/",  # assembled: see hygiene
        "http://official-domain.example%2F@evil.example/",
        # the official name only in a path, query string or fragment
        "https://evil.example/www.official-domain.example",
        "https://evil.example/?u=www.official-domain.example",
        "https://evil.example/?next=https://official-domain.example",
        "https://evil.example#official-domain.example",
        "https://evil.example?official-domain.example",
        "https://evil.example\\@official-domain.example",  # backslash ends the host
        # look-alike suffixes and prefixes
        "https://evil-official-domain.example/",
        "https://official-domain.example.evil.com/",
        "https://xofficial-domain.example/",
        "https://official-domain.examplee/",
        "https://official-domain-example.com/",
        # punycode and non-Latin look-alikes are other names
        "https://xn--fficial-domain-ycb.example/",
        "https://оfficial-domain.example/",  # Cyrillic o
        # a mention that is not a link
        "official-domain.example",
        "visita official-domain.example/sismos",
        "httpsofficial-domain.example",
        "xhttps://official-domain.example/",
        "ftp://official-domain.example/",
    ],
)
def test_a_spoofed_or_look_alike_link_is_not_official(link):
    assert not official(f"Mira {link}")


def test_a_link_with_a_login_is_reported_as_spoofing_and_its_host_is_not_trusted():
    found = links_in("https://official-domain.example@evil.example/x")
    assert found.spoofed is True
    assert found.hosts == frozenset()


def test_a_plain_link_is_not_spoofing():
    found = links_in("https://official-domain.example/x y https://evil.example")
    assert found.spoofed is False
    assert found.hosts == {"official-domain.example", "evil.example"}


def test_a_second_link_does_not_save_a_spoofed_one_and_the_reverse():
    assert official("https://evil.example@x.example https://official-domain.example/")
    assert not official("https://official-domain.example@x.example https://evil.example/")


def test_only_the_first_links_are_read():
    text = "https://a.example " * 60 + "https://official-domain.example"
    assert not official(text)  # past MAX_LINKS: a wall of links hides nothing and costs little


def test_a_host_matches_on_label_boundaries_only():
    assert is_official_host("official-domain.example", DOMAINS)
    assert is_official_host("a.b.official-domain.example", DOMAINS)
    assert not is_official_host("xofficial-domain.example", DOMAINS)
    assert not is_official_host("example", DOMAINS)
    assert not is_official_host("", DOMAINS)
    assert not is_official_host("official-domain.example", ())


# ---- through the rules ----


@pytest.mark.parametrize(
    "link",
    [
        "https://official-domain.example@evil.example/sismos",
        "https://evil.example/?u=www.official-domain.example",
        "https://evil-official-domain.example/sismos",
        "official-domain.example/sismos",
    ],
)
def test_a_spoofed_link_leaves_the_attribution_unlinked_and_gives_no_credit(link):
    result = run_rules(f"{BASE}Evacuen la zona {link}")
    assert "official_link" not in result.names
    assert "official_attribution_unlinked" in result.names


def test_a_real_link_removes_the_unlinked_attribution_and_counts_as_benign_evidence():
    result = run_rules(f"{BASE}Más información en https://official-domain.example/sismos")
    assert "official_link" in result.names
    assert "official_attribution_unlinked" not in result.names


def test_the_address_of_the_item_counts_like_a_link_in_its_text():
    result = run_rules(BASE, url="https://www.official-domain.example/boletin/1")
    assert "official_link" in result.names
    spoof = run_rules(BASE, url="https://evil.example/official-domain.example")
    assert "official_link" not in spoof.names


def test_the_last_link_that_is_read_is_the_fiftieth():
    from social_tracker.classification.links import MAX_LINKS

    filler = "https://a.example " * (MAX_LINKS - 1)
    assert official(filler + "https://official-domain.example")  # the 50th is read
    assert not official(filler + "https://b.example https://official-domain.example")  # the 51st


def test_a_backslash_ends_the_host_as_a_browser_reads_it():
    backslash = chr(92)
    # the address goes to the official domain, whatever follows the backslash
    assert official("https://official-domain.example" + backslash + "@evil.example/x")
    found = links_in("https://official-domain.example" + backslash + "@evil.example/x")
    assert found.spoofed is False
    # and the official name after a backslash is only a path
    assert not official("https://evil.example" + backslash + "@official-domain.example/")
    assert not official("https://evil.example" + backslash + "official-domain.example")

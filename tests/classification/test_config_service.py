"""The topic configuration service: versioned, audited, atomic."""

import json
from datetime import timedelta

import pytest

from social_tracker.classification.service import ConfigChange, TopicConfigService
from social_tracker.classification.topics import TopicConfig
from social_tracker.classification.yaml_io import default_topic_config, dump_topic_config
from social_tracker.shared.errors import StorageError, ValidationError
from tests.actors import WHO
from tests.classification.harness import World, memory_world
from tests.fakes.flaky_control import FlakyControl

CUSTOM = "version: 1\nearthquake: [sismo]\nplaces: [Cali]\n"


@pytest.fixture
def world() -> World:
    return memory_world()


def events(world: World) -> list[dict]:
    return [
        json.loads(e.detail)
        for e in world.control_repo.list_control_events()
        if e.kind == "topic_config_changed"
    ]


def test_without_a_stored_version_the_shipped_defaults_are_in_force(world):
    current = world.topics.current()
    assert (current.version, current.config) == (0, default_topic_config())


def test_setting_a_configuration_stores_version_one_and_audits_it(world):
    change = world.topics.set_text(CUSTOM, **WHO)
    assert (change.version, change.changed) == (1, True)
    current = world.topics.current()
    assert (current.version, current.config.earthquake, current.config.places) == (
        1,
        ("sismo",),
        ("cali",),
    )
    [event] = events(world)
    assert (event["actor"], event["reason"]) == (WHO["actor"], WHO["reason"])
    assert (event["from"], event["to"], event["action"]) == (0, 1, "set")
    stored = world.topic_repo.latest()
    assert (stored.actor, stored.reason, stored.created_at) == (
        WHO["actor"],
        WHO["reason"],
        world.clock.now(),
    )


def test_setting_the_same_configuration_again_changes_nothing_and_writes_no_event(world):
    world.topics.set_text(CUSTOM, **WHO)
    again = world.topics.set_text(CUSTOM, **WHO)
    assert (again.version, again.changed) == (1, False)
    assert len(world.topic_repo.versions(10)) == 1
    assert len(events(world)) == 1


def test_a_reordered_but_equal_file_is_not_a_change(world):
    world.topics.set_text("version: 1\nearthquake: [sismo, temblor]\n", **WHO)
    again = world.topics.set_text("version: 1\nearthquake: [temblor, SISMO]\n", **WHO)
    assert again.changed is False


@pytest.mark.parametrize("bad", ["version: 1", None, {"version": 1}, 5])
def test_only_a_topic_configuration_can_be_set(world, bad):
    with pytest.raises(ValidationError):
        world.topics.set_config(bad, **WHO)
    assert world.topic_repo.latest() is None


def test_each_real_change_is_a_new_version(world):
    world.clock.advance(timedelta(minutes=1))
    assert world.topics.set_text(CUSTOM, **WHO).version == 1
    assert world.topics.set_text("version: 1\nearthquake: [temblor]\n", **WHO).version == 2
    assert [v.version for v in world.topic_repo.versions(10)] == [2, 1]
    assert [(e["from"], e["to"]) for e in events(world)] == [(0, 1), (1, 2)]


def test_resetting_stores_the_defaults_as_a_new_version(world):
    world.topics.set_text(CUSTOM, **WHO)
    change = world.topics.reset(**WHO)
    assert (change.version, change.changed) == (2, True)
    assert world.topics.current().config == default_topic_config()
    assert events(world)[-1]["reason"] == WHO["reason"]


def test_resetting_when_the_defaults_are_already_in_force_is_a_no_op(world):
    assert world.topics.reset(**WHO) == ConfigChange(0, False)
    assert world.topic_repo.latest() is None
    assert events(world) == []


def test_resetting_after_a_reset_changes_nothing_more(world):
    world.topics.set_text(CUSTOM, **WHO)
    world.topics.reset(**WHO)
    assert world.topics.reset(**WHO).changed is False
    assert len(world.topic_repo.versions(10)) == 2


def test_the_shipped_file_set_as_a_config_while_defaults_are_in_force_is_a_no_op(world):
    assert world.topics.set_config(default_topic_config(), **WHO).changed is False


@pytest.mark.parametrize(
    "text",
    [
        "",
        "version: 2\nearthquake: [sismo]",
        "version: 1\nunknown: 1",
        "[",
        "version: 1\nplaces: [Cali]",
    ],
)
def test_an_invalid_file_changes_nothing(world, text):
    world.topics.set_text(CUSTOM, **WHO)
    with pytest.raises(ValidationError):
        world.topics.set_text(text, **WHO)
    assert world.topics.current().version == 1
    assert len(events(world)) == 1


@pytest.mark.parametrize(
    "who",
    [
        {"actor": "", "reason": "r"},
        {"actor": "  ", "reason": "r"},
        {"actor": "a", "reason": ""},
        {"actor": "a", "reason": "x" * 501},
        {"actor": "a" * 101, "reason": "r"},
        {"actor": "a\x00b", "reason": "r"},
        {"actor": "a", "reason": "bi‭di"},
    ],
)
def test_actor_and_reason_are_required_and_validated(world, who):
    with pytest.raises(ValidationError):
        world.topics.set_text(CUSTOM, **who)
    with pytest.raises(ValidationError):
        world.topics.reset(**who)
    assert world.topic_repo.latest() is None


def test_a_failing_audit_write_rolls_the_new_version_back(world):
    flaky = FlakyControl(world.control_repo, "append_control_event")
    topics = TopicConfigService(
        world.topic_repo, flaky, world.uow, world.clock, default_topic_config()
    )
    with pytest.raises(StorageError):
        topics.set_text(CUSTOM, **WHO)
    assert world.topic_repo.latest() is None
    assert topics.current().version == 0


def test_the_text_of_the_current_configuration_loads_back_to_the_same_one(world):
    world.topics.set_text(CUSTOM, **WHO)
    shown = world.topics.show_text()
    assert shown == dump_topic_config(world.topics.current().config)
    assert world.topics.set_text(shown, **WHO).changed is False


def test_a_stored_default_is_still_the_defaults_after_the_shipped_file_changes():
    old_default = TopicConfig.from_mapping({"version": 1, "earthquake": ["sismo"]})
    world = memory_world(default=old_default)
    world.topics.set_text(CUSTOM, **WHO)
    assert world.topics.reset(**WHO).changed is True
    assert world.topics.current().config == old_default

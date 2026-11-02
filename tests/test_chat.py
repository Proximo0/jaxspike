import numpy as np

from game.config import MAX_PLAYERS
from net.chat import CHAT_MAX_LEN, CHAT_RATE_PER_TICK, CHAT_VIEW_RADIUS, ChatRouter, validate_chat


def world():
    pos = np.zeros((MAX_PLAYERS, 2), np.int32)
    active = np.zeros(MAX_PLAYERS, bool)
    active[:6] = True
    pid = np.full(MAX_PLAYERS, -1, np.int32)
    names = {0: "Alice", 1: "Bob", 2: "Cara", 3: "Dan"}  # slots 4,5 are bots
    pos[1] = (CHAT_VIEW_RADIUS[0], CHAT_VIEW_RADIUS[1])
    pos[2] = (CHAT_VIEW_RADIUS[0] + 1, 0)
    pos[3] = (0, CHAT_VIEW_RADIUS[1] + 1)
    return pos, active, pid, names


def m(slot, kind="say", text="hi", to=None):
    return {"slot": slot, "kind": kind, "text": text, "to": to}


def test_say_radius_and_bots():
    pos, active, pid, names = world()
    out = ChatRouter().route(1, [m(0)], pos, active, pid, names)
    assert set(out) == {0, 1}
    assert out[1][0] == {"t": "chat", "kind": "say", "from": "Alice", "text": "hi", "tick": 1}


def test_party():
    pos, active, pid, names = world()
    pid[[0, 2, 3]] = 7
    out = ChatRouter().route(1, [m(0, "party")], pos, active, pid, names)
    assert set(out) == {0, 2, 3}
    assert ChatRouter().route(1, [m(1, "party")], pos, active, pid, names) == {}


def test_whisper_and_failure():
    pos, active, pid, names = world()
    r = ChatRouter()
    out = r.route(1, [m(0, "whisper", "psst", "  BOB ")], pos, active, pid, names)
    assert set(out) == {1} and out[1][0]["kind"] == "whisper"
    out = r.route(2, [m(0, "whisper", "x", "nobody"), m(0, "whisper", "x", "")], pos, active, pid, names)
    assert out == {0: [{"t": "chat", "kind": "system", "text": "no such player"}] * 2}
    active[1] = False
    out = r.route(3, [m(0, "whisper", "x", "bob")], pos, active, pid, names)
    assert out[0][0]["kind"] == "system"


def test_rate_limit_resets_per_tick():
    pos, active, pid, names = world()
    r = ChatRouter()
    msgs = [m(0, text=str(i)) for i in range(CHAT_RATE_PER_TICK + 2)]
    assert len(r.route(5, msgs, pos, active, pid, names)[0]) == CHAT_RATE_PER_TICK
    assert len(r.route(6, msgs, pos, active, pid, names)[0]) == CHAT_RATE_PER_TICK


def test_length_cap_empty_and_malformed():
    pos, active, pid, names = world()
    r = ChatRouter()
    out = r.route(1, [m(0, text="x" * 500), m(0, text="   "), m(0, kind="shout"), m(0, text=5),
                      {"kind": "say", "text": "a"}, m(99), m(4)], pos, active, pid, names)
    assert len(out[0]) == 1 and len(out[0][0]["text"]) == CHAT_MAX_LEN


def test_validate_chat():
    assert validate_chat({"t": "chat", "kind": "say", "text": "a", "to": None, "x": 1}) == \
        {"t": "chat", "kind": "say", "text": "a", "to": None}
    assert validate_chat({"t": "chat", "kind": "bad", "text": "a"}) is None
    assert validate_chat({"t": "chat", "kind": "say", "text": 3}) is None
    assert validate_chat({"t": "chat", "kind": "whisper", "text": "a"}) is None

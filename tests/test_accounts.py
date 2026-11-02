import json

import pytest

from net.accounts import AccountStore


class FakeClock:
    def __init__(self):
        self.t = [1000.0]

    def __call__(self):
        return self.t[0]

    def advance(self, s):
        self.t[0] += s


def _record(**over):
    d = {"v": 1, "salt": "ab", "hash": "cd", "n": 256, "r": 8, "p": 1,
         "created": 0.0, "last_login": None, "characters": []}
    d.update(over)
    return json.dumps(d).encode()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def store(tmp_path, clock, monkeypatch):
    monkeypatch.setattr(AccountStore, "SCRYPT_N", 2 ** 8)
    return AccountStore(tmp_path / "accounts", clock=clock)


def test_create_then_verify(store):
    assert store.create("Alice", "hunter2") is None
    assert store.exists("alice") and store.exists("ALICE")
    assert store.verify("alice", "hunter2") and store.verify("Alice", "hunter2")


def test_wrong_password_and_unknown_name(store):
    store.create("alice", "hunter2")
    assert not store.verify("alice", "hunter3")
    assert not store.verify("nobody", "hunter2")
    assert not store.exists("nobody")


def test_second_create_is_name_taken_and_keeps_password(store):
    assert store.create("alice", "hunter2") is None
    assert store.create("ALICE", "other") == "name taken"
    assert store.verify("alice", "hunter2") and not store.verify("alice", "other")


def test_bad_names_and_passwords(store):
    for bad in ("ab", "a" * 17, "bad name", "", "ünï"):
        assert store.create(bad, "hunter2") == "bad name"
        assert not store.verify(bad, "hunter2")
    assert store.create("alice", "") == "bad password"
    assert store.create("alice", "x" * 129) == "bad password"
    assert not store.exists("alice")
    assert store.create("alice", "x" * 128) is None
    assert store.verify("alice", "x" * 128) and not store.verify("alice", "x" * 129)


def test_file_layout_and_params(store, tmp_path):
    store.create("alice", "hunter2")
    d = json.loads((tmp_path / "accounts" / "alice.json").read_text())
    assert set(d) == {"v", "salt", "hash", "n", "r", "p", "created", "last_login", "characters"}
    assert d["v"] == 1 and d["n"] == 2 ** 8 and d["r"] == 8 and d["p"] == 1
    assert len(bytes.fromhex(d["salt"])) == 32 and len(bytes.fromhex(d["hash"])) == 32
    assert d["last_login"] is None and d["characters"] == []


def test_salts_differ(store, tmp_path):
    store.create("alice", "hunter2")
    store.create("bobby", "hunter2")
    a = json.loads((tmp_path / "accounts" / "alice.json").read_text())
    b = json.loads((tmp_path / "accounts" / "bobby.json").read_text())
    assert a["salt"] != b["salt"] and a["hash"] != b["hash"]


def test_file_params_are_read_back(store, monkeypatch):
    store.create("alice", "hunter2")
    monkeypatch.setattr(AccountStore, "SCRYPT_N", 2 ** 9)   # class default changes; old file still verifies
    assert store.verify("alice", "hunter2")


def test_lockout_after_five_fails_then_expires(store, clock):
    store.create("alice", "hunter2")
    for _ in range(4):
        assert not store.verify("alice", "wrong")
        assert not store.locked_out("alice")
    assert not store.verify("alice", "wrong")
    assert store.locked_out("alice")
    assert not store.verify("alice", "hunter2")          # correct password refused while locked
    clock.advance(30)
    assert store.locked_out("alice") and not store.verify("alice", "hunter2")
    clock.advance(31)
    assert not store.locked_out("alice")
    assert store.verify("alice", "hunter2")
    for _ in range(4):                                    # counter was reset: four more fails do not lock
        assert not store.verify("alice", "wrong")
    assert not store.locked_out("alice")


def test_success_resets_counter(store):
    store.create("alice", "hunter2")
    for _ in range(4):
        store.verify("alice", "wrong")
    assert store.verify("alice", "hunter2")
    for _ in range(4):
        store.verify("alice", "wrong")
    assert not store.locked_out("alice")


def test_unknown_name_failures_lock_too(store):
    for _ in range(5):
        assert not store.verify("ghost", "hunter2")
    assert store.locked_out("ghost")


@pytest.mark.parametrize("content", [b"{", b"", b"{}", b'{"v": 1}', b"[]",
                                     _record(salt="zz", hash="zz"),
                                     _record()])
def test_corrupt_file_refused_and_never_overwritten(store, tmp_path, content):
    path = tmp_path / "accounts" / "mallory.json"
    path.write_bytes(content)
    assert store.exists("mallory")
    assert not store.verify("mallory", "hunter2")
    assert store.create("mallory", "hunter2") == "name taken"
    assert store.characters("mallory") == []
    assert store.add_character("mallory", "mallory") == "no such account"
    store.touch_login("mallory")
    assert path.read_bytes() == content


def test_no_tmp_left_behind(store, tmp_path):
    store.create("alice", "hunter2")
    store.touch_login("alice")
    store.add_character("alice", "alice")
    assert not list((tmp_path / "accounts").glob("*.tmp"))


def test_touch_login_sets_last_login(store, tmp_path):
    store.create("alice", "hunter2")
    store.touch_login("alice")
    d = json.loads((tmp_path / "accounts" / "alice.json").read_text())
    assert isinstance(d["last_login"], float)
    store.touch_login("nobody")   # unknown account is a no-op
    assert not store.exists("nobody")


def test_add_character_limit(store):
    store.create("alice", "hunter2")
    assert store.add_character("alice", "one") is None
    assert store.add_character("alice", "two") is None
    assert store.add_character("alice", "three") is None
    assert store.add_character("alice", "four") == "too many characters"
    assert store.add_character("alice", "ONE") is None   # already owned: no new slot used
    assert store.add_character("alice", "bad name") == "bad name"
    assert store.add_character("nobody", "one") == "no such account"
    assert store.characters("alice") == ["one", "two", "three"]
    assert store.characters("nobody") == []


def test_characters_round_trip_fresh_store(store, tmp_path, clock):
    store.create("alice", "hunter2")
    store.add_character("alice", "one")
    store.add_character("alice", "two")
    fresh = AccountStore(tmp_path / "accounts", clock=clock)
    assert fresh.characters("alice") == ["one", "two"]
    assert fresh.verify("alice", "hunter2")

"""Account store for the networked server: scrypt-hashed passwords in one JSON file per account (stdlib only)."""
import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import Path

from net.characters import valid_name

_KEYS = ("v", "salt", "hash", "n", "r", "p", "created", "last_login", "characters")
_DUMMY_SALT = bytes(32)


def _key(name) -> str | None:
    return valid_name(name) if isinstance(name, str) else None


class AccountStore:
    MAX_CHARACTERS = 3
    LOCKOUT_FAILS = 5          # failed verifies before lockout
    LOCKOUT_SECONDS = 60.0
    SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1
    SALT_BYTES = 32
    DKLEN = 32
    MAX_PASSWORD = 128
    MAX_N = 2 ** 20            # largest scrypt n accepted from a file (128 MiB at r=8)

    def __init__(self, dir, clock=time.monotonic):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self._file_lock = threading.Lock()   # file read-modify-write and the shared temp file
        self._state_lock = threading.Lock()  # in-memory rate-limit counters
        self._fails: dict[str, int] = {}
        self._locked_until: dict[str, float] = {}

    # ---- helpers -------------------------------------------------------------------------------
    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def _hash(self, password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
        maxmem = 128 * n * r * p + 128 * r * (p + 2) + (1 << 20)   # scaled so other n/r/p in a file still verify
        return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=self.DKLEN, maxmem=maxmem)

    def _read(self, key: str) -> dict | None:
        """Parsed and validated record, or None if the file is missing, corrupt or incomplete."""
        try:
            data = json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or not all(k in data for k in _KEYS):
            return None
        try:
            salt = bytes.fromhex(data["salt"])
            digest = bytes.fromhex(data["hash"])
        except (TypeError, ValueError):
            return None
        if len(salt) != self.SALT_BYTES or len(digest) != self.DKLEN:
            return None
        n, r, p = data["n"], data["r"], data["p"]
        if not all(type(x) is int and x > 0 for x in (n, r, p)) or n & (n - 1) or n < 2 or n > self.MAX_N:
            return None   # MAX_N bounds the memory a tampered file can make verify allocate
        chars = data["characters"]
        if not isinstance(chars, list) or not all(isinstance(c, str) for c in chars):
            return None
        data["_salt"], data["_digest"] = salt, digest
        return data

    def _write(self, key: str, record: dict) -> None:
        """Atomic write; the caller holds _file_lock."""
        final = self._path(key)
        tmp = final.with_name(f"{key}.json.tmp")
        tmp.write_text(json.dumps({k: record[k] for k in _KEYS}), encoding="utf-8")
        os.replace(tmp, final)

    # ---- public API ----------------------------------------------------------------------------
    def exists(self, name) -> bool:
        key = _key(name)
        return key is not None and self._path(key).exists()

    def create(self, name, password) -> str | None:
        key = _key(name)
        if key is None:
            return "bad name"
        if not isinstance(password, str) or not 1 <= len(password) <= self.MAX_PASSWORD:
            return "bad password"
        n, r, p = self.SCRYPT_N, self.SCRYPT_R, self.SCRYPT_P
        salt = os.urandom(self.SALT_BYTES)
        digest = self._hash(password, salt, n, r, p)   # slow part outside the lock
        record = {"v": 1, "salt": salt.hex(), "hash": digest.hex(), "n": n, "r": r, "p": p,
                  "created": time.time(), "last_login": None, "characters": []}
        with self._file_lock:
            if self._path(key).exists():
                return "name taken"
            self._write(key, record)
        return None

    def locked_out(self, name) -> bool:
        key = _key(name)
        if key is None:
            return False
        with self._state_lock:
            until = self._locked_until.get(key)
            if until is None:
                return False
            if self.clock() < until:
                return True
            del self._locked_until[key]   # lock expired: fresh counter
            self._fails.pop(key, None)
            return False

    def _fail(self, key: str) -> None:
        with self._state_lock:
            count = self._fails.get(key, 0) + 1
            self._fails[key] = count
            if count >= self.LOCKOUT_FAILS:
                self._locked_until[key] = self.clock() + self.LOCKOUT_SECONDS

    def verify(self, name, password) -> bool:
        key = _key(name)
        if key is None or not isinstance(password, str):
            return False
        if self.locked_out(key):
            return False
        rec = self._read(key)
        pw = password[: self.MAX_PASSWORD + 1]   # bound the work an oversized password can cause
        if rec is None:
            # unknown or corrupt: burn the same scrypt time, then fail
            self._hash(pw, _DUMMY_SALT, self.SCRYPT_N, self.SCRYPT_R, self.SCRYPT_P)
            self._fail(key)
            return False
        got = self._hash(pw, rec["_salt"], rec["n"], rec["r"], rec["p"])
        ok = hmac.compare_digest(got, rec["_digest"]) and len(password) <= self.MAX_PASSWORD
        if ok:
            with self._state_lock:
                self._fails.pop(key, None)
                self._locked_until.pop(key, None)
        else:
            self._fail(key)
        return ok

    def add_character(self, account, char_name) -> str | None:
        key, char = _key(account), _key(char_name)
        with self._file_lock:
            rec = self._read(key) if key else None
            if rec is None:
                return "no such account"
            if char is None:
                return "bad name"
            if char in rec["characters"]:
                return None   # idempotent: a returning character does not use a new slot
            if len(rec["characters"]) >= self.MAX_CHARACTERS:
                return "too many characters"
            rec["characters"].append(char)
            self._write(key, rec)
        return None

    def characters(self, account) -> list[str]:
        key = _key(account)
        rec = self._read(key) if key else None
        return list(rec["characters"]) if rec else []

    def touch_login(self, name) -> None:
        key = _key(name)
        if key is None:
            return
        with self._file_lock:
            rec = self._read(key)
            if rec is None:
                return
            rec["last_login"] = time.time()
            self._write(key, rec)

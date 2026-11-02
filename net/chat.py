"""Chat routing: pure Python/numpy, no sockets. Say by view radius, party by id, whisper by name."""
import numpy as np

from game.config import MAX_PLAYERS, VIEW_H, VIEW_W

CHAT_MAX_LEN = 200
CHAT_RATE_PER_TICK = 3
CHAT_VIEW_RADIUS = (VIEW_W // 2, VIEW_H // 2)
CHAT_KINDS = ("say", "party", "whisper")


def validate_chat(msg: dict) -> dict | None:
    """Reduce a decoded client chat message to {"t","kind","text","to"}, or None if malformed."""
    if not isinstance(msg, dict) or msg.get("t") != "chat":
        return None
    kind, text, to = msg.get("kind"), msg.get("text"), msg.get("to")
    if kind not in CHAT_KINDS or not isinstance(text, str):
        return None
    if to is not None and not isinstance(to, str):
        return None
    if kind == "whisper" and not to:
        return None
    return {"t": "chat", "kind": kind, "text": text, "to": to}


class ChatRouter:
    """Stateless apart from a per-tick rate counter."""

    def __init__(self):
        self._tick = None
        self._count: dict[int, int] = {}

    def route(self, tick: int, msgs: list[dict], pos: np.ndarray, active: np.ndarray,
              party_id: np.ndarray, names: dict[int, str]) -> dict[int, list[dict]]:
        if tick != self._tick:
            self._tick = tick
            self._count = {}
        out: dict[int, list[dict]] = {}
        pos = np.asarray(pos)
        active = np.asarray(active, dtype=bool)
        party_id = np.asarray(party_id)
        # slots that can receive: active and named (bots have no name)
        named = np.zeros(MAX_PLAYERS, dtype=bool)
        for s in names:
            if isinstance(s, int) and 0 <= s < MAX_PLAYERS:
                named[s] = True
        can_recv = active & named
        by_name = {n.lower(): s for s, n in names.items() if isinstance(n, str)}
        rx, ry = CHAT_VIEW_RADIUS
        for m in msgs:
            if not isinstance(m, dict):
                continue
            slot, kind, text = m.get("slot"), m.get("kind"), m.get("text")
            if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < MAX_PLAYERS:
                continue
            if kind not in CHAT_KINDS or not isinstance(text, str):
                continue
            if slot not in names or not active[slot]:
                continue
            text = text.strip()[:CHAT_MAX_LEN].strip()
            if not text:
                continue
            n = self._count.get(slot, 0)
            if n >= CHAT_RATE_PER_TICK:
                continue
            self._count[slot] = n + 1
            ev = {"t": "chat", "kind": kind, "from": names[slot], "text": text, "tick": int(tick)}
            if kind == "say":
                d = np.abs(pos - pos[slot])
                mask = can_recv & (d[:, 0] <= rx) & (d[:, 1] <= ry)
                targets = np.flatnonzero(mask)
            elif kind == "party":
                pid = party_id[slot]
                targets = np.flatnonzero(can_recv & (party_id == pid)) if pid >= 0 else []
            else:
                to = m.get("to")
                tgt = by_name.get(to.strip().lower()) if isinstance(to, str) else None
                if tgt is not None and can_recv[tgt]:
                    targets = [tgt]
                else:
                    out.setdefault(slot, []).append({"t": "chat", "kind": "system", "text": "no such player"})
                    continue
            for t in targets:
                out.setdefault(int(t), []).append(dict(ev))
        return out

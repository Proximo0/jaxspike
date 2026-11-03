"""Generates the placeholder feel sounds (walk, attack, hit, level_up, pickup, warp) as 16-bit mono 22050 Hz WAVs.

Procedural, made for jaxspike, dedicated to the public domain (CC0 1.0). Standard library only; deterministic
(fixed seed). Run: python client/assets/audio/gen_sounds.py  (writes next to this file).
"""
import math
import os
import random
import struct
import wave

RATE = 22050
OUT = os.path.dirname(os.path.abspath(__file__))


def env(i, n, attack=0.01, release=0.5):
    """Linear attack, exponential-ish release, as a 0..1 gain for sample i of n."""
    t = i / RATE
    a = min(1.0, t / attack) if attack > 0 else 1.0
    r = max(0.0, 1.0 - i / n) ** (1.0 / max(release, 1e-3))
    return a * r


def write(name, samples):
    peak = max(1e-6, max(abs(s) for s in samples))
    gain = 0.8 / peak
    with wave.open(os.path.join(OUT, name + ".wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s * gain)) * 32767)) for s in samples))


def walk(rng):
    # Soft footstep: a short low-passed noise thump.
    n = int(RATE * 0.09)
    out, lp = [], 0.0
    for i in range(n):
        lp += 0.15 * (rng.uniform(-1, 1) - lp)
        out.append(lp * env(i, n, 0.004, 0.3) + 0.3 * math.sin(2 * math.pi * 110 * i / RATE) * env(i, n, 0.002, 0.2))
    return out


def attack(rng):
    # Whoosh: noise with a falling band, like a sword swing.
    n = int(RATE * 0.18)
    out, lp = [], 0.0
    for i in range(n):
        k = 0.05 + 0.4 * (1 - i / n)
        lp += k * (rng.uniform(-1, 1) - lp)
        out.append(lp * env(i, n, 0.03, 0.6))
    return out


def hit(rng):
    # Thwack: pitch-dropping square-ish tone plus a noise click.
    n = int(RATE * 0.16)
    out, ph = [], 0.0
    for i in range(n):
        f = 320 * (1 - 0.6 * i / n)
        ph += 2 * math.pi * f / RATE
        tone = math.copysign(1.0, math.sin(ph)) * 0.5
        click = rng.uniform(-1, 1) * (1.0 if i < RATE * 0.01 else 0.0)
        out.append((tone + click) * env(i, n, 0.001, 0.35))
    return out


def level_up(rng):
    # Bright rising arpeggio (C major: C5 E5 G5 C6) with a sparkle tail.
    notes = [523.25, 659.25, 783.99, 1046.5]
    step = int(RATE * 0.09)
    n = step * len(notes) + int(RATE * 0.35)
    out = [0.0] * n
    for k, f in enumerate(notes):
        start = k * step
        m = n - start
        for i in range(m):
            out[start + i] += 0.5 * math.sin(2 * math.pi * f * i / RATE) * env(i, m, 0.005, 0.4)
            out[start + i] += 0.15 * math.sin(2 * math.pi * f * 2 * i / RATE) * env(i, m, 0.005, 0.25)
    return out


def pickup(rng):
    # Two quick blips up a fifth.
    out = []
    for f in (880.0, 1318.5):
        n = int(RATE * 0.06)
        out += [0.6 * math.sin(2 * math.pi * f * i / RATE) * env(i, n, 0.002, 0.5) for i in range(n)]
    return out


def warp(rng):
    # Shimmering upward sweep.
    n = int(RATE * 0.45)
    out, ph = [], 0.0
    for i in range(n):
        f = 300 + 900 * (i / n) ** 1.5
        ph += 2 * math.pi * f / RATE
        trem = 0.6 + 0.4 * math.sin(2 * math.pi * 18 * i / RATE)
        out.append(math.sin(ph) * trem * env(i, n, 0.05, 0.5))
    return out


if __name__ == "__main__":
    rng = random.Random(1234)
    for fn in (walk, attack, hit, level_up, pickup, warp):
        write(fn.__name__, fn(rng))
    print("gen_sounds wrote=6")

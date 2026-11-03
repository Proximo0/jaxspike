"""Elements: ids, names, the attack-vs-defender percent table and the per-monster-kind element.

Indexing: ``ELEM_MULT[attack_elem, defend_elem]`` is an int32 percent (100 = normal, 0 = immune). Rows are the
attacking element, columns the defending element, both in ELEM_* id order. Use ``elem_mult`` for broadcast gathers.
``MONSTER_ELEMENT[kind]`` is the defending element of each monster kind (same length as ``game.config.MONSTER_HP``).
Only ``jnp.asarray`` is used at import; everything else is plain Python tuples.
"""
import jax.numpy as jnp

from game import config

ELEM_NEUTRAL = 0
ELEM_WATER = 1
ELEM_EARTH = 2
ELEM_FIRE = 3
ELEM_WIND = 4
ELEM_POISON = 5
ELEM_HOLY = 6
ELEM_SHADOW = 7
ELEM_GHOST = 8
ELEM_UNDEAD = 9
N_ELEMENTS = 10

ELEM_NAME = ("Neutral", "Water", "Earth", "Fire", "Wind", "Poison", "Holy", "Shadow", "Ghost", "Undead")

# Source: rAthena db/pre-re/attr_fix.yml, defender element level 1 (audited by ro-fidelity 2026-11-02; iRO wiki
# "Element" agrees). Negative entries (Poison/Shadow vs Undead heal 25%) are clamped to 0 (no healing hits).
# Monsters with element level 2-4 (Lunatic, Thief Bug Neutral 3; Hode Earth 2; Mummy Undead 2; GTB Fire 2) use
# this level-1 row: a deliberate simplification (docs/ro_fidelity.md).
# Rows = attack element, columns = defend element.
#            Neu  Wat  Ear  Fir  Win  Poi  Hol  Sha  Gho  Und
_TABLE = (
    (100, 100, 100, 100, 100, 100, 100, 100,  25, 100),  # Neutral attack
    (100,  25, 100, 150,  50, 100,  75, 100, 100, 100),  # Water attack
    (100, 100, 100,  50, 150, 100,  75, 100, 100, 100),  # Earth attack
    (100,  50, 150,  25, 100, 100,  75, 100, 100, 125),  # Fire attack
    (100, 175,  50, 100,  25, 100,  75, 100, 100, 100),  # Wind attack
    (100, 100, 125, 125, 125,   0,  75,  50, 100,   0),  # Poison attack (RO -25: heals; clamped to 0)
    (100, 100, 100, 100, 100, 100,   0, 125, 100, 150),  # Holy attack
    (100, 100, 100, 100, 100,  50, 125,   0, 100,   0),  # Shadow attack (RO -25 vs Undead: clamped to 0)
    ( 25, 100, 100, 100, 100, 100,  75,  75, 125, 100),  # Ghost attack
    (100, 100, 100, 100, 100,  50, 100,   0, 100,   0),  # Undead attack
)
ELEM_MULT = jnp.asarray(_TABLE, dtype=jnp.int32)

# Defending element per monster kind, from the RO monster table (same ids as config.MONSTER_HP).
from game.data.monsters import MONSTERS as _MONSTERS  # noqa: E402
_MONSTER_ELEMENT = tuple(m.element for m in _MONSTERS)
assert len(_MONSTER_ELEMENT) == len(config.MONSTER_HP)
MONSTER_ELEMENT = jnp.asarray(_MONSTER_ELEMENT, dtype=jnp.int32)


def elem_mult(attack_elem, defend_elem):
    """Percent multiplier ``ELEM_MULT[attack_elem, defend_elem]``; integer arrays broadcast."""
    return ELEM_MULT[attack_elem, defend_elem]

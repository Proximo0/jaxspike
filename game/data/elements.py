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

# Source: classic Ragnarok Online pre-renewal element chart (iRO wiki "Element" page), defender level 1 column
# (same-element level 1 is 25%; the 0% immunities appear only at higher defender levels).
# Rows = attack element, columns = defend element.
#            Neu  Wat  Ear  Fir  Win  Poi  Hol  Sha  Gho  Und
_TABLE = (
    (100, 100, 100, 100, 100, 100, 100, 100,  25, 100),  # Neutral attack
    (100,  25, 100, 150,  50, 100,  75, 100, 100, 100),  # Water attack
    (100, 100, 100,  50, 150, 125,  75, 100, 100, 100),  # Earth attack
    (100,  50, 150,  25, 100, 100,  75, 100, 100, 125),  # Fire attack
    (100, 175,  50, 100,  25, 100,  75, 100, 100, 100),  # Wind attack
    (100, 100, 125, 125, 125,   0,  75,  50, 100,  50),  # Poison attack
    (100, 100, 100, 100, 100, 100,   0, 125, 100, 125),  # Holy attack
    (100, 100, 100, 100, 100,  50, 125,   0, 100,   0),  # Shadow attack
    ( 25, 100, 100, 100, 100, 100,  75, 100, 125, 100),  # Ghost attack
    (100, 100, 100, 100, 100,  50, 100, 100, 100, 100),  # Undead attack
)
ELEM_MULT = jnp.asarray(_TABLE, dtype=jnp.int32)

# rat Neutral, orc Earth (indexed by monster kind, length matches MONSTER_HP).
_MONSTER_ELEMENT = (ELEM_NEUTRAL, ELEM_EARTH)
assert len(_MONSTER_ELEMENT) == len(config.MONSTER_HP)
MONSTER_ELEMENT = jnp.asarray(_MONSTER_ELEMENT, dtype=jnp.int32)


def elem_mult(attack_elem, defend_elem):
    """Percent multiplier ``ELEM_MULT[attack_elem, defend_elem]``; integer arrays broadcast."""
    return ELEM_MULT[attack_elem, defend_elem]

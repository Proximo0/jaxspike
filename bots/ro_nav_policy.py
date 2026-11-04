"""Bot navigation for the RO world: hunting, retreat and rest by max-HP fraction, zone crossing, dungeon diving,
party following. Pure, read-only.

Modes are derived from the world every tick (no bot state):
- hunting: wall- and crowd-aware chase and the attack flag (see nav_moves); this replaces the base policy's chase,
  whose absolute HUNT_RETREAT_HP no longer fits derived max_hp (a level-1 Novice has 41);
- hurt (hp below RETREAT_PCT of derived max_hp): sit down where it is safe (field rest, ro4-balance), else walk
  home along the warp-aware home-gate field (inside a dungeon that field leads through the exit warp, from another
  zone through its portal);
- resting (on a protection zone below REST_PCT, or sitting in the field below REST_PCT): stand still and sit;
- diving (base level >= the dungeon's DUNGEON_DIVE_LEVEL, healthy, not a gatherer): walk down the nearest eligible
  dungeon's entrance field to its entrance warp; inside a dungeon, hunt (chase within BOT_CHASE_RANGE, else keep a
  wander heading) instead of the base policy's "too far from the gate, walk back";
- following (party member, leader alive and hunting in the same zone, more than FOLLOW_MIN cells away): step
  toward the leader, so a party of 2-4 hunts together in one zone (ro_party.share_exp pays same-zone members).
Otherwise bots roam like bots.policies (leave town, walk out to the hunt ring, keep a heading, turn back), with
health measured as a fraction of derived max_hp. The ring is zone-aware: it is a band of the warp-aware
world["dist_town"] field (walk distance from the home gate across the whole map, portals included) chosen by base
level (RING_BY_LEVEL), so bots above BAND1_LEVEL hunt in the second field band and cross the zone portals on the way;
a portal warp tile counts as the farthest neighbour when walking away, so a road that ends at a rock line is taken
through. Moves are codes into DELTAS (the engine's move table, 8 directions when game.ro_move exists).

The entrance fields come from `world["dist_dungeon"]` (int16 [ND, MAP, MAP], `dungeon_fields(terrain)`: walk
distance across warps, like map_gen.distance_fields_ro, so every town reaches both dungeons through the portals);
without that key bots never dive (bots.runner adds it next to dist_town).
"""
import jax
import jax.numpy as jnp
import numpy as np

from bots.policies import HUNT_RADIUS
from bots.ro_party_policy import leader_of
from game.config import MAP, MAX_PLAYERS, MONSTER_ATTACK_COOLDOWN
from game.data import monsters as mdata
from game.data.monsters import LEVEL as MON_LEVEL
from game.data.terrain import T_WARP
from game.map_gen import DIST_UNREACHABLE, _shift_min, blocking, warp_table
from game.spatial import build_grid, nearest_in_block
from game.world_layout import (DUNGEONS, TOWNS, dungeon_index, dungeon_pads, waybell_tiles, portal_warp_tiles,
                               zone_of)

try:  # 8 directions once ro-move's module exists (movement reads these codes)
    from game.ro_move import DIR_DELTAS9 as _DELTAS_T
except ImportError:  # pragma: no cover - legacy 4-direction engine
    from game.config import DIR_DELTAS as _DELTAS_T

DELTAS = jnp.asarray(_DELTAS_T, jnp.int32)  # [K + 1, 2], code 0 = stay
N_DIRS = len(_DELTAS_T) - 1
RETREAT_PCT = 55        # below this percent of max_hp: stop hunting (sit where it is safe, else walk home)
REST_PCT = 90           # a sitting bot (in town or in a safe field spot) stands up at this percent
# Prey choice (ro4-balance, "fight what you can hit"): the nearest monster is chased and hit only when the bot's
# hit chance on it (80 + HIT - (level + AGI), combat.hit_percent_player) is at least MIN_HIT_PCT and the HP the bot
# expects to lose killing it, times COST_SAFETY_PCT / 100, leaves it above FLOOR_PCT of max_hp. A passive monster
# that fails the test is left alone even when adjacent (it never starts a fight); a provoked one is fought back.
MIN_HIT_PCT = 70
COST_SAFETY_PCT = 125
FLOOR_PCT = 10
SAFE_DIST = 3           # field rest only when no aggressive or provoked monster is within this many cells
# Monster swing period the bot's estimate assumes: the engine's (combat.py swings every MONSTER_ATTACK_COOLDOWN
# ticks); switch to max(1, (mdata.ADELAY_MS + TICK_MS // 2) // TICK_MS) per kind together with combat.
MON_SWING_TICKS = jnp.full(mdata.LEVEL.shape, MONSTER_ATTACK_COOLDOWN, jnp.int32)
DIVE_PCT = 70           # leave for a dungeon only at or above this percent (= REST_PCT: rested bots dive)
DUNGEON_MIN_LEVEL = 10  # base level from which bots walk into the easiest dungeon (Culvert monsters are level 6-16)
BOT_CHASE_RANGE = 6     # same bound as bots.policies (3x3 spatial cells)
HEADING_TICKS = 50
BORED_TICKS = 40        # chase-boredom windows (ticks)
BORED_PCT = 25          # percent of windows a bot ignores monsters it is not adjacent to
LEVEL_MARGIN = 2        # never chase a monster more than this many levels above the bot's base level
GATHER_EVERY = 4        # bots.ro_craft_policy: slot i % 4 == 1 gathers; gatherers never dive
BAND1_LEVEL = 5         # from this base level a bot hunts out to the inner half of the second field band
# (lo, hi) home-gate walk distances of the hunting ring per level band: below BAND1_LEVEL the ring is field band 0
# (game/ro_spawns: walk distance < HUNT_RADIUS, Blobbin-class prey; it used to reach 2 * HUNT_RADIUS, half of it in
# band 1, where the level 9-16 kinds killed level-1 bots); from BAND1_LEVEL band 0's outer half and band 1's inner
# half (Bramblet / Fiddlehop; the aggressive Pilferbug and Vexblob / Capling live in band 1's outer half).
RING_BY_LEVEL = ((HUNT_RADIUS // 3, HUNT_RADIUS), (HUNT_RADIUS * 4 // 5, HUNT_RADIUS * 3 // 2))
FOLLOW_MIN = 6          # a party member farther than this (manhattan) from its leader walks toward it
WARP_WAIT_TICKS = 4     # a bot left standing on a warp tile (destination busy) waits all but one tick in four

# Dive level per dungeon (index = DUNGEONS index): the Culvert from DUNGEON_MIN_LEVEL (Pilferbug 6, Clackbone 10,
# Rotling 15, Capling 16), Pinehollow Cave only from 25 (its Soldier Skeletons are level 29 and aggressive).
DUNGEON_DIVE_LEVEL = (DUNGEON_MIN_LEVEL, 25)
assert len(DUNGEON_DIVE_LEVEL) == len(DUNGEONS) and min(DUNGEON_DIVE_LEVEL) == DUNGEON_MIN_LEVEL
_DIVE_LEVEL = jnp.asarray(DUNGEON_DIVE_LEVEL, jnp.int32)  # [ND]

_SLOT = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)
_HOME = _SLOT % len(TOWNS)
_IS_GATHERER = _SLOT % GATHER_EVERY == 1
_STEPS = DELTAS[1:]  # [K, 2]
_RING = jnp.asarray(RING_BY_LEVEL, jnp.int32)  # [2, 2]


def _tile_mask(tiles):
    m = np.zeros((MAP, MAP), np.bool_)
    t = np.asarray(tiles).reshape(-1, 2)
    m[t[:, 0], t[:, 1]] = True
    return jnp.asarray(m)


PORTAL_TILES = _tile_mask(portal_warp_tiles()[0])       # zone-portal warp sources: taken when walking away
ENTRANCE_TILES = _tile_mask(dungeon_pads()[:, 0])       # dungeon entrance warps: only diving bots step on them


def _shop_of_town():
    """int32 [NT, 2]: the middle tile of each town's shop row (the waybell tile for a layout without shop rows)."""
    out = np.array(waybell_tiles(), np.int32)
    try:
        from game.world_layout import shop_tiles
        tiles, town = shop_tiles()
    except ImportError:  # pragma: no cover - layouts without shop rows
        return jnp.asarray(out)
    for t in range(len(TOWNS)):
        row = tiles[town == t]
        if len(row):
            out[t] = row[len(row) // 2]
    return jnp.asarray(out)


SHOP_OF_TOWN = _shop_of_town()


def blocked(world, pos, deltas):
    """[N, K] True where stepping pos [N, 2] by deltas [N, K, 2] hits a wall or the map edge; a diagonal may
    not cut a wall corner (the movement phase refuses those steps)."""
    walls = world["walls"]
    raw = pos[:, None, :] + deltas
    out = jnp.any((raw < 0) | (raw >= MAP), axis=-1)
    tgt = jnp.clip(raw, 0, MAP - 1)
    side_x = walls[tgt[..., 0], pos[:, None, 1]] != 0
    side_y = walls[pos[:, None, 0], tgt[..., 1]] != 0
    diag = (deltas[..., 0] != 0) & (deltas[..., 1] != 0)
    return out | (walls[tgt[..., 0], tgt[..., 1]] != 0) | (diag & (side_x | side_y))


def neighbours(pos):
    """Return (tgt [N, K, 2] clipped neighbour tiles, deltas [N, K, 2])."""
    deltas = jnp.broadcast_to(_STEPS[None], (pos.shape[0], N_DIRS, 2))
    return jnp.clip(pos[:, None, :] + deltas, 0, MAP - 1), deltas


def descend(own, nbr, bad):
    """Smallest-neighbour step: (move code [N], can [N]); `bad` [N, K] neighbours are never chosen."""
    nbr = jnp.where(bad, DIST_UNREACHABLE, nbr.astype(jnp.int32))
    move = (jnp.argmin(nbr, axis=1) + 1).astype(jnp.int32)
    return move, nbr.min(axis=1) < own.astype(jnp.int32)


def home_step(world):
    """(move, can) down the warp-aware home-gate field, never through other living players' tiles."""
    pos = world["players"]["pos"]
    field = world["dist_town"]
    tgt, deltas = neighbours(pos)
    own = field[_HOME, pos[:, 0], pos[:, 1]]
    nbr = field[_HOME[:, None], tgt[..., 0], tgt[..., 1]]
    return descend(own, nbr, blocked(world, pos, deltas))


def dungeon_fields(terrain):
    """Return int16 [ND, MAP, MAP]: walking distance (8-connected, no corner cutting, relaxed across warps so
    d[src] <= d[dst] + 1 like map_gen.distance_fields_ro) from every tile to each dungeon's entrance warp tile;
    DIST_UNREACHABLE where no walk exists. Once per world; one dungeon at a time so the loop's temporaries are
    [MAP, MAP]."""
    free = blocking(terrain) == 0
    ent = np.asarray(dungeon_pads())[:, 0]  # [ND, 2]
    nd = ent.shape[0]
    d0 = jnp.full((nd, MAP, MAP), DIST_UNREACHABLE, jnp.int32).at[jnp.arange(nd), ent[:, 0], ent[:, 1]].set(0)
    big = DIST_UNREACHABLE
    src, dst = warp_table(terrain)
    real = src[:, 0] >= 0
    sx, sy = jnp.clip(src[:, 0], 0, MAP - 1), jnp.clip(src[:, 1], 0, MAP - 1)
    dx_, dy_ = jnp.clip(dst[:, 0], 0, MAP - 1), jnp.clip(dst[:, 1], 0, MAP - 1)

    def shift(a, dx, dy, fill):
        p = jnp.pad(a, ((1, 1), (1, 1)), constant_values=fill)
        return p[1 + dx:1 + dx + MAP, 1 + dy:1 + dy + MAP]

    # diagonal (dx, dy) usable from tile t when both t+(dx,0) and t+(0,dy) are free
    fr = free.astype(jnp.int32)
    diag_ok = {(dx, dy): (shift(fr, dx, 0, 0) & shift(fr, 0, dy, 0)) != 0
               for dx in (-1, 1) for dy in (-1, 1)}

    def relax(d):
        best = _shift_min(d)
        for (dx, dy), ok in diag_ok.items():
            best = jnp.minimum(best, jnp.where(ok, shift(d, dx, dy, big), big))
        d = jnp.where(free, jnp.minimum(d, jnp.minimum(best + 1, big)), big)
        via = jnp.where(real, jnp.minimum(d[dx_, dy_] + 1, big), big)
        return d.at[sx, sy].min(via)

    def body(carry):
        d, _ = carry
        nd_ = relax(d)
        return nd_, jnp.any(nd_ != d)

    def one(d):
        d, _ = jax.lax.while_loop(lambda c: c[1], body, (d, jnp.bool_(True)))
        return d.astype(jnp.int16)

    return jax.lax.map(one, d0)


def with_dungeon_fields(world):
    """The world plus world["dist_dungeon"] (one world, unbatched; use lax.map over a batch)."""
    return {**world, "dist_dungeon": dungeon_fields(world["terrain"])}


def _max_hp(world):
    if "stats" in world:
        from game import ro_stats
        return ro_stats.derived(world)["max_hp"].astype(jnp.int32)
    from game.config import PLAYER_MAX_HP
    return jnp.full((MAX_PLAYERS,), PLAYER_MAX_HP, jnp.int32)


def _base_level(world):
    if "stats" in world and "base_level" in world["stats"]:
        return world["stats"]["base_level"].astype(jnp.int32)
    return jnp.ones((MAX_PLAYERS,), jnp.int32)


def _sitting(world):
    """bool [MAX_PLAYERS]: the anim action is SIT (the sit input toggles it; no anim block: nobody sits)."""
    if "anim" not in world:
        return jnp.zeros((MAX_PLAYERS,), jnp.bool_)
    from game.ro_anim import SIT
    return world["anim"]["action"][:MAX_PLAYERS].astype(jnp.int32) == SIT


def prey_cost(world, idx):
    """(hit_pct, cost) int32 [MAX_PLAYERS] against monster idx [MAX_PLAYERS] (each bot's nearest monster):
    the bot's chance in percent to hit it (combat: clip(80 + HIT - (level + AGI), 5, 95)) and the HP it expects to
    lose killing it from its current hp: swings = ceil(hp_left / dmg) * 100 / hit_pct at derived aspd_ticks, the
    monster swinging every MON_SWING_TICKS with clip(80 + level + DEX - FLEE, 5, 95) for its mean ATK less the
    bot's armour (hard DEF percent, then VIT flat), plus one swing for the opening. Buffs and elements are left out."""
    from game import ro_stats
    mon = world["monsters"]
    kind = jnp.clip(mon["kind"][idx], 0, mdata.LEVEL.shape[0] - 1)
    if "stats" in world:
        d = ro_stats.derived(world)
        atk, hit, flee, soft, aspd = d["atk"], d["hit"], d["flee"], d["def"], d["aspd_ticks"]
    else:  # pragma: no cover - worlds without the stats block (legacy tests)
        one = jnp.ones((MAX_PLAYERS,), jnp.int32)
        atk, hit, flee, soft, aspd = one * 10, one, one, one * 0, one * 10
    hard = jnp.zeros((MAX_PLAYERS,), jnp.int32)
    if "items" in world:
        from game import ro_items
        eq = ro_items.equipped(world)
        atk = atk + eq["weapon_atk"]
        hard = jnp.clip(eq["armor_def"], 0, 80)
    hit_pct = jnp.clip(80 + hit - (mdata.LEVEL[kind] + mdata.AGI[kind]), 5, 95)
    dmg = jnp.maximum(atk * (100 - jnp.clip(mdata.DEF[kind], 0, 80)) // 100 - mdata.VIT[kind], 1)
    hp_left = jnp.maximum(mon["hp"][idx], 1)
    hits = (jnp.minimum(hp_left, 1 << 20) + dmg - 1) // dmg
    swings = jnp.minimum((hits * 100 + hit_pct - 1) // hit_pct, 1000)  # int32-safe
    ticks = swings * jnp.clip(aspd, 1, 100)
    m_hit = jnp.clip(80 + mdata.LEVEL[kind] + mdata.DEX[kind] - flee, 5, 95)
    m_dmg = jnp.maximum((mdata.ATK_MIN[kind] + mdata.ATK_MAX[kind]) // 2 * (100 - hard) // 100 - soft, 1)
    cost = (ticks // MON_SWING_TICKS[kind] + 1) * m_hit * m_dmg // 100
    return hit_pct.astype(jnp.int32), cost.astype(jnp.int32)


def modes(world):
    """Per-player masks the composed policy uses: hurt, rest, in_dungeon, dive (all bool [MAX_PLAYERS])."""
    pl = world["players"]
    live = pl["active"] & ~pl["dead"]
    hp, mx = pl["hp"].astype(jnp.int32), jnp.maximum(_max_hp(world), 1)
    on_pz = world["pz"][pl["pos"][:, 0], pl["pos"][:, 1]]
    in_dun = dungeon_index(pl["pos"]) >= 0
    hurt = live & (hp * 100 < RETREAT_PCT * mx)
    rest = live & on_pz & (hp * 100 < REST_PCT * mx)
    has_field = "dist_dungeon" in world
    dive = (live & ~_IS_GATHERER & (_base_level(world) >= DUNGEON_MIN_LEVEL) & ~hurt & ~rest
            & (hp * 100 >= DIVE_PCT * mx) & has_field)
    return {"live": live, "hurt": hurt, "rest": rest, "in_dungeon": in_dun & live, "dive": dive, "on_pz": on_pz}


def _occupied(world, tgt):
    """[N, K] True where a neighbour tile holds a living player (one MAP*MAP bool table per world)."""
    pl = world["players"]
    live = pl["active"] & ~pl["dead"]
    flat = jnp.where(live, pl["pos"][:, 0] * MAP + pl["pos"][:, 1], MAP * MAP)
    occ = jnp.zeros((MAP * MAP,), jnp.bool_).at[flat].set(True, mode="drop")
    return occ[tgt[..., 0] * MAP + tgt[..., 1]]


def _pick(cost, ok):
    """(move code, any): the lowest-cost ok neighbour (ties to the lowest code)."""
    c = jnp.where(ok, cost, 1 << 30)
    return (jnp.argmin(c, axis=1) + 1).astype(jnp.int32), ok.any(axis=1)


def nav_moves(world, key, base_move, shop=None):
    """Return (move int32, sit bool, attack bool, fight bool), each [MAX_PLAYERS].

    base_move is accepted for the composition's call shape and ignored: every slot (gatherers included) roams,
    hunts, retreats and rests by these rules; the composition keeps a gatherer's craft move only while the craft
    policy reports it busy and `fight` is off. `fight` is True for a bot that chases or stands next to a monster.

    shop (optional bool [P]): bots that want to buy; on a protection zone and not hurt they walk to their home town's
    shop row (greedy over free tiles) and stop next to it.

    Hunting (every live bot that is not hurt, not on a protection zone): chase the nearest monster within
    BOT_CHASE_RANGE by stepping to the free neighbour (no wall, no corner cut, no living player) with the smallest
    manhattan distance to it (equal to the current one allowed, so it sidesteps obstacles), attack it when orthogonally
    adjacent. Only prey the bot can win is chased or hit (prey_cost: hit chance >= MIN_HIT_PCT, expected HP loss x
    COST_SAFETY_PCT within the hp above FLOOR_PCT; never more than LEVEL_MARGIN levels above the bot); an adjacent
    monster that fails it is hit only once provoked (aggressive, or already damaged). Below RETREAT_PCT a bot sits
    where no provoked or aggressive monster is within SAFE_DIST (outside dungeons) until REST_PCT, else walks home;
    `sit` is the engine's toggle (sent once to sit down, once to stand up). A bot with no such neighbour takes a random free step; one that is "bored" this window (BORED_PCT of
    (slot, tick // BORED_TICKS) windows) keeps its roam move: that breaks the stateless chase loops at tree lines and
    around monsters other bots surround. A heading step into a blocked tile is replaced by a random free one.
    """
    pl, mon = world["players"], world["monsters"]
    pos = pl["pos"]
    m = modes(world)
    level = _base_level(world)
    k_rand, k_bored = jax.random.split(key)

    table, _ = build_grid(mon["pos"], mon["active"])
    idx, dist = nearest_in_block(table, pos, mon["pos"])
    mkind = jnp.clip(mon["kind"][idx], 0, MON_LEVEL.shape[0] - 1)
    weak = MON_LEVEL[mkind] <= level + LEVEL_MARGIN
    # prey choice: hittable and affordable from the current hp (prey_cost); provoked monsters are fought back
    hp, mx = pl["hp"].astype(jnp.int32), jnp.maximum(_max_hp(world), 1)
    hit_pct, cost = prey_cost(world, idx)
    win = (hit_pct >= MIN_HIT_PCT) & (cost * COST_SAFETY_PCT <= hp * 100 - FLOOR_PCT * mx)
    provoked = mon["active"][idx] & (mdata.AGGRO[mkind] | (mon["hp"][idx] < mdata.HP[mkind]))
    near = (dist <= BOT_CHASE_RANGE) & (dist > 0) & weak & win
    fighting = dist <= 2
    # field rest (sit where it is safe instead of walking home): no aggressive / provoked monster close, not in a
    # dungeon; a bot starts below RETREAT_PCT and keeps sitting (the anim SIT action is its memory) until REST_PCT
    sitting = _sitting(world)
    threat = provoked & (dist <= SAFE_DIST)
    below_rest = m["live"] & (hp * 100 < REST_PCT * mx)
    field_rest = below_rest & ~m["on_pz"] & ~m["in_dungeon"] & ~threat & (m["hurt"] | sitting)
    resting = m["rest"] | field_rest

    tgt, deltas = neighbours(pos)
    free_all = ~blocked(world, pos, deltas) & ~_occupied(world, tgt)
    free = free_all & ~ENTRANCE_TILES[tgt[..., 0], tgt[..., 1]]  # only a dive steps onto a dungeon entrance
    mpos = mon["pos"][idx]
    md = jnp.sum(jnp.abs(tgt - mpos[:, None, :]), axis=2)  # [N, K]
    chase_mv, _ = _pick(md, free)
    # an equal-distance sidestep is allowed: it is how a chase goes round a tree or a bot (boredom ends loops)
    can_chase = jnp.min(jnp.where(free, md, 1 << 30), axis=1) <= dist

    k_head = jax.random.fold_in(jax.random.PRNGKey(7), world["tick"] // HEADING_TICKS)
    heading = jax.random.randint(k_head, (MAX_PLAYERS,), 1, N_DIRS + 1)
    rand_cost = jax.random.uniform(k_rand, (MAX_PLAYERS, N_DIRS))
    rand_mv, any_free = _pick(rand_cost, free)
    rand_mv = jnp.where(any_free, rand_mv, 0)

    def unblock(mv):
        ok = jnp.take_along_axis(free, jnp.clip(mv - 1, 0, N_DIRS - 1)[:, None], axis=1)[:, 0]
        return jnp.where((mv > 0) & ~ok, rand_mv, mv)

    heading = unblock(heading)
    k_b = jax.random.fold_in(jax.random.PRNGKey(11), world["tick"] // BORED_TICKS)
    bored = jax.random.uniform(k_b, (MAX_PLAYERS,)) * 100 < BORED_PCT
    able = m["live"] & ~m["hurt"] & ~m["on_pz"] & ~field_rest
    # a hurt bot next to a provoked monster it can still afford finishes it instead of turning its back on it
    finish = m["live"] & m["hurt"] & ~m["on_pz"] & (dist == 1) & provoked & win & weak
    adjacent = (able & (dist == 1) & ((win & weak) | provoked)) | finish
    chasing = able & near & ~bored & can_chase

    home, can_home = home_step(world)

    # roaming on the home-gate field (the base policy's rule, with HP as a fraction of max_hp and a ring by level):
    # leave town through the gate, walk away from it to ring_lo, keep a heading out to ring_hi, turn back beyond
    field = world["dist_town"]
    own = field[_HOME, pos[:, 0], pos[:, 1]].astype(jnp.int32)
    nbr = field[_HOME[:, None], tgt[..., 0], tgt[..., 1]].astype(jnp.int32)
    ring = _RING[(level >= BAND1_LEVEL).astype(jnp.int32)]  # [N, 2]
    ring_lo, ring_hi = ring[:, 0], ring[:, 1]
    toward, can_toward = _pick(nbr, free)
    can_toward = can_toward & (jnp.min(jnp.where(free, nbr, 1 << 30), axis=1) < own)
    town_nbr = world["pz"][tgt[..., 0], tgt[..., 1]]
    away_ok = free & ~town_nbr & (nbr < DIST_UNREACHABLE)  # never step back into town (the gate-jam lesson)
    # a zone portal's warp tile is a dead end of the road that the warp phase carries through: prefer it
    portal_nbr = PORTAL_TILES[tgt[..., 0], tgt[..., 1]]
    away_val = nbr + jnp.where(portal_nbr, 2, 0)
    away, _ = _pick(-away_val, away_ok)
    can_away = jnp.max(jnp.where(away_ok, away_val, -1), axis=1) > own
    roam = heading
    roam = jnp.where(~m["on_pz"] & (own < ring_lo) & can_away, away, roam)
    roam = jnp.where((m["on_pz"] | (own > ring_hi)) & can_toward, toward, roam)
    move = roam
    if "dist_dungeon" in world:
        dfield = world["dist_dungeon"]
        here = dfield[:, pos[:, 0], pos[:, 1]].astype(jnp.int32)  # [ND, N]
        here = jnp.where(level[None, :] >= _DIVE_LEVEL[:, None], here, DIST_UNREACHABLE)  # eligible dungeons only
        which = jnp.argmin(here, axis=0).astype(jnp.int32)
        down = here[which, _SLOT]
        dnbr = dfield[which[:, None], tgt[..., 0], tgt[..., 1]]
        dmove, dcan = descend(down, dnbr, ~free_all)
        go_dun = m["dive"] & ~m["in_dungeon"] & ~fighting & (down < DIST_UNREACHABLE) & (dcan | (down == 0))
        # standing on the entrance warp already: the warp phase moves the bot; stay put
        move = jnp.where(go_dun, jnp.where(down == 0, 0, dmove), move)
    # inside a dungeon the base policy's "too far from the gate" would walk the bot straight out: wander instead
    move = jnp.where(m["in_dungeon"] & ~m["hurt"], heading, move)
    if "party" in world:
        lpos, has_leader, _ = leader_of(world)
        ld = jnp.sum(jnp.abs(pos - lpos), axis=1)
        lead_pz = world["pz"][lpos[:, 0], lpos[:, 1]]
        same_zone = zone_of(pos) == zone_of(lpos)
        follow = has_leader & able & ~lead_pz & same_zone & (ld > FOLLOW_MIN)
        fd = jnp.sum(jnp.abs(tgt - lpos[:, None, :]), axis=2)
        follow_mv, can_follow = _pick(fd, free)
        move = jnp.where(follow & can_follow, follow_mv, move)
    move = jnp.where(able & near & ~bored & ~can_chase, rand_mv, move)  # boxed in: jiggle free
    move = jnp.where(chasing, chase_mv, move)
    move = jnp.where(adjacent, 0, move)
    move = jnp.where(m["hurt"] & ~m["on_pz"] & ~field_rest & ~finish & can_home, home, move)  # unsafe: walk home
    move = jnp.where(resting, 0, move)  # a hurt bot that reached town rests too
    if shop is not None:
        goal = SHOP_OF_TOWN[_HOME]
        cheb = jnp.max(jnp.abs(tgt - goal[:, None, :]), axis=2)
        shop_mv, _ = _pick(cheb, free)
        far = jnp.max(jnp.abs(pos - goal), axis=1) > 1
        move = jnp.where(shop & m["on_pz"] & ~m["hurt"], jnp.where(far, shop_mv, 0), move)  # wait there to buy
    # left standing on a warp tile (its destination was busy this tick): wait for the warp phase, mostly
    on_warp = world["terrain"][pos[:, 0], pos[:, 1]] == T_WARP
    move = jnp.where(on_warp & (world["tick"] % WARP_WAIT_TICKS != 0), 0, move)
    move = jnp.where(m["live"], move, 0).astype(jnp.int32)
    # the engine's sit input is a toggle (game/survival.py): sit down once when resting starts, stand up once when it
    # ends while still sitting (sending it every resting tick, as before, stood the bot up every other tick)
    resting = resting & (move == 0)
    sit = m["live"] & (resting != sitting)
    return move, sit, adjacent, chasing | adjacent

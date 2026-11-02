"""Bounded A* path search for click-to-walk."""
import heapq
from itertools import count

import numpy as np

from game.config import DIR_DELTAS, PATH_RADIUS


def _walkable(walls: np.ndarray, x: int, y: int) -> bool:
    """True when (x, y) is inside the map and not a wall."""
    return 0 <= x < walls.shape[0] and 0 <= y < walls.shape[1] and walls[x, y] == 0


def _in_box(start, cell, radius: int) -> bool:
    """True when cell lies within start +/- radius on both axes."""
    return abs(cell[0] - start[0]) <= radius and abs(cell[1] - start[1]) <= radius


def _rebuild(came: dict, goal) -> list[int]:
    """Walk parent links back from goal into a move-code list."""
    moves = []
    while came[goal] is not None:
        goal, move = came[goal]
        moves.append(move)
    return moves[::-1]


def find_path(walls: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
              max_radius: int = PATH_RADIUS) -> list[int] | None:
    """Return move codes 1..4 from start to goal, or None if no bounded path."""
    start, goal = tuple(start), tuple(goal)
    if not _walkable(walls, *goal) or not _in_box(start, goal, max_radius):
        return None
    tie = count()
    heap = [(0, next(tie), 0, start)]
    came = {start: None}
    best = {start: 0}
    while heap:
        _, _, g, cur = heapq.heappop(heap)
        if cur == goal:
            return _rebuild(came, goal)
        if g > best[cur]:
            continue
        for move in range(1, 5):
            dx, dy = DIR_DELTAS[move]
            nxt = (cur[0] + dx, cur[1] + dy)
            if not _walkable(walls, *nxt) or not _in_box(start, nxt, max_radius):
                continue
            if g + 1 < best.get(nxt, 1 << 30):
                best[nxt] = g + 1
                came[nxt] = (cur, move)
                h = abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1])
                heapq.heappush(heap, (g + 1 + h, next(tie), g + 1, nxt))
    return None

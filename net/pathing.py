"""Bounded 8-connected A* path search for click-to-walk (RO costs: 10 straight, 14 diagonal)."""
import heapq
from itertools import count

import numpy as np

from game.config import PATH_RADIUS
from game.ro_move import DIR_DELTAS9


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


STRAIGHT_COST, DIAG_COST = 10, 14


def _octile(a, b) -> int:
    """Octile distance 10*max + 4*min: admissible and consistent for the 10/14 costs."""
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return 10 * max(dx, dy) + 4 * min(dx, dy)


def find_path(walls: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
              max_radius: int = PATH_RADIUS) -> list[int] | None:
    """Return move codes 1..8 (DIR_DELTAS9) from start to goal, or None if no bounded path.

    Cost-optimal for 10 per straight and 14 per diagonal step. A diagonal step needs
    both orthogonal neighbours and the target walkable (no corner cutting). The search
    stays inside the Chebyshev box of max_radius around start.
    """
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
        for move in range(1, 9):
            dx, dy = DIR_DELTAS9[move]
            nxt = (cur[0] + dx, cur[1] + dy)
            if not _walkable(walls, *nxt) or not _in_box(start, nxt, max_radius):
                continue
            diag = dx != 0 and dy != 0
            if diag and not (_walkable(walls, cur[0] + dx, cur[1])
                             and _walkable(walls, cur[0], cur[1] + dy)):
                continue
            ng = g + (DIAG_COST if diag else STRAIGHT_COST)
            if ng < best.get(nxt, 1 << 30):
                best[nxt] = ng
                came[nxt] = (cur, move)
                heapq.heappush(heap, (ng + _octile(nxt, goal), next(tie), ng, nxt))
    return None

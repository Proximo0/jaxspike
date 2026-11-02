import numpy as np
from game.config import DIR_DELTAS, MAP, PATH_RADIUS
from net.pathing import find_path

def replay(start, moves):
    x, y = start
    for m in moves:
        dx, dy = DIR_DELTAS[m]; x, y = x + dx, y + dy
    return x, y

def test_straight_line():
    walls = np.zeros((MAP, MAP), np.int8)
    p = find_path(walls, (10, 10), (10, 15))
    assert p == [2] * 5

def test_around_wall():
    walls = np.zeros((MAP, MAP), np.int8); walls[12, 5:16] = 1
    p = find_path(walls, (10, 10), (14, 10))
    assert p is not None and replay((10, 10), p) == (14, 10)
    assert all(walls[xy] == 0 for xy in [replay((10, 10), p[:i]) for i in range(len(p) + 1)])
    assert len(p) == 4 + 2 * 6  # detour around the wall end at y=16

def test_unreachable_and_invalid_goal():
    walls = np.zeros((MAP, MAP), np.int8); walls[8:13, 8:13] = 1; walls[10, 10] = 0
    assert find_path(walls, (10, 10), (20, 10)) is None      # boxed in
    assert find_path(walls, (20, 10), (9, 9)) is None        # goal is a wall
    assert find_path(walls, (20, 10), (MAP, 10)) is None     # out of map

def test_radius_cutoff():
    walls = np.zeros((MAP, MAP), np.int8)
    assert find_path(walls, (100, 100), (100, 100 + PATH_RADIUS)) is not None
    assert find_path(walls, (100, 100), (100, 101 + PATH_RADIUS)) is None

def test_path_to_self_and_adjacent():
    walls = np.zeros((MAP, MAP), np.int8)
    assert find_path(walls, (10, 10), (10, 10)) == []
    assert find_path(walls, (10, 10), (11, 10)) == [4]

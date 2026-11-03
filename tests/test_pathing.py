import numpy as np
from game.config import MAP, PATH_RADIUS
from game.ro_move import DIR_DELTAS9
from net.pathing import find_path

def replay(start, moves):
    x, y = start
    for m in moves:
        dx, dy = DIR_DELTAS9[m]; x, y = x + dx, y + dy
    return x, y

def test_straight_line():
    walls = np.zeros((MAP, MAP), np.int8)
    p = find_path(walls, (10, 10), (10, 15))
    assert p == [1] * 5

def cost(p):
    return sum(14 if DIR_DELTAS9[m][0] and DIR_DELTAS9[m][1] else 10 for m in p)

def ref_cost(walls, start, goal, box=60):
    """Independent Dijkstra (10/14, no corner cutting) inside a box around start."""
    import heapq
    ok = lambda x, y: (0 <= x < walls.shape[0] and 0 <= y < walls.shape[1] and walls[x, y] == 0
                       and abs(x - start[0]) <= box and abs(y - start[1]) <= box)
    dist, pq = {start: 0}, [(0, start)]
    while pq:
        d, (x, y) = heapq.heappop(pq)
        if (x, y) == goal:
            return d
        if d > dist[(x, y)]:
            continue
        for dx, dy in DIR_DELTAS9[1:]:
            if not ok(x + dx, y + dy):
                continue
            if dx and dy and not (ok(x + dx, y) and ok(x, y + dy)):
                continue
            nd = d + (14 if dx and dy else 10)
            if nd < dist.get((x + dx, y + dy), 1 << 30):
                dist[(x + dx, y + dy)] = nd
                heapq.heappush(pq, (nd, (x + dx, y + dy)))
    return None

def test_around_wall():
    walls = np.zeros((MAP, MAP), np.int8); walls[12, 5:16] = 1
    p = find_path(walls, (10, 10), (14, 10))
    assert p is not None and replay((10, 10), p) == (14, 10)
    cells = [replay((10, 10), p[:i]) for i in range(len(p) + 1)]
    assert all(walls[xy] == 0 for xy in cells)
    for (x, y), m in zip(cells, p):   # never cut a wall corner
        dx, dy = DIR_DELTAS9[m]
        if dx and dy:
            assert walls[x + dx, y] == 0 and walls[x, y + dy] == 0
    assert cost(p) == ref_cost(walls, (10, 10), (14, 10))
    assert cost(p) == 148             # corner rule forces straight steps beside the wall ends

def test_pure_diagonal():
    walls = np.zeros((MAP, MAP), np.int8)
    assert find_path(walls, (10, 10), (15, 15)) == [8] * 5

def test_no_corner_cutting():
    for wall_cells in ([(11, 10), (10, 11)], [(11, 10)], [(10, 11)]):
        walls = np.zeros((MAP, MAP), np.int8)
        for c in wall_cells:
            walls[c] = 1
        p = find_path(walls, (10, 10), (11, 11))
        assert p != [8]
        if len(wall_cells) == 1:
            assert p is not None and len(p) == 2 and cost(p) == 20
            assert replay((10, 10), p) == (11, 11)

def test_mixed_path():
    walls = np.zeros((MAP, MAP), np.int8)
    p = find_path(walls, (10, 10), (14, 12))
    assert replay((10, 10), p) == (14, 12)
    assert sorted(m in (2, 4, 6, 8) for m in p).count(True) == 2 and len(p) == 4
    assert cost(p) == 48

def test_unreachable_and_invalid_goal():
    walls = np.zeros((MAP, MAP), np.int8); walls[8:13, 8:13] = 1; walls[10, 10] = 0
    assert find_path(walls, (10, 10), (20, 10)) is None      # boxed in
    assert find_path(walls, (20, 10), (9, 9)) is None        # goal is a wall

def test_out_of_map_goal():
    walls = np.zeros((MAP, MAP), np.int8)
    assert find_path(walls, (MAP - 1, 10), (MAP, 10)) is None
    assert find_path(walls, (0, 10), (-1, 10)) is None       # no negative-index wrap

def test_search_nodes_stay_in_box():
    walls = np.zeros((MAP, MAP), np.int8)
    r = PATH_RADIUS
    walls[100 - r:100 + r + 1, 110] = 1                      # spans the full box width
    assert find_path(walls, (100, 100), (100, 120)) is None  # only gap is outside the box
    assert find_path(walls, (100, 100), (100, 120), max_radius=r + 10) is not None

def test_deterministic():
    walls = np.zeros((MAP, MAP), np.int8); walls[12, 5:16] = 1
    assert find_path(walls, (10, 10), (14, 10)) == find_path(walls, (10, 10), (14, 10))

def test_radius_cutoff():
    walls = np.zeros((MAP, MAP), np.int8)
    assert find_path(walls, (100, 100), (100, 100 + PATH_RADIUS)) is not None
    assert find_path(walls, (100, 100), (100, 101 + PATH_RADIUS)) is None

def test_radius_x_axis_and_diagonal():
    walls = np.zeros((MAP, MAP), np.int8)
    r = PATH_RADIUS
    assert find_path(walls, (100, 100), (100 + r, 100 + r)) is not None
    assert find_path(walls, (100, 100), (101 + r, 100)) is None

def test_path_to_self_and_adjacent():
    walls = np.zeros((MAP, MAP), np.int8)
    assert find_path(walls, (10, 10), (10, 10)) == []
    assert find_path(walls, (10, 10), (11, 10)) == [7]

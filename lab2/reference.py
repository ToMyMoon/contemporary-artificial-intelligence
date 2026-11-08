"""复核最短路和宝藏博弈的独立实现。"""
from functools import lru_cache


def reference_distance(cells, start, goal, weighted=False, free_goal=False):
    """数组扫描版 Dijkstra，无优先队列，不调用被测 A* 的邻居或代价函数。"""
    rows, cols = len(cells), len(cells[0])
    remaining = {(r, c) for r in range(rows) for c in range(cols) if cells[r][c] != "X"}
    if start not in remaining or goal not in remaining:
        return -1
    dist = {start: 0}
    while remaining:
        p = min(remaining, key=lambda p: dist.get(p, float("inf")))
        if p not in dist:
            return -1
        remaining.remove(p)
        if p == goal:
            return dist[p]
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            q = (p[0]+dr, p[1]+dc)
            if q not in remaining:
                continue
            token = str(cells[q[0]][q[1]])
            cost = int(token) if weighted and token.isdigit() else 1
            if free_goal and q == goal:
                cost = 0
            dist[q] = min(dist.get(q, float("inf")), dist[p]+cost)
    return -1


def reference_game(game, depth):
    """用未来收益差递推，独立于 State/move/solve_game；缓存仅用于测试。"""
    @lru_cache(None)
    def value(a, b, remaining, player, left):
        if left == 0 or not remaining:
            return 0, None
        options = []
        for t in remaining:
            rest = tuple(x for x in remaining if x != t)
            if player == 0:
                future = value(t, b, rest, 1, left-1)[0]
                score = game.values[t] - game.distances[a][t] + future
            else:
                future = value(a, t, rest, 0, left-1)[0]
                score = -(game.values[t] - game.distances[b][t]) + future
            options.append((score, t))
        best = max(x[0] for x in options) if player == 0 else min(x[0] for x in options)
        return best, min(t for score, t in options if score == best)
    score, t = value(0, 1, tuple(game.values), 0, depth)
    return dict(best_target=game.names[t] if t is not None else None, best_value=score)

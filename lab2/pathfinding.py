"""四邻域网格上的 A*；题目一、二的直接调用接口位于文件末尾。"""

from dataclasses import dataclass, field
import heapq
from itertools import count
from math import inf

Position = tuple[int, int]
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


@dataclass
class PathResult:
    path: list[Position]
    cost: int
    expanded: int = 0
    generated: int = 0
    max_frontier: int = 0
    expanded_order: list[Position] = field(default_factory=list)


class Grid:
    def __init__(self, cells, weighted=False, free_goal=None):
        if not cells or not cells[0] or any(len(row) != len(cells[0]) for row in cells):
            raise ValueError("地图必须是非空矩形")
        self.cells = tuple(tuple(row) for row in cells)
        self.rows, self.cols = len(cells), len(cells[0])
        self.weighted = weighted
        self.free_goal = free_goal
        for r in range(self.rows):
            for c in range(self.cols):
                if self.passable((r, c)):
                    self.enter_cost((r, c))

    def passable(self, p):
        r, c = p
        return 0 <= r < self.rows and 0 <= c < self.cols and str(self.cells[r][c]) != "X"

    def enter_cost(self, p):
        if p == self.free_goal:
            return 0
        value = self.cells[p[0]][p[1]]
        if not self.weighted or str(value) in ("S", "E", "A", "B", ".") or str(value).startswith("T"):
            return 1
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"非法地形代价: {value}") from exc
        if result < 0 or result != float(value):
            raise ValueError("地形代价必须是非负整数")
        return result

    def neighbors(self, p):
        for dr, dc in DIRECTIONS:
            q = (p[0] + dr, p[1] + dc)
            if self.passable(q):
                yield q


def astar(grid, start, goal, heuristic="manhattan", trace=False):
    """同 f 时按入队顺序处理；失效条目不计入 expanded，终点计入。"""
    if heuristic not in ("zero", "manhattan", "corrected"):
        raise ValueError("heuristic 必须是 zero、manhattan 或 corrected")
    if not grid.passable(start) or not grid.passable(goal):
        return PathResult([], -1)
    if grid.free_goal is not None and grid.free_goal != goal and heuristic != "zero":
        raise ValueError("免费终点必须与此次搜索终点一致")
    if grid.free_goal == goal and heuristic == "manhattan":
        raise ValueError("免费终点应使用 corrected 或 zero 启发函数")
    if heuristic == "corrected" and grid.free_goal != goal:
        raise ValueError("corrected 仅用于免费终点")
    # 预处理也属于一次搜索的成本；有普通零代价格时，下界自动退化为零。
    costs = [grid.enter_cost((r, c)) for r in range(grid.rows) for c in range(grid.cols)
             if grid.passable((r, c)) and (r, c) != grid.free_goal]
    minimum = min(costs, default=0)

    def h(p):
        if heuristic == "zero":
            return 0
        distance = abs(p[0] - goal[0]) + abs(p[1] - goal[1])
        return minimum * max(distance - (heuristic == "corrected"), 0)

    serial = count()
    queue = [(h(start), next(serial), 0, start)]
    distances, parent = {start: 0}, {}
    result = PathResult([], -1, generated=1, max_frontier=1)
    while queue:
        _, _, g, p = heapq.heappop(queue)
        if g != distances[p]:
            continue
        result.expanded += 1
        if trace:
            result.expanded_order.append(p)
        if p == goal:
            result.cost = g
            result.path = [p]
            while p != start:
                p = parent[p]
                result.path.append(p)
            result.path.reverse()
            return result
        for q in grid.neighbors(p):
            new_g = g + grid.enter_cost(q)
            if new_g < distances.get(q, inf):
                distances[q], parent[q] = new_g, p
                heapq.heappush(queue, (new_g + h(q), next(serial), new_g, q))
                result.generated += 1
        result.max_frontier = max(result.max_frontier, len(queue))
    return result


def find_path(start, goal, map_data):
    return astar(Grid(map_data), tuple(start), tuple(goal)).path


def minimum_cost(start, goal, map_data):
    goal = tuple(goal)
    return astar(Grid(map_data, weighted=True, free_goal=goal), tuple(start), goal, "corrected").cost

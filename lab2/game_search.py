"""博弈树与宝藏分配。统计调用次数，不按对象身份合并重复状态。"""

from dataclasses import dataclass, field
from math import inf
import random
from pathfinding import astar, Grid


@dataclass
class SearchStats:
    nodes: int = 0
    leaves: int = 0
    cutoffs: int = 0
    tie_researches: int = 0


@dataclass
class TreeNode:
    value: int | None = None
    children: list = field(default_factory=list)

    def __post_init__(self):
        if self.children is None:
            self.children = []

    @classmethod
    def from_dict(cls, data, maximizing=True):
        if "tree_structure" in data:
            data = data["tree_structure"]
        if "is_maximizing_player" in data and data["is_maximizing_player"] != maximizing:
            raise ValueError("树的 MAX/MIN 层必须交替，根节点为 MAX")
        return cls(data.get("value"), [cls.from_dict(c, not maximizing) for c in data.get("children", [])])


def search_tree(root, pruning=True, order="fixed", seed=0, trace=False):
    if order not in ("fixed", "reverse", "random"):
        raise ValueError("未知访问顺序")
    stats, events = SearchStats(), []
    rng = random.Random(seed)

    def visit(node, maximizing, alpha, beta, path):
        stats.nodes += 1
        if not node.children:
            if node.value is None:
                raise ValueError("叶节点必须有 value")
            stats.leaves += 1
            if trace:
                events.append(dict(event="leaf", path=path, value=node.value))
            return node.value
        indices = list(range(len(node.children)))
        if order == "reverse":
            indices.reverse()
        elif order == "random":
            rng.shuffle(indices)
        best = -inf if maximizing else inf
        for j, i in enumerate(indices):
            value = visit(node.children[i], not maximizing, alpha, beta, path + [i])
            best = max(best, value) if maximizing else min(best, value)
            if maximizing:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            if trace:
                events.append(dict(event="update", path=path, child=i, value=best, alpha=alpha, beta=beta))
            if pruning and alpha >= beta and j + 1 < len(indices):
                stats.cutoffs += 1
                if trace:
                    events.append(dict(event="cut", path=path, skipped=indices[j+1:], alpha=alpha, beta=beta))
                break
        return best

    return visit(root, True, -inf, inf, []), stats, events


def alpha_beta(root):
    return search_tree(root)[0]


def minimax(root):
    return search_tree(root, pruning=False)[0]


@dataclass(frozen=True)
class State:
    a: int = 0
    b: int = 1
    remaining: tuple = ()
    score_a: int = 0
    score_b: int = 0
    player: int = 0

    @property
    def utility(self):
        return self.score_a - self.score_b


@dataclass
class TreasureGame:
    names: list
    positions: list
    values: dict
    distances: list

    @classmethod
    def from_case(cls, case):
        names = ["A", "B"] + sorted(case["treasures"])
        positions = [tuple(case["start_a"]), tuple(case["start_b"])]
        positions += [tuple(case["treasures"][name]["position"]) for name in names[2:]]
        grid = Grid(case["map"], weighted=True)
        distances = [[0 for _ in names] for _ in names]
        for i, start in enumerate(positions):
            for j, goal in enumerate(positions):
                if i != j:
                    distances[i][j] = astar(grid, start, goal).cost
                    if distances[i][j] < 0:
                        raise ValueError("宝藏场景要求所有标记位置互相可达")
        values = {i: case["treasures"][name]["value"] for i, name in enumerate(names) if i >= 2}
        return cls(names, positions, values, distances)

    def initial(self):
        return State(remaining=tuple(range(2, len(self.names))))

    def gain(self, state, action):
        position = state.a if state.player == 0 else state.b
        return self.values[action] - self.distances[position][action]

    def move(self, state, action):
        if action not in state.remaining:
            raise ValueError("不能重复选取已拿走的宝藏")
        gain = self.gain(state, action)
        rest = tuple(i for i in state.remaining if i != action)
        if state.player == 0:
            return State(action, state.b, rest, state.score_a + gain, state.score_b, 1)
        return State(state.a, action, rest, state.score_a, state.score_b + gain, 0)


def solve_game(game, depth, pruning=True, order="fixed", state=None, trace=None):
    """相同最优值时选编号最小的宝藏；必要的并列复查计入统计和耗时。"""
    if depth < 0 or order not in ("fixed", "gain"):
        raise ValueError("深度非负，顺序为 fixed 或 gain")
    initial = game.initial() if state is None else state
    stats = SearchStats()
    def visit(s, remaining_depth, alpha, beta, root=False):
        stats.nodes += 1
        if remaining_depth == 0 or not s.remaining:
            stats.leaves += 1
            return s.utility, None
        actions = list(s.remaining)
        if order == "gain":
            actions.sort(key=lambda a: (-game.gain(s, a), a))
        maximizing = s.player == 0
        best, best_action = (-inf if maximizing else inf), None
        for j, action in enumerate(actions):
            if root and trace is not None:
                before = stats.nodes
            child = game.move(s, action)
            value, _ = visit(child, remaining_depth - 1, alpha, beta)
            # 剪枝返回的边界可能只是表面并列。根节点要改选更小编号前先精确复查。
            if root and pruning and value == best and best_action is not None and action < best_action:
                stats.tie_researches += 1
                value, _ = visit(child, remaining_depth - 1, -inf, inf)
            if root and trace is not None:
                trace["root_branches"][game.names[action]] = stats.nodes - before
            better = value > best if maximizing else value < best
            if better or (value == best and (best_action is None or action < best_action)):
                best, best_action = value, action
            if maximizing:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            if pruning and alpha >= beta and j + 1 < len(actions):
                if trace is not None:
                    level = depth - remaining_depth
                    trace["cuts_by_level"][level] = trace["cuts_by_level"].get(level, 0) + 1
                stats.cutoffs += 1
                break
        return best, best_action

    if trace is not None:
        trace.update(root_branches={}, cuts_by_level={})
    value, action = visit(initial, depth, -inf, inf, True)
    return {"best_target": game.names[action] if action is not None else None,
            "best_value": value}, stats


def root_diagnostics(game, depth):
    """用完整 Minimax 单独求每个首步的精确值及后续路线，不放进计时。"""
    initial, result = game.initial(), []
    for first in initial.remaining:
        s = game.move(initial, first)
        value = solve_game(game, depth - 1, False, state=s)[0]["best_value"]
        steps = [dict(player="A", target=game.names[first], gain=game.gain(initial, first),
                      score_a=s.score_a, score_b=s.score_b)]
        for left in range(depth - 1, 0, -1):
            if not s.remaining:
                break
            choice = solve_game(game, left, False, state=s)[0]["best_target"]
            action = game.names.index(choice)
            player, gain = "AB"[s.player], game.gain(s, action)
            s = game.move(s, action)
            steps.append(dict(player=player, target=choice, gain=gain, score_a=s.score_a, score_b=s.score_b))
        result.append(dict(first=game.names[first], value=value, steps=steps))
    return result

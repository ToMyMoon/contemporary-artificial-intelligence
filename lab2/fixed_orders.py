"""在题四原图上穷举两名玩家共用的 24 种固定宝藏顺序。"""

from dataclasses import asdict, replace
from itertools import permutations
from statistics import median

from cases import TREASURE
from output import write_csv, write_json
from game_search import TreasureGame, solve_game
from reference import reference_game


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    game = TreasureGame.from_case(TREASURE)
    initial = game.initial()
    runs, summaries, traces = [], [], {}
    for depth in (2, 3, 4):
        expected = reference_game(game, depth)
        counts = []
        for actions in permutations(initial.remaining):
            # move 删除已取宝藏时保留其余元素的相对顺序，宝藏编号不变。
            state = replace(initial, remaining=actions)
            trace = {}
            result, stats = solve_game(game, depth, order="fixed", state=state, trace=trace)
            traces[f"{depth}/" + " ".join(game.names[a] for a in actions)] = trace
            assert result == expected, (depth, actions, result)
            counts.append(stats.nodes)
            runs.append(dict(depth=depth, order=" ".join(game.names[a] for a in actions),
                             **result, **asdict(stats)))
        trace = {}
        result, gain = solve_game(game, depth, order="gain", trace=trace)
        traces[f"{depth}/gain"] = trace
        assert result == expected
        summaries.append(dict(depth=depth, **result, gain_nodes=gain.nodes,
                              fixed_min=min(counts), fixed_median=median(counts), fixed_max=max(counts),
                              gain_fewer=sum(gain.nodes < n for n in counts),
                              equal=sum(gain.nodes == n for n in counts),
                              gain_more=sum(gain.nodes > n for n in counts)))
    write_csv(output / "fixed_orders.csv", runs)
    write_csv(output / "fixed_order_comparison.csv", summaries)
    write_json(output / "ordering_traces.json", traces)
    return summaries

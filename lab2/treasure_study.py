from dataclasses import asdict
from pathlib import Path

from cases import TREASURE
from output import benchmark, write_csv, write_json
from game_search import TreasureGame, solve_game, root_diagnostics
from reference import reference_distance, reference_game


def run(output, repeats=7, batch=1000):
    if repeats < 1 or batch < 1:
        raise ValueError("repeats 和 batch 至少为 1")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    game = TreasureGame.from_case(TREASURE)
    for i, p in enumerate(game.positions):
        for j, q in enumerate(game.positions):
            assert game.distances[i][j] == reference_distance(TREASURE["map"], p, q, True)

    metrics, timings, comparisons, diagnostics = [], [], [], {}
    for depth in [2, 3, 4]:
        expected = reference_game(game, depth)
        diagnostics[depth] = root_diagnostics(game, depth)
        rows = {}
        for method, pruning, order in [("minimax", False, "fixed"),
                                       ("ab_fixed", True, "fixed"),
                                       ("ab_gain", True, "gain")]:
            fn = lambda: solve_game(game, depth, pruning, order)
            result, stats = fn()
            assert result == expected, (depth, method, result, expected)
            timing, raw = benchmark(fn, repeats, batch)
            row = dict(case="task4", depth=depth, method=method,
                       **result, **asdict(stats), **timing)
            metrics.append(row)
            rows[method] = row
            timings.extend(dict(depth=depth, method=method, repeat=i,
                                batch=batch, ms_per_run=ms) for i, ms in enumerate(raw))
        fixed, gain = rows["ab_fixed"], rows["ab_gain"]
        comparisons.append(dict(depth=depth, fixed_nodes=fixed["nodes"],
                                gain_nodes=gain["nodes"],
                                saved_nodes=fixed["nodes"]-gain["nodes"],
                                saved_percent=100*(fixed["nodes"]-gain["nodes"])/fixed["nodes"],
                                fixed_ms=fixed["median_ms"], gain_ms=gain["median_ms"],
                                time_saved_percent=100*(1-gain["median_ms"]/fixed["median_ms"])))

    matrix = [dict(source=name, **dict(zip(game.names, row)))
              for name, row in zip(game.names, game.distances)]
    write_csv(output / "distance_matrix.csv", matrix)
    write_csv(output / "game_metrics.csv", metrics)
    write_csv(output / "timing_samples.csv", timings)
    write_csv(output / "depth_comparison.csv", comparisons)
    write_json(output / "root_choices.json", diagnostics)
    write_csv(output / "root_values.csv",
              [dict(depth=d, first=r["first"], value=r["value"])
               for d, choices in diagnostics.items() for r in choices])
    return matrix

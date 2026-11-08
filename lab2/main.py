"""运行报告的四道题，保存搜索记录和图表。"""
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
import json

from cases import MAZE, WEIGHTED, TREASURE, sample_tree, small_mazes
from fixed_orders import run as compare_fixed_orders
from game_search import TreeNode, TreasureGame, alpha_beta, search_tree, solve_game
from output import write_csv, write_json
from pathfinding import Grid, astar, find_path, minimum_cost
from plots import maze_figure, weighted_figure, tree_pruning_map, distance_heatmap
from reference import reference_distance
from treasure_study import run as compare_depths


def path_experiments(output, figures):
    mazes = [MAZE, *small_mazes()]
    all_paths = [*mazes, WEIGHTED]
    paths, path_detail = [], {}
    for case in all_paths:
        name, start, goal = case["name"], tuple(case["start"]), tuple(case["goal"])
        grid = Grid(case["map"], case["weighted"], goal if case["weighted"] else None)
        expected = reference_distance(case["map"], start, goal, case["weighted"], case["weighted"])
        for heuristic in ["zero", "corrected" if case["weighted"] else "manhattan"]:
            result = astar(grid, start, goal, heuristic, trace=True)
            assert result.cost == expected, (name, heuristic)
            paths.append(dict(case=name, heuristic=heuristic, cost=result.cost,
                              steps=len(result.path)-1 if result.path else -1,
                              expanded=result.expanded, generated=result.generated,
                              max_frontier=result.max_frontier))
            cumulative, accumulation = 0, []
            for i, p in enumerate(result.path):
                enter = grid.enter_cost(p) if i else 0
                cumulative += enter
                accumulation.append(dict(position=p, enter=enter, cumulative=cumulative))
            path_detail[f"{name}/{heuristic}"] = dict(**asdict(result), accumulation=accumulation)

    write_csv(output / "path_metrics.csv", paths)
    write_json(output / "path_traces.json", path_detail)
    maximum = max(path_detail[f"{case['name']}/{h}"]["expanded"]
                  for case in mazes for h in ("zero", "manhattan"))
    maze_figure(mazes[:1], path_detail, maximum, figures / "fig1_maze.png")
    maze_figure(mazes[1:], path_detail, maximum, figures / "fig2_detours.png")
    weighted_figure(WEIGHTED, path_detail, figures / "fig3_weighted.png")


def tree_experiments(output, figures):
    trees, tree_detail = [], {}
    tree = sample_tree()
    tree_variants = [(False, "fixed", 0), (True, "fixed", 0), (True, "reverse", 0)]
    tree_variants += [(True, "random", seed) for seed in range(5)]
    for pruning, order, seed in tree_variants:
        method = "ab_"+order if pruning else "minimax"
        value, stats, trace = search_tree(tree, pruning, order, seed, trace=True)
        assert value == 5
        trees.append(dict(method=method, seed=seed, value=value, **asdict(stats)))
        tree_detail[f"{method}/{seed}"] = trace
    write_csv(output / "tree_metrics.csv", trees)
    write_json(output / "tree_traces.json", tree_detail)
    tree_pruning_map(tree_detail["ab_fixed/0"], figures / "fig4_pruning.png")


def task_result(task, data, depth):
    if task in (1, 2):
        case = data if data is not None else (MAZE if task == 1 else WEIGHTED)
        solve = find_path if task == 1 else minimum_cost
        return solve(tuple(case["start"]), tuple(case["goal"]), case["map"])
    if task == 3:
        return alpha_beta(TreeNode.from_dict(data) if data is not None else sample_tree())
    game = TreasureGame.from_case(data if data is not None else TREASURE)
    return solve_game(game, depth)[0]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--task", type=int, choices=(1, 2, 3, 4), help="只输出一道题的答案")
    parser.add_argument("--input", type=Path, help="单题输入的 JSON 文件")
    parser.add_argument("--depth", type=int, help="题四搜索深度，默认 4")
    args = parser.parse_args()
    if args.input and not args.task:
        parser.error("--input 需要指定 --task")
    if args.depth is not None and (args.task != 4 or args.depth < 0):
        parser.error("--depth 适用于题四，取非负整数")
    if args.task:
        data = json.loads(args.input.read_text(encoding="utf-8-sig")) if args.input else None
        result = task_result(args.task, data, args.depth if args.depth is not None else 4)
        print(json.dumps(result, ensure_ascii=False))
        return
    output = args.output
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path_experiments(output, figures)
    tree_experiments(output, figures)
    matrix = compare_depths(output)
    distance_heatmap(matrix, figures / "fig5_distances.png")
    compare_fixed_orders(output)
    print(f"四题运行完成，结果保存在 {output.resolve()}")


if __name__ == "__main__":
    main()

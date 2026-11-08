"""题目样例和预先固定的拓展场景。所有坐标均为 (行, 列)。"""

from game_search import TreeNode

MAZE = dict(name="task1", start=(0, 0), goal=(3, 4), weighted=False,
            map=[["S", ".", ".", "X", "."], [".", "X", ".", ".", "."],
                 [".", "X", ".", "X", "."], [".", ".", ".", "X", "E"],
                 ["X", "X", ".", ".", "."]])
WEIGHTED = dict(name="task2", start=(0, 0), goal=(4, 4), weighted=True,
                map=[["S", 1, 1, 8, 1], [1, 8, 3, 1, 1], [1, 8, 1, 8, 1],
                     [1, 1, 1, 8, 1], [8, 8, 1, 1, "E"]])
TREASURE = dict(name="task4", start_a=(0, 0), start_b=(6, 6),
                map=[["A", 1, 1, "X", 2, 1, "T1"], [1, "X", 6, "X", 1, "X", 1],
                     [1, 1, "T2", 1, 1, 1, 1], ["X", 1, "X", 8, "X", 1, "X"],
                     [1, 1, 1, 1, "T3", 1, 1], [1, "X", 9, "X", 1, "X", 1],
                     ["T4", 1, 1, "X", 1, 1, "B"]],
                treasures={"T1": dict(position=(0, 6), value=9),
                           "T2": dict(position=(2, 2), value=6),
                           "T3": dict(position=(4, 4), value=7),
                           "T4": dict(position=(6, 0), value=7)})


def sample_tree():
    # 按 PDF 的 JSON 树展开；[6, 9] 出现两次，不按对象身份做缓存。
    def pair(a, b):
        return TreeNode(children=[TreeNode(a), TreeNode(b)])
    return TreeNode(children=[TreeNode(children=[pair(3, 5), pair(6, 9)]),
                              TreeNode(children=[pair(6, 9), pair(1, 2)])])


def small_mazes():
    layouts = {
        "open5": [".....", ".....", "S...E", ".....", "....."],
        "dual5": [".....", ".XXX.", "S.X.E", ".....", "....."],
    }
    return [dict(name=name, start=(2, 0), goal=(2, 4), weighted=False,
                 map=[list(row) for row in rows]) for name, rows in layouts.items()]

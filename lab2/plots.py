"""根据搜索轨迹和距离矩阵绘制报告图 1～5。"""
from functools import lru_cache
from math import hypot
from PIL import Image, ImageDraw, ImageFont

BLUE, GREEN = "#276DA8", "#287F64"


def font(size):
    for name in ["Arial.ttf", "DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


@lru_cache(None)
def map_font(size):
    # 找不到中文字体时使用英文标注，避免其他平台出现方框字。
    for name in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                 "/System/Library/Fonts/PingFang.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        try:
            return ImageFont.truetype(name, size), True
        except OSError:
            pass
    return font(size), False


def expansion_color(step, maximum):
    light, dark = (231, 243, 252), (35, 93, 150)
    fraction = (step - 1) / max(maximum - 1, 1)
    return tuple(round(a + (b-a)*fraction) for a, b in zip(dark, light))


def path_arrow(draw, start, end, cell):
    """每一步在两个格心之间画有白色衬底的方向箭头。"""
    dx, dy = end[0]-start[0], end[1]-start[1]
    length = hypot(dx, dy)
    if not length:
        return
    ux, uy = dx/length, dy/length
    width = max(3, round(cell*.045))
    color = "#DC5126"
    draw.line([start, end], fill="white", width=width+4)
    draw.line([start, end], fill=color, width=width)
    tip = (start[0]+dx*.71, start[1]+dy*.71)
    back = (tip[0]-ux*cell*.17, tip[1]-uy*cell*.17)
    wing = cell*.075
    draw.polygon([tip, (back[0]-uy*wing, back[1]+ux*wing),
                  (back[0]+uy*wing, back[1]-ux*wing)], fill=color)


def maze_figure(cases, details, maximum, destination):
    cell = 68
    image = Image.new("RGB", (810 * len(cases), 470), "white")
    draw = ImageDraw.Draw(image)
    chinese = map_font(28)[1]

    def label(xy, cn, en, size=30, color="#243746"):
        draw.text(xy, cn if chinese else en, font=map_font(size)[0], fill=color, anchor="mm")

    titles = {"task1": ("给定迷宫", "Given maze"), "open5": ("空旷直达", "Open maze"),
              "dual5": ("双路绕行", "Two detours")}
    for col, case in enumerate(cases):
        label((405 + 810*col, 15), *titles[case["name"]], color="#23374A")
        for row, h in enumerate(("zero", "manhattan")):
            panel = 2*col + row
            tag = "(" + chr(ord("a") + panel) + ")  "
            label((203 + 405*panel, 53), tag + ("零启发", "曼哈顿")[row],
                  tag + ("Zero", "Manhattan")[row], 33, "#23374A")
            x, y = 47 + 405*panel, 112
            for index in range(5):
                digit = str(index)
                label((x+(index+.5)*cell, y-17), digit, digit, 24)
                label((x-19, y+(index+.5)*cell), digit, digit, 24)
            result = details[f"{case['name']}/{h}"]
            order = {}
            for i, position in enumerate(result["expanded_order"], 1):
                order.setdefault(tuple(position), i)
            for r, values in enumerate(case["map"]):
                for c, token in enumerate(values):
                    left, top = x+c*cell, y+r*cell
                    fill = expansion_color(order[r,c], maximum) if (r,c) in order else "white"
                    draw.rectangle((left, top, left+cell, top+cell), fill=fill, outline="#92A0AE", width=2)
                    if token == "X":
                        label((left+cell/2, top+cell/2), "×", "×", 50, "#586370")
            points = [(x+(c+.5)*cell, y+(r+.5)*cell) for r,c in result["path"]]
            for a, b in zip(points, points[1:]):
                path_arrow(draw, a, b, cell)
            for tag, point, ink in [("S", points[0], GREEN), ("E", points[-1], "#A93030")]:
                cx, cy = point
                draw.rounded_rectangle((cx-17,cy-18,cx+17,cy+18),4,fill="white",outline=ink,width=2)
                label(point, tag, tag, 27, ink)
    image.save(destination, dpi=(240, 240))


def weighted_panel(case, detail, scale_max):
    image = Image.new("RGB", (820, 910), "white")
    draw = ImageDraw.Draw(image)
    chinese = map_font(24)[1]

    def label(xy, cn, en=None, size=26, fill="#202830", anchor=None):
        draw.text(xy, cn if chinese or en is None else en,
                  font=map_font(size)[0], fill=fill, anchor=anchor)

    cells = case["map"]
    rows, cols = len(cells), len(cells[0])
    cell = min(760//cols, 690//rows)
    x, y = 80+(760-cell*cols)//2, 205+(690-cell*rows)//2
    order = {}
    for step, position in enumerate(detail["expanded_order"], 1):
        order.setdefault(tuple(position), step)
    for r in range(rows):
        label((x-22, y+(r+.5)*cell), str(r), size=21, anchor="rm")
    for c in range(cols):
        label((x+(c+.5)*cell, y-25), str(c), size=21, anchor="mm")
    for r in range(rows):
        for c in range(cols):
            position, token = (r, c), str(cells[r][c])
            left, top = x+c*cell, y+r*cell
            if token == "X":
                fill = "#EDF0F2"
            else:
                fill = expansion_color(order[position], scale_max) if position in order else "white"
            draw.rectangle((left, top, left+cell, top+cell), fill=fill, outline="#98A9B8", width=1)
            if token == "X":
                pad = cell*.33
                draw.line((left+pad, top+pad, left+cell-pad, top+cell-pad), fill="#56616D", width=max(3, cell//30))
                draw.line((left+pad, top+cell-pad, left+cell-pad, top+pad), fill="#56616D", width=max(3, cell//30))
            elif case["weighted"] and token.isdigit():
                ink = "white" if position in order and (order[position]-1)/max(scale_max-1, 1) < .4 else "#202830"
                label((left+cell*.20, top+cell*.20), token, size=max(16, min(42, int(cell*.30))), fill=ink, anchor="mm")
    points = [(x+(c+.5)*cell, y+(r+.5)*cell) for r, c in detail["path"]]
    for start, end in zip(points, points[1:]):
        path_arrow(draw, start, end, cell)
    for tag, position, color in [("S", case["start"], GREEN), ("E", case["goal"], "#A93030")]:
        px, py = x+(position[1]+.5)*cell, y+(position[0]+.5)*cell
        if tuple(case["start"]) == tuple(case["goal"]):
            tag = "S/E"
        word = tag + (" 起点" if tag == "S" else " 终点" if tag == "E" else " 起终点")
        if cell < 80:
            word = tag
        size = max(15, min(28, int(cell*.20)))
        content = word if chinese else tag
        box = draw.textbbox((0, 0), content, font=map_font(size)[0])
        half_width = (box[2]-box[0])/2+8
        half_height = size*.7+3
        draw.rounded_rectangle((px-half_width, py-half_height, px+half_width, py+half_height),
                               radius=5, fill="white", outline=color, width=2)
        label((px, py), word, tag, size=size, fill=color, anchor="mm")

    return image.crop((66, 158, 812, 902))


def weighted_figure(case, details, destination):
    image = Image.new("RGB", (1820, 830), "white")
    draw = ImageDraw.Draw(image)
    keys = ("zero", "corrected")
    maximum = max(len(details[f"task2/{h}"]["expanded_order"]) for h in keys)
    chinese = map_font(28)[1]

    def label(x, y, cn, en, size=36):
        draw.text((x, y), cn if chinese else en, font=map_font(size)[0], fill="#23374A", anchor="mm")

    for j, h in enumerate(keys):
        label(387+760*j, 47, ("零启发", "修正启发")[j], ("Zero", "Corrected")[j], 48)
        image.paste(weighted_panel(case, details[f"task2/{h}"], maximum), (12+760*j, 79))
    label(1650, 177, "进入代价", "Entry cost", 38)
    label(1650, 241, "见格内数字", "Cell labels", 32)
    label(1650, 370, "扩展次序", "Visit order", 38)
    for i in range(210):
        color = expansion_color(1 + (maximum-1)*i/209, maximum)
        draw.line((1570, 428+i, 1614, 428+i), fill=color)
    label(1700, 432, "1  早", "1 early", 32)
    label(1700, 638, f"{maximum}  晚", f"{maximum} late", 32)
    image.save(destination, dpi=(240, 240))


def tree_pruning_map(trace, destination):
    """按给定树的实际轨迹标出第一次剪枝前的访问与回传。"""
    from cases import sample_tree

    img = Image.new("RGB", (1500, 660), "white")
    draw = ImageDraw.Draw(img)
    cut = next(e for e in trace if e["event"] == "cut")
    cut_node = tuple(cut["path"])
    skipped = {cut_node+(i,) for i in cut["skipped"]}
    leaves = [tuple(e["path"]) for e in trace[:trace.index(cut)] if e["event"] == "leaf"]
    values = {tuple(e["path"]): e["value"] for e in trace if e["event"] == "update"}
    nodes, positions = {}, {}

    def place(node, path, left, right):
        mid = (left+right)/2
        nodes[path] = node
        positions[path] = (mid, (70, 205, 360, 530)[len(path)])
        for i, child in enumerate(node.children):
            place(child, path+(i,), left if i == 0 else mid, mid if i == 0 else right)

    def label(x, y, string, size, ink="#243642"):
        draw.text((x, y), string, font=map_font(size)[0], fill=ink, anchor="mm")

    place(sample_tree(), (), 30, 1470)
    for path, end in positions.items():
        if path:
            draw.line((positions[path[:-1]], end), fill="#ADBCC5", width=3)

    # 两次读叶之间，先回到公共祖先，再下降至下一个叶子。
    walk = [leaves[0]]
    for leaf in leaves[1:]:
        while leaf[:len(walk[-1])] != walk[-1]:
            walk.append(walk[-1][:-1])
        walk.extend(leaf[:depth] for depth in range(len(walk[-1])+1, len(leaf)+1))
    walk.append(cut_node)
    edges = list(zip(walk, walk[1:]))
    for step, (src, dst) in enumerate(edges, 1):
        x, y = positions[src]
        dx, dy = positions[dst][0]-x, positions[dst][1]-y
        length = hypot(dx, dy)
        ux, uy = dx/length, dy/length
        offset = 22 if (dst, src) in edges else 0
        x, y = x-uy*offset, y+ux*offset
        ink = BLUE if len(dst)>len(src) else "#B95020"
        draw.line((x, y, x+dx, y+dy), fill=ink, width=4)
        tx, ty = x+.70*dx, y+.70*dy
        draw.polygon([(tx,ty), (tx-ux*17-uy*8,ty-uy*17+ux*8),
                      (tx-ux*17+uy*8,ty-uy*17-ux*8)], fill=ink)
        bx, by = x+.43*dx, y+.43*dy
        draw.ellipse((bx-18,by-18,bx+18,by+18), fill="white", outline=ink, width=2)
        label(bx, by, str(step), 24, ink)

    chinese = map_font(25)[1]
    for path, node in nodes.items():
        x, y = positions[path]
        if not node.children:
            fill = "#FBEFED" if path in skipped else "#ECF3F8"
            draw.ellipse((x-34,y-30,x+34,y+30), fill=fill, outline="#68869C", width=2)
            label(x, y, str(node.value), 32)
            if path in skipped:
                for sign in (-1, 1):
                    draw.line((x+12,y-20+sign*13,x+38,y-20-sign*13), fill="#AE2935", width=6)
                label(x, y+51, f"α={cut['alpha']} ≥ β={cut['beta']}", 24, "#992535")
            continue
        is_parent = path == cut_node[:-1]
        half_height = 42 if is_parent else 30
        draw.rounded_rectangle((x-87,y-half_height,x+87,y+half_height), radius=15,
                               fill="#F1F6F9", outline="#4A6C84", width=3)
        if is_parent:
            label(x, y-14, f"MIN ≤ {cut['beta']}", 32)
            label(x, y+17, ('最终' if chinese else 'final') + f" = {values[path]}", 23)
        else:
            role = "MAX" if len(path)%2 == 0 else "MIN"
            label(x, y, f"{role} {'≥' if path == cut_node else '='} {values[path]}", 32)
    legend = ("从左侧叶子 3 起按编号阅读     蓝箭头：访问子节点     橙箭头：返回父节点     红叉：未访问"
              if chinese else "From leaf 3, follow the numbers.   Blue: descend    Orange: return    Red cross: skipped")
    label(750, 623, legend, 25)
    img.save(destination, dpi=(240, 240))


def distance_heatmap(rows, destination):
    labels = [row["source"] for row in rows]
    maximum = max(int(row[label]) for row in rows for label in labels)
    image = Image.new("RGB", (1030, 795), "white")
    draw = ImageDraw.Draw(image)
    chinese = map_font(28)[1]

    def label(x, y, value, size=32, ink="#203547"):
        draw.text((x, y), str(value), font=map_font(size)[0], fill=ink, anchor="mm")

    def color(value):
        ratio = value/max(maximum, 1)
        return tuple(round(a+(b-a)*ratio) for a,b in zip((249,252,255),(19,77,123)))

    x, y, cell = 136, 114, 105
    label(x+3*cell, 31, "目的地" if chinese else "Destination")
    label(49, y+3*cell, "起点" if chinese else "From", 27)
    for j, key in enumerate(labels):
        label(x+(j+.5)*cell, y-34, key)
    for i, row in enumerate(rows):
        label(x-34, y+(i+.5)*cell, row["source"])
        for j, key in enumerate(labels):
            value = int(row[key])
            left, top = x+j*cell, y+i*cell
            draw.rectangle((left,top,left+cell-1,top+cell-1), fill=color(value), outline="#BDCCD7", width=2)
            label(left+cell/2, top+cell/2, value, 39, "white" if value > maximum*.53 else "#17344B")
    label(880, y-38, "移动代价" if chinese else "Cost", 28)
    height=len(rows)*cell
    for offset in range(height):
        draw.line((840,y+offset,876,y+offset),fill=color(maximum*(1-offset/(height-1))))
    for v in sorted({0, maximum, *range(0, maximum, max(1,maximum//6))}):
        label(909,y+(height-1)*(1-v/max(maximum,1)),v,25)
    image.save(destination, dpi=(240,240))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CyberMath: Hệ BPT 2 Ẩn
=======================
Ứng dụng minh họa trực quan, theo từng bước, cách giải hệ bất phương trình
bậc nhất hai ẩn (x, y) bằng phương pháp vẽ miền nghiệm trên mặt phẳng tọa độ.

Công nghệ:
    - Giao diện: CustomTkinter (Dark / Cyberpunk theme)
    - Đồ thị:    Matplotlib nhúng trực tiếp (FigureCanvasTkAgg)
    - Toán học:  NumPy (mặt lưới để tô miền nghiệm chính xác)

Cài đặt:
    pip install customtkinter matplotlib numpy

Chạy:
    python cybermath_app.py
"""

import re
import math
import itertools
from fractions import Fraction
import numpy as np

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon


# =============================================================================
# 1. BẢNG MÀU (THEME & COLOR PALETTE)
# =============================================================================
BG_MAIN        = "#121218"   # Nền ứng dụng chính
BG_SIDEBAR     = "#181820"   # Nền sidebar (hơi sáng hơn nền chính)
BG_CARD        = "#1B1B26"   # Nền các thẻ / card
BG_CANVAS      = "#0D0D11"   # Nền vùng đồ thị
GRID_COLOR     = "#2A2A38"   # Lưới
AXIS_COLOR     = "#71718A"   # Trục tọa độ
TEXT_COLOR     = "#E2E8F0"   # Chữ trắng băng giá
TEXT_MUTED     = "#8A8AA0"
DANGER_COLOR   = "#FF5566"

NEON_COLORS = ["#00F0FF", "#FF007F", "#00FF66", "#FFA53C", "#B15CFF", "#39C7FF"]
SOLUTION_YELLOW = "#FFE600"
MIN_COLOR = "#00FF66"
MAX_COLOR = "#FFA53C"

# Ký hiệu toán học hiển thị cho người dùng (không bao giờ hiện "<=" / ">=")
OP_SYMBOLS = {"<=": "≤", ">=": "≥", "<": "<", ">": ">", "=": "="}
OP_SYMBOL_TO_INTERNAL = {v: k for k, v in OP_SYMBOLS.items()}

XMIN, XMAX = -10, 10          # Khung nhìn (view) mặc định khi khởi động
YMIN, YMAX = -10, 10
DOMAIN_MIN, DOMAIN_MAX = -30, 30  # Miền tính toán rộng hơn để hỗ trợ Zoom/Pan
GRID_RES = 500                 # Độ phân giải lưới numpy dùng để tô miền nghiệm
ZOOM_MIN_SPAN = 0.8             # Không cho phóng to quá gần
ZOOM_MAX_SPAN = 55              # Không cho thu nhỏ vượt quá miền domain
SNAP_TOLERANCE = 0.01           # Khoảng cách (đơn vị trục) để đường F "khớp" vào đỉnh


# =============================================================================
# 2. LOGIC TOÁN HỌC (PARSING & GIẢI HỆ BPT)
# =============================================================================
class ParseError(Exception):
    """Lỗi cú pháp khi người dùng nhập bất phương trình sai định dạng."""
    pass


def _parse_coef(coef_str: str) -> float:
    """Chuyển chuỗi hệ số (có thể là phân số 'a/b') sang float."""
    if "/" in coef_str:
        num_s, _, den_s = coef_str.partition("/")
        num_v = float(num_s) if num_s not in ("", ".") else 1.0
        if den_s in ("", "."):
            raise ParseError(f"Thiếu mẫu số trong phân số '{coef_str}'.")
        den_v = float(den_s)
        if den_v == 0:
            raise ParseError("Mẫu số của phân số không được bằng 0.")
        return num_v / den_v
    return float(coef_str)


def _parse_linear_expr(expr: str) -> dict:
    """Phân tích một biểu thức bậc nhất dạng 'ax + by + c' thành dict hệ số."""
    expr = expr.replace(" ", "")
    if expr == "":
        return {"x": 0.0, "y": 0.0, "c": 0.0}
    if expr[0] not in "+-":
        expr = "+" + expr
    terms = re.findall(r"[+-][^+-]+", expr)
    if not terms or "".join(terms) != expr:
        raise ParseError(f"Biểu thức không hợp lệ: '{expr}'")

    result = {"x": 0.0, "y": 0.0, "c": 0.0}
    for term in terms:
        sign = -1.0 if term[0] == "-" else 1.0
        body = term[1:]
        if body == "":
            raise ParseError(f"Số hạng rỗng gần: '{term}'")
        # Tách biến (x hoặc y) khỏi phần hệ số, cho phép phân số ở bất kỳ vị
        # trí nào: '1/2x', 'x/2', '2x/3', '3/4' đều hợp lệ.
        var_match = re.search(r"[xy]", body)
        if var_match:
            var = var_match.group()
            coef_str = body[:var_match.start()] + body[var_match.end():]
        else:
            var = ""
            coef_str = body
        if not re.match(r"^\d*\.?\d*(?:/\d*\.?\d*)?$", coef_str):
            raise ParseError(f"Không thể phân tích số hạng: '{term}'")
        if coef_str == "" and var == "":
            raise ParseError(f"Số hạng không xác định: '{term}'")
        coef = _parse_coef(coef_str) if coef_str else 1.0
        if var == "x":
            result["x"] += sign * coef
        elif var == "y":
            result["y"] += sign * coef
        else:
            result["c"] += sign * coef
    return result


def parse_inequality(text: str) -> dict:
    """
    Chuyển một chuỗi người dùng nhập (vd: '2x + y <= 3') thành dạng chuẩn:
        a*x + b*y  [op]  c
    Trả về dict {'a', 'b', 'c', 'op', 'raw'}.
    """
    raw = text.strip()
    if not raw:
        raise ParseError("Vui lòng nhập một bất phương trình.")

    m = re.search(r"(<=|>=|<|>|=)", raw)
    if not m:
        raise ParseError("Thiếu dấu so sánh (<=, >=, <, >, =).")

    op = m.group(1)
    left_str, right_str = raw[:m.start()], raw[m.end():]

    try:
        left = _parse_linear_expr(left_str)
        right = _parse_linear_expr(right_str)
    except ParseError:
        raise
    except Exception as exc:  # bảo vệ chống crash với input lạ
        raise ParseError(f"Cú pháp không hợp lệ: {exc}")

    a = left["x"] - right["x"]
    b = left["y"] - right["y"]
    c = right["c"] - left["c"]

    if abs(a) < 1e-9 and abs(b) < 1e-9:
        raise ParseError("Bất phương trình phải chứa biến x hoặc y.")

    return {"a": a, "b": b, "c": c, "op": op, "raw": raw}


def format_linear(a: float, b: float, c: float, op: str = "=") -> str:
    """Định dạng 'ax + by [op] c' cho đẹp mắt, bỏ hệ số 0/1 thừa.

    `op` mặc định là dấu "=" (dùng để hiển thị PHƯƠNG TRÌNH đường biên), nhưng
    có thể truyền vào một ký hiệu toán học khác (≤, ≥, <, >) để hiển thị
    chính BẤT PHƯƠNG TRÌNH — không bao giờ dùng chuỗi kiểu "<=" / ">=".
    """
    def fmt_term(coef, var):
        if abs(coef) < 1e-9:
            return None
        r = round(coef, 4)
        if r == int(r):
            r = int(r)
        if r == 1:
            return var
        if r == -1:
            return f"-{var}"
        return f"{r}{var}"

    tx, ty = fmt_term(a, "x"), fmt_term(b, "y")
    parts = []
    if tx:
        parts.append(tx)
    if ty:
        if parts:
            parts.append(("+ " + ty) if not ty.startswith("-") else ("- " + ty[1:]))
        else:
            parts.append(ty)
    if not parts:
        parts = ["0"]
    cr = round(c, 4)
    if cr == int(cr):
        cr = int(cr)
    return f"{' '.join(parts)} {op} {cr}"


def num(v):
    """Làm gọn số để hiển thị (bỏ .0 nếu là số nguyên)."""
    v = round(v, 4)
    return int(v) if v == int(v) else v


def format_frac(v, max_denominator=1000):
    """Định dạng một số để hiển thị: trả về số nguyên nếu v là số nguyên,
    ngược lại trả về dạng PHÂN SỐ tối giản (vd: '3/2', '-7/3') thay vì số
    thập phân dài dòng — giúp tọa độ đỉnh dễ đọc và chính xác hơn."""
    r = round(v, 6)
    if abs(r - round(r)) < 1e-6:
        return str(int(round(r)))
    frac = Fraction(r).limit_denominator(max_denominator)
    if frac.denominator == 1:
        return str(frac.numerator)
    sign = "-" if frac.numerator < 0 else ""
    return f"{sign}{abs(frac.numerator)}/{frac.denominator}"


def is_satisfied(ineq: dict, x, y):
    """Kiểm tra điểm/mảng (x, y) có thỏa mãn bất phương trình không (vector hóa)."""
    val = ineq["a"] * x + ineq["b"] * y
    op = ineq["op"]
    if op == "<=":
        return val <= ineq["c"] + 1e-9
    if op == ">=":
        return val >= ineq["c"] - 1e-9
    if op == "<":
        return val < ineq["c"] - 1e-9
    if op == ">":
        return val > ineq["c"] + 1e-9
    return np.abs(val - ineq["c"]) < 1e-6  # '='


def choose_test_point(ineq: dict):
    """Chọn điểm thử, ưu tiên gốc tọa độ O(0,0) nếu nó không nằm trên đường thẳng."""
    for (x, y) in [(0, 0), (1, 0), (0, 1), (2, 3), (1, 1), (-1, 2)]:
        if abs(ineq["a"] * x + ineq["b"] * y - ineq["c"]) > 1e-6:
            return x, y
    return 5, 7


def clip_polygon_halfplane(poly, a, b, c, keep):
    """Cắt một đa giác LỒI `poly` (danh sách đỉnh (x, y)) bằng nửa mặt
    phẳng a*x + b*y <= c (nếu keep='le') hoặc a*x + b*y >= c (nếu
    keep='ge'), dùng thuật toán Sutherland–Hodgman.

    Vì đây là phép cắt HÌNH HỌC bằng giao điểm thực sự (không dựa vào
    lưới numpy rời rạc), biên kết quả luôn là các đoạn thẳng chính xác,
    áp sát tuyệt đối vào đường biên thật — không bị răng cưa/bậc thang
    như khi tô màu bằng contourf trên lưới.
    """
    if not poly:
        return []

    def inside(p):
        val = a * p[0] + b * p[1]
        return val <= c + 1e-9 if keep == "le" else val >= c - 1e-9

    def intersect(p1, p2):
        v1 = a * p1[0] + b * p1[1] - c
        v2 = a * p2[0] + b * p2[1] - c
        if abs(v1 - v2) < 1e-12:
            return p2
        t = v1 / (v1 - v2)
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    output = []
    n = len(poly)
    for i in range(n):
        curr, prev = poly[i], poly[i - 1]
        curr_in, prev_in = inside(curr), inside(prev)
        if curr_in:
            if not prev_in:
                output.append(intersect(prev, curr))
            output.append(curr)
        elif prev_in:
            output.append(intersect(prev, curr))
    return output


def halfplane_polygon(a, b, c, keep, bound=None):
    """Trả về đa giác là phần giao giữa hình chữ nhật miền tính toán
    (mặc định [DOMAIN_MIN, DOMAIN_MAX]^2, hoặc `bound` nếu truyền vào)
    với nửa mặt phẳng a*x + b*y <= c (keep='le') hay >= c (keep='ge')."""
    rect = bound or [(DOMAIN_MIN, DOMAIN_MIN), (DOMAIN_MAX, DOMAIN_MIN),
                      (DOMAIN_MAX, DOMAIN_MAX), (DOMAIN_MIN, DOMAIN_MAX)]
    return clip_polygon_halfplane(rect, a, b, c, keep)


def line_points(ineq: dict, xlim=(DOMAIN_MIN, DOMAIN_MAX), ylim=(DOMAIN_MIN, DOMAIN_MAX)):
    """Trả về 2 điểm để vẽ đường thẳng biên trong phạm vi khung nhìn."""
    a, b, c = ineq["a"], ineq["b"], ineq["c"]
    x0, x1 = xlim
    y0, y1 = ylim
    if abs(b) > 1e-9:
        return [(x0, (c - a * x0) / b), (x1, (c - a * x1) / b)]
    xv = c / a
    return [(xv, y0), (xv, y1)]


def compute_polygon_vertices(inequalities, tol=1e-6):
    """Tìm các đỉnh của đa giác miền nghiệm cuối cùng.

    Cách làm: lấy giao điểm của mọi cặp đường biên, giữ lại những giao điểm
    thỏa mãn ĐỒNG THỜI toàn bộ hệ bất phương trình (đang hiển thị), sau đó
    sắp xếp chúng theo góc quanh trọng tâm để tạo thành một đa giác lồi khép
    kín đúng thứ tự (dùng để vẽ/nối các đỉnh theo đúng biên).
    """
    active = [q for q in inequalities if q.get("visible", True)]
    pts = []
    for i, j in itertools.combinations(range(len(active)), 2):
        a1, b1, c1 = active[i]["a"], active[i]["b"], active[i]["c"]
        a2, b2, c2 = active[j]["a"], active[j]["b"], active[j]["c"]
        det = a1 * b2 - a2 * b1
        if abs(det) < tol:
            continue  # hai đường song song hoặc trùng nhau -> bỏ qua
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det

        ok = True
        for q in active:
            val = q["a"] * x + q["b"] * y
            op = q["op"]
            if op == "<=" and val > q["c"] + 1e-6:
                ok = False
            elif op == ">=" and val < q["c"] - 1e-6:
                ok = False
            elif op == "<" and val >= q["c"] - 1e-6:
                ok = False
            elif op == ">" and val <= q["c"] + 1e-6:
                ok = False
            elif op == "=" and abs(val - q["c"]) > 1e-4:
                ok = False
            if not ok:
                break
        if ok:
            pts.append((x, y))

    # Loại các điểm trùng nhau do sai số dấu phẩy động
    uniq = []
    for p in pts:
        if not any(abs(p[0] - u[0]) < 1e-6 and abs(p[1] - u[1]) < 1e-6 for u in uniq):
            uniq.append(p)

    if len(uniq) < 3:
        return uniq  # có thể là 0, 1 hoặc 2 đỉnh (miền rỗng / nửa đường thẳng)

    cx = sum(p[0] for p in uniq) / len(uniq)
    cy = sum(p[1] for p in uniq) / len(uniq)
    uniq.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return uniq


def evaluate_objective_at_vertices(vertices, a, b):
    """Tính F = a*x + b*y tại từng đỉnh, trả về danh sách
    [{'x', 'y', 'F'}] và (vertex_min, vertex_max) kèm giá trị F tương ứng."""
    rows = [{"x": vx, "y": vy, "F": a * vx + b * vy} for (vx, vy) in vertices]
    if not rows:
        return rows, None, None
    row_min = min(rows, key=lambda r: r["F"])
    row_max = max(rows, key=lambda r: r["F"])
    return rows, row_min, row_max


class StepEngine:
    """Sinh ra danh sách các bước minh họa (line -> test -> ... -> conclusion)."""

    @staticmethod
    def generate(inequalities, has_objective=False):
        steps = []
        for i, ineq in enumerate(inequalities):
            steps.append({"type": "line", "idx": i})
            steps.append({"type": "test", "idx": i})
        if inequalities:
            steps.append({"type": "conclusion"})
            # Nếu người dùng đã bật đường mức F, thêm bước cuối cùng để
            # tính F tại từng đỉnh của đa giác nghiệm và chốt min/max.
            if has_objective:
                steps.append({"type": "optimize"})
        return steps

    @staticmethod
    def explain(step, inequalities, step_no, total, objective=None):
        header = f"BƯỚC {step_no}/{total}\n" + "─" * 26 + "\n"
        if step["type"] == "line":
            i = step["idx"]
            ineq = inequalities[i]
            strict = ineq["op"] in ("<", ">")
            style = "nét đứt (không lấy dấu bằng)" if strict else "nét liền (có dấu bằng)"
            return (header +
                    f"Vẽ đường thẳng biên d{i + 1} của bất phương trình "
                    f"({ineq['raw']}):\n\n"
                    f"    d{i + 1}:  {format_linear(ineq['a'], ineq['b'], ineq['c'])}\n\n"
                    f"Đường được vẽ bằng {style}, màu neon đại diện cho BPT {i + 1}.")
        if step["type"] == "test":
            i = step["idx"]
            ineq = inequalities[i]
            tx, ty = choose_test_point(ineq)
            val = ineq["a"] * tx + ineq["b"] * ty
            ok = is_satisfied(ineq, tx, ty)
            side = "CHỨA điểm thử" if ok else "KHÔNG chứa điểm thử"
            return (header +
                    f"Chọn điểm thử M({num(tx)}; {num(ty)}) (không nằm trên d{i + 1}).\n\n"
                    f"Thay vào vế trái:\n"
                    f"    {num(ineq['a'])}×{num(tx)} + {num(ineq['b'])}×{num(ty)} = {num(val)}\n\n"
                    f"So sánh: {num(val)} {OP_SYMBOLS[ineq['op']]} {num(ineq['c'])}  →  "
                    f"{'ĐÚNG ✓' if ok else 'SAI ✗'}\n\n"
                    f"=> Miền nghiệm của BPT {i + 1} là nửa mặt phẳng {side}. "
                    f"Nửa mặt phẳng được CHỌN sẽ sáng màu hơn (ánh màu neon riêng), "
                    f"nửa còn lại bị loại bỏ và phủ ĐEN VĨNH VIỄN (không mờ dần, "
                    f"không đổi khi sang bước sau).")
        if step["type"] == "optimize":
            return StepEngine._explain_optimize(header, inequalities, objective)
        # conclusion
        return (header +
                "KẾT LUẬN\n\n"
                "Miền nghiệm của cả hệ là phần GIAO NHAU của tất cả các miền nghiệm "
                "riêng lẻ. Vùng này được làm sáng rực rỡ bằng màu VÀNG NEON trên đồ "
                "thị, khớp chính xác với đường biên của đa giác — đây chính là tập "
                "hợp mọi điểm (x, y) thỏa mãn ĐỒNG THỜI toàn bộ hệ bất phương trình. "
                "Bấm \"📐 Hiện đỉnh\" để xem tọa độ các đỉnh của đa giác nghiệm này.")

    @staticmethod
    def _explain_optimize(header, inequalities, objective):
        """Sinh nội dung giải thích bước tối ưu: bảng F tại từng đỉnh + kết
        luận min/max bằng lời."""
        if not objective:
            return (header +
                    "TỐI ƯU HÀM MỤC TIÊU F = ax + by\n\n"
                    "Chưa có hàm mục tiêu nào được áp dụng. Hãy nhập hệ số a, b rồi "
                    "bấm \"📈 Hiện đường mức F\" ở mục 4 trong sidebar.")

        vertices = compute_polygon_vertices(inequalities)
        a, b = objective["a"], objective["b"]
        obj_str = format_linear(a, b, 0, "=").rsplit(" = ", 1)[0]

        if len(vertices) < 3:
            return (header +
                    f"TỐI ƯU HÀM MỤC TIÊU F = {obj_str}\n\n"
                    "Miền nghiệm KHÔNG BỊ CHẶN (không phải một đa giác kín), nên không "
                    "thể liệt kê đầy đủ các đỉnh để so sánh. Tùy theo hướng của F, giá "
                    "trị F có thể tiến ra vô cực (không tồn tại min hoặc max hữu hạn "
                    "trên toàn miền).")

        rows, row_min, row_max = evaluate_objective_at_vertices(vertices, a, b)

        # --- Bảng F tại từng đỉnh (canh cột bằng font Consolas) ---
        table_lines = [f"{'Đỉnh (x; y)':<16}{'F = ' + obj_str:<14}"]
        table_lines.append("-" * 30)
        for r in rows:
            point_str = f"({format_frac(r['x'])}; {format_frac(r['y'])})"
            mark = ""
            if r is row_min:
                mark = " ← min"
            if r is row_max:
                mark = " ← max" if mark == "" else mark + " / max"
            table_lines.append(f"{point_str:<16}{str(num(r['F'])):<8}{mark}")
        table_str = "\n".join(table_lines)

        min_point = f"({format_frac(row_min['x'])}; {format_frac(row_min['y'])})"
        max_point = f"({format_frac(row_max['x'])}; {format_frac(row_max['y'])})"

        narrative = (
            f"Vì miền nghiệm là một đa giác LỒI và BỊ CHẶN, giá trị nhỏ nhất và lớn "
            f"nhất của F = {obj_str} (nếu có) chỉ có thể đạt được TẠI MỘT TRONG CÁC "
            f"ĐỈNH của đa giác — không cần kiểm tra các điểm khác bên trong miền.\n\n"
            f"Lần lượt thay tọa độ từng đỉnh vào F rồi so sánh các kết quả:\n\n"
            f"{table_str}\n\n"
            f"So sánh {len(rows)} giá trị F ở trên:\n"
            f"  • F NHỎ NHẤT (min) = {num(row_min['F'])}  tại đỉnh {min_point}\n"
            f"  • F LỚN NHẤT  (max) = {num(row_max['F'])}  tại đỉnh {max_point}\n\n"
            f"=> Kết luận: min F = {num(row_min['F'])} tại {min_point}; "
            f"max F = {num(row_max['F'])} tại {max_point}."
        )

        return header + f"TỐI ƯU HÀM MỤC TIÊU F = {obj_str}\n\n" + narrative


# =============================================================================
# 3. VÙNG ĐỒ THỊ (MathCanvas)
# =============================================================================
class MathCanvas:
    """Bọc một Figure/Axes Matplotlib nhúng vào CTk, chịu trách nhiệm vẽ.

    Hỗ trợ tương tác chuột:
      - Cuộn (scroll) để phóng to / thu nhỏ, lấy vị trí con trỏ làm tâm.
      - Giữ chuột trái và kéo để di chuyển (pan) khung nhìn.
    """

    def __init__(self, parent, on_mouse_move=None):
        self.figure = Figure(figsize=(7.5, 6.2), dpi=100, facecolor=BG_CANVAS)
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.08, right=0.97, top=0.96, bottom=0.08)

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.widget = self.canvas.get_tk_widget()
        self.widget.configure(bg=BG_CANVAS, highlightthickness=0)
        self.widget.pack(fill="both", expand=True)

        # Lưới tính toán rộng (dùng chung cho mọi lần render, không phụ
        # thuộc khung nhìn hiện tại) để việc zoom/pan không cần tính lại.
        xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, GRID_RES)
        ys = np.linspace(DOMAIN_MIN, DOMAIN_MAX, GRID_RES)
        self.X, self.Y = np.meshgrid(xs, ys)

        # Khung nhìn hiện tại (thay đổi khi người dùng zoom/pan)
        self.view_xlim = (XMIN, XMAX)
        self.view_ylim = (YMIN, YMAX)
        self._pan_active = False
        self._pan_start_pixel = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None

        self._last_render_args = None  # (step_idx, inequalities, steps, show_vertices)

        if on_mouse_move:
            self.canvas.mpl_connect("motion_notify_event", on_mouse_move)

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_drag)

        self._style_axes()
        self.canvas.draw()

    # ------------------------------------------------------ VẼ NỀN ĐỒ THỊ
    def _style_axes(self):
        ax = self.ax
        ax.clear()
        ax.set_facecolor(BG_CANVAS)
        ax.set_xlim(self.view_xlim)
        ax.set_ylim(self.view_ylim)
        ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.9, zorder=0)
        ax.axhline(0, color=AXIS_COLOR, linewidth=1.3, zorder=2)
        ax.axvline(0, color=AXIS_COLOR, linewidth=1.3, zorder=2)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        # Ẩn số ở viền ngoài (border) và thay bằng số nhỏ ngay sát 2 đường
        # trục x=0 / y=0 -> dễ đọc hơn khi zoom/pan vì số luôn đi theo trục.
        ax.tick_params(colors=TEXT_COLOR, labelsize=9,
                       labelbottom=False, labelleft=False)
        ax.set_xlabel("x", color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel("y", color=TEXT_COLOR, fontsize=10)
        ax.set_title("Miền nghiệm hệ bất phương trình", color=TEXT_COLOR,
                     fontsize=11, pad=10)
        self._draw_axis_numbers()

    def _draw_axis_numbers(self):
        """Vẽ các số nhỏ ngay trên 2 đường trục x=0 và y=0 (thay cho số ở
        viền ngoài đồ thị), giúp đọc tọa độ trực quan như một mặt phẳng
        Oxy thật, bất kể đang zoom / pan tới đâu."""
        ax = self.ax
        x0, x1 = self.view_xlim
        y0, y1 = self.view_ylim

        for t in ax.get_xticks():
            if not (x0 < t < x1) or abs(t) < 1e-9:
                continue
            ax.annotate(str(num(t)), xy=(t, 0), xytext=(0, -3),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=7, color=AXIS_COLOR, zorder=2.2)

        for t in ax.get_yticks():
            if not (y0 < t < y1) or abs(t) < 1e-9:
                continue
            ax.annotate(str(num(t)), xy=(0, t), xytext=(-4, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=7, color=AXIS_COLOR, zorder=2.2)

        if x0 < 0 < x1 and y0 < 0 < y1:
            ax.annotate("0", xy=(0, 0), xytext=(-4, -3),
                        textcoords="offset points", ha="right", va="top",
                        fontsize=7, color=AXIS_COLOR, zorder=2.2)

    def render(self, step_idx, inequalities, steps, show_vertices=False, objective=None,
               highlight_optimum=False, snap_vertex=None):
        """Vẽ lại toàn bộ đồ thị dựa trên tiến độ tích lũy tới step_idx.

        `highlight_optimum`: khi True (bước "optimize"), khoanh tròn đỉnh
        đạt min bằng màu xanh lá và đỉnh đạt max bằng màu cam ngay trên đồ
        thị, đồng bộ với bảng/giải thích bên panel phải.

        `snap_vertex`: nếu khác None, là tọa độ (x, y) của đỉnh mà đường
        mức F vừa "khớp" vào (do người dùng kéo thanh trượt tới rất gần
        đỉnh đó, trong phạm vi SNAP_TOLERANCE đơn vị trục) -> vẽ một vòng
        highlight nổi bật quanh đỉnh này để xác nhận trực quan cho người
        dùng rằng đường F đang đi ĐÚNG QUA đỉnh, không phải xấp xỉ.
        """
        self._last_render_args = (step_idx, inequalities, steps, show_vertices, objective,
                                   highlight_optimum, snap_vertex)
        self._style_axes()
        ax = self.ax
        n = len(inequalities)
        final_step = 2 * n

        # Danh sách các đa giác (nửa mặt phẳng bị loại bỏ) tính tới step_idx.
        # Mỗi đa giác được cắt CHÍNH XÁC bằng hình học (Sutherland-Hodgman),
        # không qua lưới numpy -> biên luôn là đường thẳng áp sát tuyệt đối
        # vào đường biên thật, không còn răng cưa/bậc thang. Tô ĐEN từng
        # đa giác này (chồng lên nhau nếu có) để phần bị loại luôn hiển thị
        # đậm, ổn định và không đổi khi chuyển sang bước sau.
        excluded_polys = []

        for i, ineq in enumerate(inequalities):
            if not ineq.get("visible", True):
                continue
            color = ineq["color"]
            line_step, shade_step = 2 * i, 2 * i + 1
            op = ineq["op"]
            if op in ("<=", "<"):
                keep_side, other_side = "le", "ge"
            elif op in (">=", ">"):
                keep_side, other_side = "ge", "le"
            else:
                keep_side = other_side = None  # '=' : chỉ vẽ đường biên, không tô miền

            if step_idx >= line_step:
                pts = line_points(ineq)
                xs_l = [p[0] for p in pts]
                ys_l = [p[1] for p in pts]
                strict = ineq["op"] in ("<", ">")
                ls = "--" if strict else "-"
                is_current = (step_idx == line_step)
                if is_current:
                    ax.plot(xs_l, ys_l, color=color, linewidth=7, alpha=0.22, zorder=3)
                ax.plot(xs_l, ys_l, color=color, linewidth=2.6 if is_current else 1.8,
                         linestyle=ls, alpha=1.0 if is_current else 0.8, zorder=3.5,
                         label=f"d{i + 1}: {ineq['raw']}")

            if step_idx >= shade_step:
                if keep_side:
                    sat_poly = halfplane_polygon(ineq["a"], ineq["b"], ineq["c"], keep_side)
                    exc_poly = halfplane_polygon(ineq["a"], ineq["b"], ineq["c"], other_side)
                    if exc_poly and len(exc_poly) >= 3:
                        excluded_polys.append(exc_poly)

                    # Từ bước KẾT LUẬN trở đi (kể cả bước tối ưu phía sau),
                    # không tô riêng màu của từng BPT nữa: nếu tô, các lớp
                    # màu neon của n bất phương trình sẽ chồng alpha lên
                    # nhau bên trong miền giao, khiến lớp vàng cuối cùng bị
                    # pha trộn và lệch màu so với đường biên thật. Ta chỉ
                    # giữ lại đường biên + phần loại bỏ màu đen, còn màu
                    # vàng của miền nghiệm được tô riêng, sạch, ở khối dưới.
                    if step_idx < final_step and sat_poly and len(sat_poly) >= 3:
                        ax.add_patch(Polygon(sat_poly, closed=True, facecolor=color,
                                              edgecolor="none", alpha=0.16, zorder=1))

                if step_idx == shade_step:
                    tx, ty = choose_test_point(ineq)
                    ax.scatter([tx], [ty], s=500, color=color, alpha=0.12, zorder=5)
                    ax.scatter([tx], [ty], s=180, color=color, alpha=0.30, zorder=5)
                    ax.scatter([tx], [ty], s=45, color=color, alpha=1.0, zorder=5.2)
                    ax.annotate(f"M({num(tx)};{num(ty)})", (tx, ty),
                                textcoords="offset points", xytext=(8, 8),
                                color=TEXT_COLOR, fontsize=8.5, zorder=5.2)

        for exc_poly in excluded_polys:
            # Alpha < 1 để phần bị loại luôn hiện ĐEN rõ, đậm, dứt khoát;
            # nơi nhiều BPT cùng loại trừ một vùng sẽ tự nhiên đậm hơn do
            # các lớp chồng lên nhau — biên vẫn luôn là đường thẳng chính xác.
            ax.add_patch(Polygon(exc_poly, closed=True, facecolor="#000000",
                                  edgecolor="none", alpha=0.72, zorder=1.5))

        vertices = []
        # Từ bước KẾT LUẬN trở đi (>= final_step) -> vẫn còn đúng ở bước tối
        # ưu phía sau (final_step + 1), nên polygon/đỉnh luôn được vẽ lại.
        if n and step_idx >= final_step:
            visible_ineqs = [q for q in inequalities if q.get("visible", True)]
            if visible_ineqs:
                vertices = compute_polygon_vertices(inequalities)

                if len(vertices) >= 3:
                    # Miền nghiệm bị chặn (đa giác kín): tô và vẽ viền bằng
                    # chính TỌA ĐỘ CÁC ĐỈNH -> khớp tuyệt đối với các cạnh/
                    # đường biên thật, không phụ thuộc độ phân giải lưới.
                    fill_poly = Polygon(vertices, closed=True,
                                         facecolor=SOLUTION_YELLOW, edgecolor="none",
                                         alpha=0.5, zorder=2.5)
                    edge_poly = Polygon(vertices, closed=True,
                                         facecolor="none", edgecolor=SOLUTION_YELLOW,
                                         linewidth=2.4, zorder=3.6)
                    ax.add_patch(fill_poly)
                    ax.add_patch(edge_poly)
                else:
                    # Miền nghiệm không bị chặn (vô hạn, vd chỉ x>=0, y>=0):
                    # không thể vẽ đa giác kín 3+ đỉnh -> lấy GIAO của tất cả
                    # nửa mặt phẳng bằng cách CẮT tuần tự (Sutherland-Hodgman)
                    # hình chữ nhật miền tính toán qua từng bất phương trình.
                    # Kết quả là một đa giác CHÍNH XÁC (được xén gọn theo
                    # khung domain để hiển thị), biên luôn áp sát đường thẳng
                    # thật, không còn phụ thuộc độ phân giải lưới.
                    region_poly = [(DOMAIN_MIN, DOMAIN_MIN), (DOMAIN_MAX, DOMAIN_MIN),
                                   (DOMAIN_MAX, DOMAIN_MAX), (DOMAIN_MIN, DOMAIN_MAX)]
                    for ineq2 in visible_ineqs:
                        op2 = ineq2["op"]
                        if op2 in ("<=", "<"):
                            region_poly = clip_polygon_halfplane(
                                region_poly, ineq2["a"], ineq2["b"], ineq2["c"], "le")
                        elif op2 in (">=", ">"):
                            region_poly = clip_polygon_halfplane(
                                region_poly, ineq2["a"], ineq2["b"], ineq2["c"], "ge")
                        if len(region_poly) < 3:
                            break
                    if len(region_poly) >= 3:
                        ax.add_patch(Polygon(region_poly, closed=True,
                                              facecolor=SOLUTION_YELLOW,
                                              edgecolor=SOLUTION_YELLOW, linewidth=2.4,
                                              alpha=0.5, zorder=2.5))

                if show_vertices:
                    # Tâm xấp xỉ của đa giác nghiệm, dùng để xác định hướng
                    # "ra ngoài" cho mỗi đỉnh -> nhãn tọa độ luôn nằm ngoài
                    # vùng tô màu, không đè lên màu vàng của miền nghiệm.
                    if len(vertices) >= 3:
                        cxg = sum(p[0] for p in vertices) / len(vertices)
                        cyg = sum(p[1] for p in vertices) / len(vertices)
                    else:
                        cxg = cyg = None

                    for (vx, vy) in vertices:
                        ax.scatter([vx], [vy], s=95, color=SOLUTION_YELLOW,
                                   edgecolors=BG_MAIN, linewidths=1.4, zorder=6)
                        label = f"({format_frac(vx)}; {format_frac(vy)})"

                        if cxg is not None:
                            dx, dy = vx - cxg, vy - cyg
                            dist = math.hypot(dx, dy)
                            if dist < 1e-6:
                                dx, dy, dist = 1.0, 1.0, math.sqrt(2)
                            ux, uy = dx / dist, dy / dist
                            offset_px = 20
                            xytext = (ux * offset_px, uy * offset_px)
                            ha = "left" if ux >= 0 else "right"
                            va = "bottom" if uy >= 0 else "top"
                        else:
                            xytext = (10, 10)
                            ha, va = "left", "bottom"

                        ax.annotate(label, (vx, vy),
                                    textcoords="offset points", xytext=xytext,
                                    ha=ha, va=va,
                                    color=SOLUTION_YELLOW, fontsize=8.5,
                                    fontweight="bold", zorder=6)

        # --- Đường mức của hàm mục tiêu F = a·x + b·y (dùng cho bài toán
        # tối ưu tuyến tính): vẽ đường thẳng ax + by = F, có thể "trượt"
        # song song khi F thay đổi qua thanh trượt bên sidebar. ---
        if objective:
            obj_a = objective.get("a", 1.0)
            obj_b = objective.get("b", 0.0)
            obj_F = objective.get("F", 0.0)
            obj_line = {"a": obj_a, "b": obj_b, "c": obj_F}
            pts = line_points(obj_line)
            xs_l = [p[0] for p in pts]
            ys_l = [p[1] for p in pts]
            obj_label = (f"F = {format_frac(obj_a)}x + {format_frac(obj_b)}y "
                         f"= {num(obj_F)}")
            ax.plot(xs_l, ys_l, color="#FF3EC9", linewidth=2.4,
                    linestyle="-.", alpha=0.95, zorder=4.4, label=obj_label)

            # --- Bước TỐI ƯU: khoanh tròn đỉnh đạt min (xanh lá) / max (cam)
            # ngay trên đồ thị, khớp với bảng giải thích bên panel phải. ---
            if highlight_optimum and len(vertices) >= 3:
                rows, row_min, row_max = evaluate_objective_at_vertices(
                    vertices, obj_a, obj_b)
                if row_min is not None:
                    ax.scatter([row_min["x"]], [row_min["y"]], s=260,
                               facecolors="none", edgecolors=MIN_COLOR,
                               linewidths=2.6, zorder=7)
                    ax.annotate("MIN", (row_min["x"], row_min["y"]),
                                textcoords="offset points", xytext=(0, -18),
                                ha="center", color=MIN_COLOR, fontsize=9,
                                fontweight="bold", zorder=7)
                if row_max is not None:
                    ax.scatter([row_max["x"]], [row_max["y"]], s=260,
                               facecolors="none", edgecolors=MAX_COLOR,
                               linewidths=2.6, zorder=7)
                    ax.annotate("MAX", (row_max["x"], row_max["y"]),
                                textcoords="offset points", xytext=(0, 18),
                                ha="center", color=MAX_COLOR, fontsize=9,
                                fontweight="bold", zorder=7)

            # --- Đường F vừa "khớp" (snap) chính xác vào một đỉnh: vẽ 2
            # vòng tròn đồng tâm (trắng ngoài + vàng trong) để nổi bật rõ
            # ràng, kèm nhãn "KHỚP ĐỈNH", giúp người dùng biết đường mức
            # đang đi QUA ĐÚNG đỉnh chứ không phải chỉ gần đúng. ---
            if snap_vertex is not None:
                svx, svy = snap_vertex
                ax.scatter([svx], [svy], s=480, facecolors="none",
                           edgecolors="#FFFFFF", linewidths=2.2, alpha=0.95,
                           zorder=7.5)
                ax.scatter([svx], [svy], s=210, facecolors="none",
                           edgecolors=SOLUTION_YELLOW, linewidths=2.6,
                           zorder=7.6)
                ax.annotate("🔒 KHỚP ĐỈNH", (svx, svy),
                            textcoords="offset points", xytext=(0, 24),
                            ha="center", color="#FFFFFF", fontsize=8.5,
                            fontweight="bold", zorder=7.7)

        if n or objective:
            legend = ax.legend(loc="upper left", fontsize=7.5, framealpha=0.25,
                               facecolor=BG_CARD, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
            if legend:
                legend.get_frame().set_alpha(0.35)

        self.canvas.draw_idle()

    # -------------------------------------------------- TƯƠNG TÁC CHUỘT
    def reset_view(self):
        """Đưa khung nhìn về mặc định."""
        self.view_xlim = (XMIN, XMAX)
        self.view_ylim = (YMIN, YMAX)
        if self._last_render_args:
            self.render(*self._last_render_args)
        else:
            self.ax.set_xlim(self.view_xlim)
            self.ax.set_ylim(self.view_ylim)
            self.canvas.draw_idle()

    def _clamp_span(self, span):
        return max(ZOOM_MIN_SPAN, min(ZOOM_MAX_SPAN, span))

    def _on_scroll(self, event):
        if event.xdata is None or event.ydata is None:
            return
        ax = self.ax
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        factor = 0.85 if event.button == "up" else (1 / 0.85)

        new_w = self._clamp_span((x1 - x0) * factor)
        new_h = self._clamp_span((y1 - y0) * factor)

        relx = (event.xdata - x0) / (x1 - x0) if (x1 - x0) else 0.5
        rely = (event.ydata - y0) / (y1 - y0) if (y1 - y0) else 0.5

        new_x0 = event.xdata - relx * new_w
        new_y0 = event.ydata - rely * new_h
        self.view_xlim = (new_x0, new_x0 + new_w)
        self.view_ylim = (new_y0, new_y0 + new_h)
        ax.set_xlim(self.view_xlim)
        ax.set_ylim(self.view_ylim)
        self.canvas.draw_idle()

    def _on_press(self, event):
        if event.button != 1 or event.inaxes != self.ax:
            return
        self._pan_active = True
        self._pan_start_pixel = (event.x, event.y)
        self._pan_start_xlim = self.ax.get_xlim()
        self._pan_start_ylim = self.ax.get_ylim()

    def _on_release(self, event):
        self._pan_active = False
        self._pan_start_pixel = None

    def _on_drag(self, event):
        if not self._pan_active or event.x is None or event.y is None:
            return
        dx_px = event.x - self._pan_start_pixel[0]
        dy_px = event.y - self._pan_start_pixel[1]
        inv = self.ax.transData.inverted()
        ox, oy = inv.transform((0, 0))
        nx, ny = inv.transform((dx_px, dy_px))
        ddx, ddy = nx - ox, ny - oy
        x0, x1 = self._pan_start_xlim
        y0, y1 = self._pan_start_ylim
        self.view_xlim = (x0 - ddx, x1 - ddx)
        self.view_ylim = (y0 - ddy, y1 - ddy)
        self.ax.set_xlim(self.view_xlim)
        self.ax.set_ylim(self.view_ylim)
        self.canvas.draw_idle()


# =============================================================================
# 4. GIAO DIỆN NGƯỜI DÙNG (APP)
# =============================================================================
class InequalityRow(ctk.CTkFrame):
    """Một dòng trong danh sách hệ BPT: checkbox hiện/ẩn + màu + text + nút xóa."""

    def __init__(self, parent, ineq, on_toggle, on_delete):
        super().__init__(parent, fg_color=BG_CARD, corner_radius=8, height=36)
        self.ineq = ineq
        self.grid_propagate(False)

        self.var = ctk.BooleanVar(value=True)
        chk = ctk.CTkCheckBox(self, text="", variable=self.var, width=18,
                              command=lambda: on_toggle(ineq, self.var.get()),
                              fg_color=ineq["color"], hover_color=ineq["color"],
                              border_color=TEXT_MUTED, checkmark_color=BG_MAIN)
        chk.grid(row=0, column=0, padx=(8, 4), pady=6)

        swatch = ctk.CTkLabel(self, text="", width=14, height=14,
                              fg_color=ineq["color"], corner_radius=3)
        swatch.grid(row=0, column=1, padx=4, pady=6)

        label = ctk.CTkLabel(self, text=ineq["raw"], text_color=TEXT_COLOR,
                             font=ctk.CTkFont(family="Consolas", size=12), anchor="w")
        label.grid(row=0, column=2, padx=6, pady=6, sticky="w")

        del_btn = ctk.CTkButton(self, text="✕", width=26, height=22,
                                fg_color="transparent", hover_color=DANGER_COLOR,
                                text_color=TEXT_MUTED, corner_radius=6,
                                command=lambda: on_delete(ineq))
        del_btn.grid(row=0, column=3, padx=(4, 8), pady=6)

        self.columnconfigure(2, weight=1)


class App(ctk.CTk):
    DEFAULT_SYSTEM = [
        "2x + y <= 4",
        "x - y <= 1",
        "x >= 0",
        "y >= 0",
    ]

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")

        self.title("CyberMath: Hệ BPT 2 Ẩn")
        self.geometry("1440x880")
        self.minsize(1150, 700)
        self.configure(fg_color=BG_MAIN)

        self.inequalities = []
        self.steps = []
        self.step_idx = -1
        self.color_cycle = itertools.cycle(NEON_COLORS)
        self.autoplay_on = False
        self._autoplay_job = None
        self.show_vertices = False   # Bật/tắt hiển thị đỉnh của đa giác nghiệm cuối

        # Hàm mục tiêu F = a·x + b·y dùng cho bài toán tối ưu tuyến tính
        self.show_objective_line = False
        self.obj_a = 1.0
        self.obj_b = 1.0
        self.obj_F = 0.0
        self.snap_vertex = None      # Đỉnh mà đường F đang "khớp" chính xác vào (nếu có)

        self._build_body()
        self._build_footer()

        self._load_default_system()

    # ------------------------------------------------------------------ BODY
    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color=BG_MAIN, corner_radius=0)
        body.pack(side="top", fill="both", expand=True)
        body.columnconfigure(0, weight=19, minsize=250)   # Sidebar nhập liệu (hẹp)
        body.columnconfigure(1, weight=63)                # Vùng đồ thị (lớn nhất, mở rộng thêm)
        body.columnconfigure(2, weight=18, minsize=210)   # Khung giải thích (thu hẹp lại)
        body.rowconfigure(0, weight=1)

        self._build_sidebar(body)
        self._build_canvas_area(body)
        self._build_explain_panel(body)

    # --------------------------------------------------------------- SIDEBAR
    def _build_sidebar(self, parent):
        sidebar = ctk.CTkScrollableFrame(parent, fg_color=BG_SIDEBAR, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")

        pad = 12

        # --- Nhập liệu ---
        ctk.CTkLabel(sidebar, text="1. NHẬP BẤT PHƯƠNG TRÌNH",
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=NEON_COLORS[1]).pack(anchor="w", padx=pad, pady=(16, 6))

        ctk.CTkLabel(sidebar, text="Dạng:  a·x + b·y  [dấu]  c",
                    font=ctk.CTkFont(size=11), text_color=TEXT_MUTED
                    ).pack(anchor="w", padx=pad, pady=(0, 6))

        # x và y luôn có sẵn, cố định trong công thức — người dùng chỉ cần
        # điền các hệ số a, b, c và chọn dấu so sánh (hiển thị bằng ký hiệu
        # toán học ≤ ≥, không bao giờ hiện dạng gõ máy "<=" / ">=").
        form = ctk.CTkFrame(sidebar, fg_color="transparent")
        form.pack(fill="x", padx=pad, pady=(0, 4))

        entry_style = dict(width=38, height=34, corner_radius=8, fg_color=BG_CARD,
                           border_color=GRID_COLOR, text_color=TEXT_COLOR,
                           font=ctk.CTkFont(family="Consolas", size=13),
                           justify="center")
        lbl_style = dict(font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
                         text_color=NEON_COLORS[0])

        self.entry_a = ctk.CTkEntry(form, **entry_style)
        self.entry_a.insert(0, "1")
        self.entry_a.grid(row=0, column=0, padx=(0, 3))
        self.entry_a.bind("<FocusIn>", lambda ev: self._set_active_field(self.entry_a))

        ctk.CTkLabel(form, text="x +", **lbl_style).grid(row=0, column=1, padx=2)

        self.entry_b = ctk.CTkEntry(form, **entry_style)
        self.entry_b.insert(0, "0")
        self.entry_b.grid(row=0, column=2, padx=3)
        self.entry_b.bind("<FocusIn>", lambda ev: self._set_active_field(self.entry_b))

        ctk.CTkLabel(form, text="y", **lbl_style).grid(row=0, column=3, padx=2)

        # Ô chọn dấu so sánh: MỘT nút bấm duy nhất, chỉ hiện đúng ký hiệu
        # đang chọn (không có đường kẻ / mũi tên dropdown đi kèm). Nhấp vào
        # đây sẽ hiện ra danh sách các dấu khác để chọn.
        self.op_menu_var = ctk.StringVar(value="≤")
        self.op_button = ctk.CTkButton(
            form, textvariable=self.op_menu_var, width=52, height=34,
            corner_radius=8, fg_color=BG_CARD, hover_color="#26263A",
            text_color=TEXT_COLOR, font=ctk.CTkFont(size=14, weight="bold"),
            command=self._show_op_popup)
        self.op_button.grid(row=0, column=4, padx=3)

        # Danh sách dấu hiện ra khi bấm vào ô trên (menu popup gốc của
        # tkinter, không có phần divider/arrow riêng như CTkOptionMenu).
        self._op_popup = tk.Menu(self, tearoff=0, bg=BG_CARD, fg=TEXT_COLOR,
                                 activebackground="#26263A", activeforeground=TEXT_COLOR,
                                 bd=0, relief="flat",
                                 font=("Segoe UI", 13, "bold"))
        for sym in ["≤", "≥", "<", ">", "="]:
            self._op_popup.add_command(
                label=sym, command=lambda s=sym: self.op_menu_var.set(s))

        self.entry_c = ctk.CTkEntry(form, **entry_style)
        self.entry_c.insert(0, "0")
        self.entry_c.grid(row=0, column=5, padx=(3, 0))
        self.entry_c.bind("<FocusIn>", lambda ev: self._set_active_field(self.entry_c))

        for e in (self.entry_a, self.entry_b, self.entry_c):
            e.bind("<Return>", lambda ev: self._add_inequality())

        self.active_field = self.entry_a  # ô a/b/c đang được nhắm để bấm số vào

        # --- Bàn phím số nhỏ: bấm chuột để nhập nhanh vào ô a/b/c đang chọn ---
        ctk.CTkLabel(sidebar, text="Bấm số để điền nhanh vào ô đang chọn (a/b/c):",
                    font=ctk.CTkFont(size=10), text_color=TEXT_MUTED,
                    wraplength=228, justify="left").pack(anchor="w", padx=pad, pady=(6, 4))

        keypad = ctk.CTkFrame(sidebar, fg_color="transparent")
        keypad.pack(fill="x", padx=pad, pady=(0, 10))
        for c in range(4):
            keypad.columnconfigure(c, weight=1)

        key_style = dict(height=30, corner_radius=6, fg_color=BG_CARD,
                         hover_color="#26263A", text_color=TEXT_COLOR,
                         font=ctk.CTkFont(family="Consolas", size=13))
        util_style = dict(height=30, corner_radius=6, fg_color="#232336",
                          hover_color="#2E2E46", text_color=NEON_COLORS[0],
                          font=ctk.CTkFont(family="Consolas", size=12, weight="bold"))

        keypad_rows = [
            [("7", key_style), ("8", key_style), ("9", key_style), ("⌫", util_style)],
            [("4", key_style), ("5", key_style), ("6", key_style), ("C", util_style)],
            [("1", key_style), ("2", key_style), ("3", key_style), ("/", key_style)],
            [("0", key_style), (".", key_style), ("-", key_style), ("+", key_style)],
        ]
        for r, row in enumerate(keypad_rows):
            for c, (label, style) in enumerate(row):
                if label == "⌫":
                    cmd = self._keypad_backspace
                elif label == "C":
                    cmd = self._keypad_clear
                else:
                    cmd = lambda t=label: self._keypad_insert(t)
                ctk.CTkButton(keypad, text=label, width=1, command=cmd,
                             **style).grid(row=r, column=c, padx=2, pady=2, sticky="ew")

        add_btn = ctk.CTkButton(
            sidebar, text="+ THÊM VÀO HỆ", height=36, corner_radius=8,
            fg_color=NEON_COLORS[2], hover_color="#00CC50", text_color=BG_MAIN,
            font=ctk.CTkFont(weight="bold"), command=self._add_inequality)
        add_btn.pack(fill="x", padx=pad, pady=(10, 14))

        # --- Danh sách hệ ---
        ctk.CTkLabel(sidebar, text="2. DANH SÁCH HỆ BPT",
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=NEON_COLORS[1]).pack(anchor="w", padx=pad, pady=(4, 6))

        self.list_frame = ctk.CTkFrame(sidebar, fg_color=BG_MAIN, corner_radius=8)
        self.list_frame.pack(fill="x", padx=pad, pady=(0, 10))

        # --- Nút giải ---
        self.solve_btn = ctk.CTkButton(
            sidebar, text="▶  BẮT ĐẦU GIẢI", height=44, corner_radius=10,
            fg_color=NEON_COLORS[0], hover_color="#00B8C6", text_color=BG_MAIN,
            font=ctk.CTkFont(size=14, weight="bold"), command=self._solve)
        self.solve_btn.pack(fill="x", padx=pad, pady=(6, 16))

        # --- Điều hướng bước ---
        ctk.CTkLabel(sidebar, text="3. ĐIỀU HƯỚNG TỪNG BƯỚC",
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=NEON_COLORS[1]).pack(anchor="w", padx=pad, pady=(4, 6))

        nav = ctk.CTkFrame(sidebar, fg_color="transparent")
        nav.pack(fill="x", padx=pad, pady=(0, 6))
        btn_style = dict(width=40, height=32, corner_radius=8, fg_color=BG_CARD,
                         hover_color="#26263A", text_color=TEXT_COLOR)
        ctk.CTkButton(nav, text="⏮", command=self._go_first, **btn_style).pack(side="left", padx=2)
        ctk.CTkButton(nav, text="◀", command=self._go_prev, **btn_style).pack(side="left", padx=2)
        ctk.CTkButton(nav, text="▶", command=self._go_next, **btn_style).pack(side="left", padx=2)
        ctk.CTkButton(nav, text="⏭", command=self._go_last, **btn_style).pack(side="left", padx=2)

        self.step_label = ctk.CTkLabel(sidebar, text="Bước 0/0",
                                       text_color=TEXT_COLOR,
                                       font=ctk.CTkFont(size=12, weight="bold"))
        self.step_label.pack(anchor="w", padx=pad, pady=(4, 8))

        self.autoplay_btn = ctk.CTkButton(
            sidebar, text="⏵  Auto Play (2.5s / bước)", height=36, corner_radius=8,
            fg_color=BG_CARD, hover_color="#26263A", text_color=TEXT_COLOR,
            command=self._toggle_autoplay)
        self.autoplay_btn.pack(fill="x", padx=pad, pady=(0, 16))

        # --- Đường mức tối ưu F = ax + by ---
        ctk.CTkLabel(sidebar, text="4. ĐƯỜNG MỨC TỐI ƯU  F = ax + by",
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=NEON_COLORS[1]).pack(anchor="w", padx=pad, pady=(4, 6))

        ctk.CTkLabel(sidebar, text="Nhập hệ số a, b của hàm mục tiêu:",
                    font=ctk.CTkFont(size=11), text_color=TEXT_MUTED
                    ).pack(anchor="w", padx=pad, pady=(0, 6))

        obj_form = ctk.CTkFrame(sidebar, fg_color="transparent")
        obj_form.pack(fill="x", padx=pad, pady=(0, 8))

        obj_entry_style = dict(width=48, height=32, corner_radius=8, fg_color=BG_CARD,
                               border_color=GRID_COLOR, text_color=TEXT_COLOR,
                               font=ctk.CTkFont(family="Consolas", size=13),
                               justify="center")

        self.obj_entry_a = ctk.CTkEntry(obj_form, **obj_entry_style)
        self.obj_entry_a.insert(0, "1")
        self.obj_entry_a.grid(row=0, column=0, padx=(0, 3))
        self.obj_entry_a.bind("<Return>", self._apply_objective_coeffs)

        ctk.CTkLabel(obj_form, text="x +",
                    font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
                    text_color=NEON_COLORS[0]).grid(row=0, column=1, padx=2)

        self.obj_entry_b = ctk.CTkEntry(obj_form, **obj_entry_style)
        self.obj_entry_b.insert(0, "1")
        self.obj_entry_b.grid(row=0, column=2, padx=3)
        self.obj_entry_b.bind("<Return>", self._apply_objective_coeffs)

        ctk.CTkLabel(obj_form, text="y",
                    font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
                    text_color=NEON_COLORS[0]).grid(row=0, column=3, padx=(2, 8))

        apply_obj_btn = ctk.CTkButton(
            obj_form, text="Áp dụng", width=64, height=32, corner_radius=8,
            fg_color=BG_CARD, hover_color="#26263A", text_color=TEXT_COLOR,
            font=ctk.CTkFont(size=11), command=self._apply_objective_coeffs)
        apply_obj_btn.grid(row=0, column=4, padx=(2, 0))

        self.obj_toggle_btn = ctk.CTkButton(
            sidebar, text="📈 Hiện đường mức F", height=36, corner_radius=8,
            fg_color=BG_CARD, hover_color="#26263A", text_color=TEXT_COLOR,
            command=self._toggle_objective_line)
        self.obj_toggle_btn.pack(fill="x", padx=pad, pady=(0, 8))

        self.obj_f_label = ctk.CTkLabel(
            sidebar, text="F = x + y  =  0", text_color="#FF3EC9",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            wraplength=228, justify="left")
        self.obj_f_label.pack(anchor="w", padx=pad, pady=(0, 4))

        # number_of_steps=None -> thanh trượt LIÊN TỤC (giá trị thực float
        # theo từng pixel kéo chuột), thay vì chỉ 400 mốc rời rạc cố định.
        # Nhờ vậy khi kéo ngang qua BẤT KỲ đỉnh nào của đa giác nghiệm,
        # luôn có một vị trí đủ gần (trong phạm vi SNAP_TOLERANCE) để đường
        # F khớp chính xác vào đỉnh đó — không còn bỏ sót đỉnh chỉ vì đỉnh
        # ấy rơi giữa hai mốc rời rạc của thanh trượt.
        self.obj_slider = ctk.CTkSlider(
            sidebar, from_=-10, to=10, number_of_steps=None,
            progress_color="#FF3EC9", button_color="#FF3EC9",
            button_hover_color="#D6259F", fg_color=BG_CARD,
            command=self._on_objective_slider)
        self.obj_slider.set(0)
        self.obj_slider.pack(fill="x", padx=pad, pady=(0, 6))

        refresh_range_btn = ctk.CTkButton(
            sidebar, text="🔄 Cập nhật khung trượt theo khung nhìn", height=30,
            corner_radius=8, fg_color=BG_CARD, hover_color="#26263A",
            text_color=TEXT_MUTED, font=ctk.CTkFont(size=10),
            command=lambda: self._update_objective_bounds())
        refresh_range_btn.pack(fill="x", padx=pad, pady=(0, 16))

    # ---------------------------------------------------------- CANVAS AREA
    def _build_canvas_area(self, parent):
        canvas_area = ctk.CTkFrame(parent, fg_color=BG_CANVAS, corner_radius=0)
        canvas_area.grid(row=0, column=1, sticky="nsew")

        self.math_canvas = MathCanvas(canvas_area, on_mouse_move=self._on_mouse_move)

        # --- Nút đặt lại khung nhìn (nổi ở góc DƯỚI-TRÁI vùng đồ thị) ---
        reset_view_btn = ctk.CTkButton(
            canvas_area, text="⤢ Reset View", width=110, height=28, corner_radius=8,
            fg_color=BG_CARD, hover_color="#26263A", text_color=TEXT_COLOR,
            font=ctk.CTkFont(size=11), command=lambda: self.math_canvas.reset_view())
        reset_view_btn.place(relx=0.0, rely=1.0, anchor="sw", x=12, y=-12)

        # --- Nút hiện/ẩn các đỉnh của đa giác nghiệm cuối cùng (DƯỚI-TRÁI) ---
        self.vertices_btn = ctk.CTkButton(
            canvas_area, text="📐 Hiện đỉnh", width=110, height=28, corner_radius=8,
            fg_color=BG_CARD, hover_color="#26263A", text_color=TEXT_COLOR,
            font=ctk.CTkFont(size=11), command=self._toggle_vertices)
        self.vertices_btn.place(relx=0.0, rely=1.0, anchor="sw", x=130, y=-12)

        # --- Chú thích tương tác chuột: đặt ở góc DƯỚI-PHẢI vùng đồ thị ---
        hint_lbl = ctk.CTkLabel(
            canvas_area, text="🖱 Cuộn để zoom · Giữ & kéo để di chuyển",
            fg_color=BG_CARD, corner_radius=8, text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=10))
        hint_lbl.place(relx=1.0, rely=1.0, anchor="se", x=-12, y=-12)

        # --- Nút đặt lại hệ mẫu mặc định: đặt gọn ở góc TRÊN-PHẢI đồ thị ---
        reset_default_btn = ctk.CTkButton(
            canvas_area, text="↺ Đặt lại mẫu", width=110, height=28, corner_radius=8,
            fg_color=BG_CARD, hover_color="#26263A", text_color=TEXT_COLOR,
            font=ctk.CTkFont(size=11), command=self._reset_to_default)
        reset_default_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-12, y=12)

    # ------------------------------------------------------- EXPLAIN PANEL
    def _build_explain_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=BG_SIDEBAR, corner_radius=0)
        panel.grid(row=0, column=2, sticky="nsew")

        header = ctk.CTkFrame(panel, fg_color=BG_CARD, corner_radius=12,
                              border_width=1, border_color=NEON_COLORS[0])
        header.pack(fill="x", padx=14, pady=(16, 8))
        ctk.CTkLabel(header, text="📟 GIẢI THÍCH TỪNG BƯỚC",
                    font=ctk.CTkFont(size=13, weight="bold"),
                    text_color=NEON_COLORS[0]).pack(anchor="w", padx=12, pady=10)

        # Dùng CTkScrollableFrame + CTkTextbox thay vì CTkLabel để chứa
        # được bảng F tại từng đỉnh (nhiều dòng, cần cuộn khi hệ có nhiều
        # bất phương trình / nhiều đỉnh) mà không bị cắt hoặc bóp chữ.
        self.explain_card = ctk.CTkFrame(panel, fg_color=BG_CARD, corner_radius=12,
                                         border_width=1, border_color=GRID_COLOR)
        self.explain_card.pack(fill="both", expand=True, padx=14, pady=(0, 16))

        self.explain_text = ctk.CTkTextbox(
            self.explain_card, fg_color=BG_CARD, text_color=TEXT_COLOR,
            font=ctk.CTkFont(family="Consolas", size=12), wrap="word",
            activate_scrollbars=True)
        self.explain_text.pack(fill="both", expand=True, padx=8, pady=8)
        self._set_explain_text("Nhấn “BẮT ĐẦU GIẢI” để xem\ntừng bước minh họa.")

    def _set_explain_text(self, text):
        """Ghi nội dung vào ô giải thích (CTkTextbox chỉ đọc)."""
        self.explain_text.configure(state="normal")
        self.explain_text.delete("1.0", "end")
        self.explain_text.insert("1.0", text)
        self.explain_text.configure(state="disabled")

    # ----------------------------------------------------------------FOOTER
    def _build_footer(self):
        footer = ctk.CTkFrame(self, fg_color=BG_SIDEBAR, height=30, corner_radius=0)
        footer.pack(side="bottom", fill="x")
        footer.pack_propagate(False)

        self.coord_label = ctk.CTkLabel(footer, text="x = —   y = —",
                                        text_color=TEXT_MUTED,
                                        font=ctk.CTkFont(family="Consolas", size=11))
        self.coord_label.pack(side="left", padx=16)

        self.status_label = ctk.CTkLabel(footer, text="Sẵn sàng.",
                                         text_color=TEXT_MUTED,
                                         font=ctk.CTkFont(size=11))
        self.status_label.pack(side="right", padx=16)

    # =========================================================================
    # HÀNH ĐỘNG NGƯỜI DÙNG
    # =========================================================================
    def _show_op_popup(self):
        """Hiện danh sách các dấu so sánh (≤ ≥ < > =) ngay bên dưới nút bấm."""
        x = self.op_button.winfo_rootx()
        y = self.op_button.winfo_rooty() + self.op_button.winfo_height()
        try:
            self._op_popup.tk_popup(x, y)
        finally:
            self._op_popup.grab_release()

    def _set_active_field(self, entry):
        """Ghi nhớ ô a/b/c đang được focus để bàn phím số biết nhập vào đâu."""
        self.active_field = entry

    def _keypad_insert(self, text):
        """Chèn ký tự vào đúng vị trí con trỏ của ô a/b/c đang được chọn."""
        entry = getattr(self, "active_field", None) or self.entry_a
        try:
            entry.insert("insert", text)
        except Exception:
            entry.insert(ctk.END, text)
        entry.focus_set()

    def _keypad_backspace(self):
        """Xóa 1 ký tự trước con trỏ trong ô a/b/c đang được chọn."""
        entry = getattr(self, "active_field", None) or self.entry_a
        try:
            entry.delete("insert-1c", "insert")
        except Exception:
            txt = entry.get()
            if txt:
                entry.delete(len(txt) - 1, ctk.END)
        entry.focus_set()

    def _keypad_clear(self):
        """Xóa trắng toàn bộ ô a/b/c đang được chọn."""
        entry = getattr(self, "active_field", None) or self.entry_a
        entry.delete(0, ctk.END)
        entry.focus_set()

    def _add_inequality_text(self, text):
        """Thêm một BPT từ chuỗi thô (dùng nội bộ khi nạp hệ mẫu mặc định)."""
        try:
            ineq = parse_inequality(text)
        except ParseError as e:
            self._set_status(f"⚠ Lỗi: {e}", error=True)
            return False
        except Exception as e:
            self._set_status(f"⚠ Lỗi không xác định: {e}", error=True)
            return False

        # Luôn hiển thị lại bằng ký hiệu toán học (≤, ≥...), không bao giờ
        # hiển thị dạng gõ máy "<=" / ">=" cho người dùng.
        ineq["raw"] = format_linear(ineq["a"], ineq["b"], ineq["c"], OP_SYMBOLS[ineq["op"]])
        ineq["visible"] = True
        ineq["color"] = next(self.color_cycle)
        self.inequalities.append(ineq)
        self._refresh_list()
        self._set_status(f"Đã thêm: {ineq['raw']}")
        return True

    @staticmethod
    def _parse_signed_field(text, field_name):
        """Phân tích một ô nhập hệ số đơn (cho phép số âm và phân số a/b)."""
        text = (text or "").strip().replace(" ", "")
        if text == "":
            return 0.0
        sign = 1.0
        if text[0] in "+-":
            if text[0] == "-":
                sign = -1.0
            text = text[1:]
        if text == "":
            raise ParseError(f"Thiếu giá trị cho {field_name}.")
        if not re.match(r"^\d*\.?\d*(?:/\d*\.?\d*)?$", text):
            raise ParseError(f"Giá trị {field_name} không hợp lệ: '{text}'")
        return sign * _parse_coef(text)

    def _add_inequality(self):
        """Thêm BPT từ 3 ô nhập hệ số (a, b, c) + dấu so sánh dạng ký hiệu."""
        try:
            a_val = self._parse_signed_field(self.entry_a.get(), "a")
            b_val = self._parse_signed_field(self.entry_b.get(), "b")
            c_val = self._parse_signed_field(self.entry_c.get(), "c")
        except ParseError as e:
            self._set_status(f"⚠ Lỗi: {e}", error=True)
            return

        if abs(a_val) < 1e-9 and abs(b_val) < 1e-9:
            self._set_status("⚠ Bất phương trình phải có hệ số x hoặc y khác 0.", error=True)
            return

        op_symbol = self.op_menu_var.get()
        op_internal = OP_SYMBOL_TO_INTERNAL.get(op_symbol, "<=")

        ineq = {
            "a": a_val, "b": b_val, "c": c_val, "op": op_internal,
            "raw": format_linear(a_val, b_val, c_val, op_symbol),
        }
        ineq["visible"] = True
        ineq["color"] = next(self.color_cycle)
        self.inequalities.append(ineq)
        self._refresh_list()
        self._set_status(f"Đã thêm: {ineq['raw']}")

        # Về giá trị mặc định sau khi thêm, để nhập BPT tiếp theo nhanh hơn.
        self.entry_a.delete(0, ctk.END); self.entry_a.insert(0, "1")
        self.entry_b.delete(0, ctk.END); self.entry_b.insert(0, "0")
        self.entry_c.delete(0, ctk.END); self.entry_c.insert(0, "0")
        self.entry_a.focus_set()

        self._invalidate_solution()

    def _delete_inequality(self, ineq):
        if ineq in self.inequalities:
            self.inequalities.remove(ineq)
            self._refresh_list()
            self._invalidate_solution()
            self._set_status("Đã xóa một bất phương trình khỏi hệ.")

    def _toggle_visibility(self, ineq, visible):
        ineq["visible"] = visible
        self._render_current()

    def _refresh_list(self):
        for child in self.list_frame.winfo_children():
            child.destroy()
        for ineq in self.inequalities:
            row = InequalityRow(self.list_frame, ineq,
                                on_toggle=self._toggle_visibility,
                                on_delete=self._delete_inequality)
            row.pack(fill="x", pady=3, padx=2)

    def _invalidate_solution(self):
        self.steps = []
        self.step_idx = -1
        self.snap_vertex = None
        self._stop_autoplay()
        self.step_label.configure(text="Bước 0/0")
        self._set_explain_text("Danh sách đã thay đổi.\nNhấn “BẮT ĐẦU GIẢI” để giải lại.")
        self.math_canvas.render(-1, self.inequalities, [], self.show_vertices,
                                self._current_objective(), snap_vertex=self.snap_vertex)

    def _solve(self):
        if not self.inequalities:
            self._set_status("⚠ Hệ chưa có bất phương trình nào.", error=True)
            return
        self.steps = StepEngine.generate(self.inequalities, self.show_objective_line)
        self.step_idx = 0
        self._render_current()
        self._set_status(f"Đã sinh {len(self.steps)} bước minh họa. Bắt đầu giải!")

    def _load_default_system(self):
        for text in self.DEFAULT_SYSTEM:
            self._add_inequality_text(text)
        self._solve()

    def _reset_to_default(self):
        self._stop_autoplay()
        self.inequalities = []
        self.color_cycle = itertools.cycle(NEON_COLORS)
        self._refresh_list()
        self._load_default_system()

    # ---- Điều hướng bước ----
    def _render_current(self):
        if not self.steps:
            return
        step = self.steps[self.step_idx]
        is_optimize_step = (step["type"] == "optimize")
        # Ở bước tối ưu, luôn hiện tọa độ các đỉnh trên đồ thị (dù nút
        # "Hiện đỉnh" đang tắt) để người dùng đối chiếu trực tiếp với bảng
        # giải thích bên panel phải.
        show_v = self.show_vertices or is_optimize_step
        self.math_canvas.render(self.step_idx, self.inequalities, self.steps, show_v,
                                self._current_objective(), highlight_optimum=is_optimize_step,
                                snap_vertex=self.snap_vertex)
        text = StepEngine.explain(step, self.inequalities, self.step_idx + 1, len(self.steps),
                                   self._current_objective())
        self._set_explain_text(text)
        self.step_label.configure(text=f"Bước {self.step_idx + 1}/{len(self.steps)}")

    def _go_first(self):
        if self.steps:
            self.step_idx = 0
            self._render_current()

    def _go_last(self):
        if self.steps:
            self.step_idx = len(self.steps) - 1
            self._render_current()

    def _go_next(self):
        if self.steps and self.step_idx < len(self.steps) - 1:
            self.step_idx += 1
            self._render_current()
        elif self.autoplay_on:
            self._stop_autoplay()

    def _go_prev(self):
        if self.steps and self.step_idx > 0:
            self.step_idx -= 1
            self._render_current()

    def _toggle_autoplay(self):
        if not self.steps:
            self._set_status("⚠ Hãy giải hệ trước khi dùng Auto Play.", error=True)
            return
        if self.autoplay_on:
            self._stop_autoplay()
        else:
            self.autoplay_on = True
            self.autoplay_btn.configure(text="⏸  Dừng Auto Play", fg_color=NEON_COLORS[1],
                                        text_color=BG_MAIN)
            if self.step_idx >= len(self.steps) - 1:
                self.step_idx = 0
                self._render_current()
            self._autoplay_tick()

    def _stop_autoplay(self):
        self.autoplay_on = False
        self.autoplay_btn.configure(text="⏵  Auto Play (2.5s / bước)",
                                    fg_color=BG_CARD, text_color=TEXT_COLOR)
        if self._autoplay_job is not None:
            try:
                self.after_cancel(self._autoplay_job)
            except Exception:
                pass
            self._autoplay_job = None

    def _autoplay_tick(self):
        if not self.autoplay_on:
            return
        if self.step_idx < len(self.steps) - 1:
            self.step_idx += 1
            self._render_current()
            self._autoplay_job = self.after(2500, self._autoplay_tick)
        else:
            self._stop_autoplay()

    def _toggle_vertices(self):
        """Bật/tắt hiển thị tọa độ các đỉnh của đa giác nghiệm cuối cùng."""
        self.show_vertices = not self.show_vertices
        self.vertices_btn.configure(
            text="📐 Ẩn đỉnh" if self.show_vertices else "📐 Hiện đỉnh",
            fg_color=NEON_COLORS[3] if self.show_vertices else BG_CARD,
            text_color=BG_MAIN if self.show_vertices else TEXT_COLOR)
        if self.steps:
            self._render_current()
        else:
            self.math_canvas.render(-1, self.inequalities, [], self.show_vertices,
                                    self._current_objective(), snap_vertex=self.snap_vertex)

    # ---- Đường mức tối ưu F = ax + by ----
    def _current_objective(self):
        """Trả về dict {a, b, F} nếu đường mức đang được bật hiển thị,
        ngược lại trả về None (không vẽ gì thêm)."""
        if self.show_objective_line:
            return {"a": self.obj_a, "b": self.obj_b, "F": self.obj_F}
        return None

    def _snap_objective_to_vertices(self, raw_value):
        """"Khớp" giá trị F (do thanh trượt sinh ra) vào đỉnh gần nhất của
        đa giác nghiệm, nếu đường thẳng ax+by=raw_value đi cách đỉnh đó
        trong phạm vi SNAP_TOLERANCE đơn vị trục (khoảng cách vuông góc
        thực sự trên mặt phẳng Oxy, không phải sai khác giá trị F thô).

        Trả về (F_đã_khớp, đỉnh_đã_khớp) — nếu không có đỉnh nào đủ gần,
        trả về (raw_value, None) để giữ nguyên giá trị người dùng vừa kéo.
        """
        a, b = self.obj_a, self.obj_b
        norm = math.hypot(a, b)
        if norm < 1e-9 or not self.inequalities:
            return raw_value, None

        vertices = compute_polygon_vertices(self.inequalities)
        if not vertices:
            return raw_value, None

        best_vertex = None
        best_dist = None
        best_F = raw_value
        for (vx, vy) in vertices:
            f_at_vertex = a * vx + b * vy
            # Khoảng cách vuông góc từ đỉnh (vx, vy) tới đường thẳng hiện
            # tại ax + by = raw_value, tính bằng đơn vị trục thật sự.
            dist = abs(f_at_vertex - raw_value) / norm
            if dist <= SNAP_TOLERANCE and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_vertex = (vx, vy)
                best_F = f_at_vertex

        return best_F, best_vertex

    def _render_all(self):
        """Vẽ lại đồ thị hiện tại (dù đã giải hay chưa), có tính cả
        đường mức F nếu đang bật."""
        if self.steps:
            self._render_current()
        else:
            self.math_canvas.render(-1, self.inequalities, [], self.show_vertices,
                                    self._current_objective(), snap_vertex=self.snap_vertex)

    def _apply_objective_coeffs(self, event=None):
        """Đọc hệ số a, b của hàm mục tiêu từ 2 ô nhập và áp dụng."""
        try:
            a = self._parse_signed_field(self.obj_entry_a.get(), "a")
            b = self._parse_signed_field(self.obj_entry_b.get(), "b")
        except ParseError as e:
            self._set_status(f"⚠ Lỗi: {e}", error=True)
            return
        if abs(a) < 1e-9 and abs(b) < 1e-9:
            self._set_status("⚠ Hệ số a, b của F không được đồng thời bằng 0.", error=True)
            return
        self.obj_a, self.obj_b = a, b
        self.snap_vertex = None
        self._set_status(f"Đã áp dụng F = {format_frac(a)}x + {format_frac(b)}y.")
        if self.show_objective_line:
            self._update_objective_bounds()
        else:
            self._update_obj_f_label()
        # Nếu bước tối ưu đang hiển thị, làm mới lại nội dung ngay để bảng
        # F tại các đỉnh phản ánh đúng hệ số a, b mới nhất.
        if self.steps and self.steps[self.step_idx]["type"] == "optimize":
            self._render_current()

    def _update_objective_bounds(self):
        """Tính lại khoảng giá trị [F_min, F_max] hợp lý cho thanh trượt,
        dựa trên khung nhìn hiện tại của đồ thị (vùng đang xem trên màn
        hình), rồi đặt thanh trượt về giá trị chính giữa khoảng đó."""
        a, b = self.obj_a, self.obj_b
        x0, x1 = self.math_canvas.view_xlim
        y0, y1 = self.math_canvas.view_ylim
        corners = [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
        vals = [a * cx + b * cy for cx, cy in corners]
        fmin, fmax = min(vals), max(vals)
        if fmax - fmin < 1e-6:
            fmax = fmin + 1.0
        self.obj_slider.configure(from_=fmin, to=fmax)
        mid = (fmin + fmax) / 2
        self.obj_slider.set(mid)
        self.obj_F = mid
        self.snap_vertex = None
        self._update_obj_f_label()
        self._render_all()

    def _update_obj_f_label(self):
        self.obj_f_label.configure(
            text=f"F = {format_frac(self.obj_a)}x + {format_frac(self.obj_b)}y  =  {num(self.obj_F)}")

    def _on_objective_slider(self, value):
        # Kéo thanh trượt tự do trước, sau đó thử "khớp" (snap) đường F
        # vào đỉnh gần nhất của đa giác nghiệm nếu đủ gần (<= 0.01 đơn vị
        # trục) -> giúp xác định min/max chính xác tuyệt đối thay vì chỉ
        # là một giá trị xấp xỉ do độ phân giải rời rạc của thanh trượt.
        snapped_F, snapped_vertex = self._snap_objective_to_vertices(value)
        self.obj_F = snapped_F
        self.snap_vertex = snapped_vertex
        if snapped_vertex is not None:
            # Cập nhật lại vị trí nút trượt để khớp hình ảnh với giá trị
            # F chính xác (set() chỉ thay đổi hiển thị, không gọi lại
            # command nên không gây đệ quy vô hạn).
            self.obj_slider.set(snapped_F)
            vx, vy = snapped_vertex
            self._set_status(
                f"🔒 Đường F đã khớp chính xác vào đỉnh ({format_frac(vx)}; {format_frac(vy)}).")
        self._update_obj_f_label()
        self._render_all()

    def _toggle_objective_line(self):
        """Bật/tắt hiển thị đường mức F = ax + by trên đồ thị.

        Vì bước "TỐI ƯU" (tính F tại từng đỉnh) chỉ tồn tại trong danh sách
        bước khi đường mức F đang bật, mỗi lần bật/tắt ta cần TÁI SINH lại
        danh sách bước (nếu hệ đã được giải) để thêm/bớt bước này cho khớp.
        """
        self.show_objective_line = not self.show_objective_line
        if self.show_objective_line:
            self.obj_toggle_btn.configure(
                text="📈 Ẩn đường mức F", fg_color="#FF3EC9", text_color=BG_MAIN)
            self._update_objective_bounds()  # tự tính khoảng trượt & vẽ lại
        else:
            self.obj_toggle_btn.configure(
                text="📈 Hiện đường mức F", fg_color=BG_CARD, text_color=TEXT_COLOR)
            self.snap_vertex = None
            self._render_all()

        if self.steps:
            was_at_end = (self.step_idx == len(self.steps) - 1)
            self.steps = StepEngine.generate(self.inequalities, self.show_objective_line)
            self.step_idx = min(self.step_idx, len(self.steps) - 1)
            if was_at_end:
                self.step_idx = len(self.steps) - 1
            self._render_current()

    # ---- Footer helpers ----
    def _on_mouse_move(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.coord_label.configure(text=f"x = {event.xdata:6.2f}   y = {event.ydata:6.2f}")
        else:
            self.coord_label.configure(text="x = —   y = —")

    def _set_status(self, msg, error=False):
        self.status_label.configure(text=msg, text_color=DANGER_COLOR if error else TEXT_MUTED)


# =============================================================================
# 5. ĐIỂM KHỞI CHẠY
# =============================================================================
def main():
    try:
        app = App()
        app.mainloop()
    except Exception as exc:  # phòng hờ lỗi hiếm gặp khi khởi tạo GUI
        messagebox.showerror("CyberMath — Lỗi khởi động", str(exc))
        raise


if __name__ == "__main__":
    main()
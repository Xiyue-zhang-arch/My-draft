from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(r"D:\Project\MY draft")
FIG_DIR = ROOT / "thesis_document" / "figures"
OUT_DIR = ROOT / "thesis_document" / "成果"
OUT_PATH = OUT_DIR / "物理引导双流门控融合轴承故障诊断研究.docx"

SEL_JSON = ROOT / "experiments/CWRU/实验结果" / "CWRU_V3" / "run_20261022_2105" / "selection_evaluation" / "selection.json"
WIN_DIR = ROOT / "experiments/CWRU/实验结果" / "CWRU_V3" / "window_random_20261023_0853"
FULL_DIR = ROOT / "experiments/CWRU/实验结果" / "CWRU_V3" / "load_full_20261024_0950"
SMALL_DIR = ROOT / "experiments/CWRU/实验结果" / "CWRU_V3" / "load_small_20261025_2017"

BLUE = (31, 119, 180)
ORANGE = (255, 127, 14)
GREEN = (44, 160, 44)
RED = (214, 39, 40)
PURPLE = (148, 103, 189)
PINK = (255, 51, 102)
TEAL = (23, 162, 184)
GOLD = (225, 171, 0)
GRAY = (110, 110, 110)
LIGHT_GRID = (218, 218, 218)
DARK = (32, 32, 32)
PALE_BLUE = (231, 241, 248)
PALE_LAV = (239, 235, 247)
PALE_GREEN = (232, 246, 236)
PALE_PEACH = (252, 239, 226)
PALE_PINK = (252, 233, 240)
PALE_GRAY = (246, 246, 246)

FONT_CN = Path(r"C:\Windows\Fonts\simsun.ttc")
FONT_CN_BOLD = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_EN = Path(r"C:\Windows\Fonts\times.ttf")
FONT_EN_BOLD = Path(r"C:\Windows\Fonts\timesbd.ttf")


def font(size: int, *, bold: bool = False, en: bool = False) -> ImageFont.FreeTypeFont:
    preferred = FONT_EN_BOLD if en and bold else FONT_EN if en else FONT_CN_BOLD if bold else FONT_CN
    try:
        return ImageFont.truetype(str(preferred), size=size)
    except OSError:
        return ImageFont.load_default()


def save_image(img: Image.Image, name: str) -> Path:
    path = FIG_DIR / name
    img.save(path, dpi=(220, 220), optimize=True)
    return path


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt, fill=DARK) -> None:
    draw.text(xy, text, font=fnt, fill=fill, anchor="mm")


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt, max_width: int) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        if not raw:
            lines.append("")
            continue
        current = ""
        for ch in raw:
            trial = current + ch
            if current and draw.textlength(trial, font=fnt) > max_width:
                lines.append(current)
                current = ch
            else:
                current = trial
        if current:
            lines.append(current)
    return lines


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    fill,
    outline=(70, 70, 70),
    text_fill=DARK,
    fnt=None,
    radius=16,
    width=3,
    padding=16,
) -> None:
    if fnt is None:
        fnt = font(30)
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    x0, y0, x1, y1 = box
    lines = wrap_text(draw, text, fnt, max(20, x1 - x0 - 2 * padding))
    line_h = max(1, int(fnt.size * 1.35))
    total_h = len(lines) * line_h
    y = (y0 + y1 - total_h) / 2 + line_h / 2
    for line in lines:
        draw_centered(draw, ((x0 + x1) / 2, y), line, fnt, text_fill)
        y += line_h


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], *, fill=DARK, width=5) -> None:
    draw.line([start, end], fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 16
    a1 = angle + math.pi * 0.82
    a2 = angle - math.pi * 0.82
    p1 = (end[0] + length * math.cos(a1), end[1] + length * math.sin(a1))
    p2 = (end[0] + length * math.cos(a2), end[1] + length * math.sin(a2))
    draw.polygon([end, p1, p2], fill=fill)


def dashed_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], *, fill=None, outline=(140, 140, 140), width=3, dash=14) -> None:
    x0, y0, x1, y1 = box
    if fill is not None:
        draw.rectangle(box, fill=fill)
    for x in range(x0, x1, 2 * dash):
        draw.line([(x, y0), (min(x + dash, x1), y0)], fill=outline, width=width)
        draw.line([(x, y1), (min(x + dash, x1), y1)], fill=outline, width=width)
    for y in range(y0, y1, 2 * dash):
        draw.line([(x0, y), (x0, min(y + dash, y1))], fill=outline, width=width)
        draw.line([(x1, y), (x1, min(y + dash, y1))], fill=outline, width=width)


def rotated_text(img: Image.Image, xy: tuple[int, int], text: str, fnt, fill=DARK, angle=90) -> None:
    tmp = Image.new("RGBA", (500, 120), (255, 255, 255, 0))
    d = ImageDraw.Draw(tmp)
    d.text((250, 60), text, font=fnt, fill=fill, anchor="mm")
    tmp = tmp.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    img.alpha_composite(tmp, (int(xy[0] - tmp.width / 2), int(xy[1] - tmp.height / 2)))


def axes(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xticks: Sequence[float],
    yticks: Sequence[float],
    xlabel: str,
    ylabel: str,
    *,
    title: str = "",
    xlabels: Sequence[str] | None = None,
    y_fmt=lambda v: f"{v:g}",
) -> tuple:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, fill="white", outline=DARK, width=3)
    f_tick = font(24, en=False)
    f_lab = font(29, bold=True)
    for v in yticks:
        py = y1 - (v - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)
        draw.line([(x0, py), (x1, py)], fill=LIGHT_GRID, width=2)
        draw.text((x0 - 14, py), y_fmt(v), font=f_tick, fill=DARK, anchor="rm")
    for i, v in enumerate(xticks):
        px = x0 + (v - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
        draw.line([(px, y0), (px, y1)], fill=(232, 232, 232), width=1)
        label = xlabels[i] if xlabels else f"{v:g}"
        draw.text((px, y1 + 18), label, font=f_tick, fill=DARK, anchor="mt")
    draw.text(((x0 + x1) / 2, y1 + 66), xlabel, font=f_lab, fill=DARK, anchor="mt")
    rotated_text(img, (x0 - 88, int((y0 + y1) / 2)), ylabel, f_lab, DARK, 90)
    if title:
        draw.text(((x0 + x1) / 2, y0 - 20), title, font=font(31, bold=True), fill=DARK, anchor="mb")

    def tx(v):
        return x0 + (v - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)

    def ty(v):
        return y1 - (v - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)

    return tx, ty


def legend(draw: ImageDraw.ImageDraw, origin: tuple[int, int], entries: Sequence[tuple[str, tuple[int, int, int]]], *, width=300) -> None:
    x, y = origin
    h = 42 * len(entries) + 18
    draw.rectangle((x, y, x + width, y + h), fill=(255, 255, 255), outline=(80, 80, 80), width=2)
    fnt = font(24)
    for i, (name, color) in enumerate(entries):
        yy = y + 28 + i * 42
        draw.line([(x + 18, yy), (x + 70, yy)], fill=color, width=6)
        draw.text((x + 82, yy), name, font=fnt, fill=DARK, anchor="lm")


def blues(value: float) -> tuple[int, int, int]:
    v = max(0.0, min(1.0, value))
    stops = [
        (247, 251, 255),
        (222, 235, 247),
        (198, 219, 239),
        (158, 202, 225),
        (107, 174, 214),
        (66, 146, 198),
        (33, 113, 181),
        (8, 81, 156),
        (8, 48, 107),
    ]
    pos = v * (len(stops) - 1)
    i = int(math.floor(pos))
    j = min(i + 1, len(stops) - 1)
    t = pos - i
    return tuple(int(stops[i][k] * (1 - t) + stops[j][k] * t) for k in range(3))


def create_mechanism_figure() -> Path:
    img = Image.new("RGB", (1900, 880), "white")
    d = ImageDraw.Draw(img)
    dashed_rect(d, (35, 55, 600, 820), fill=PALE_BLUE)
    dashed_rect(d, (665, 55, 1260, 820), fill=PALE_PINK)
    dashed_rect(d, (1325, 55, 1865, 820), fill=PALE_GREEN)
    draw_centered(d, (318, 92), "轴承局部损伤机理", font(38, bold=True))
    draw_centered(d, (962, 92), "可观测振动响应", font(38, bold=True))
    draw_centered(d, (1595, 92), "诊断信息映射", font(38, bold=True))

    cx, cy = 320, 395
    d.ellipse((cx - 190, cy - 190, cx + 190, cy + 190), outline=(60, 60, 60), width=20, fill=(252, 252, 252))
    d.ellipse((cx - 100, cy - 100, cx + 100, cy + 100), outline=(80, 80, 80), width=18, fill=(235, 235, 235))
    for a in np.linspace(0, 2 * math.pi, 9, endpoint=False):
        bx = cx + int(145 * math.cos(a))
        by = cy + int(145 * math.sin(a))
        d.ellipse((bx - 28, by - 28, bx + 28, by + 28), fill=(198, 205, 212), outline=DARK, width=3)
    d.arc((cx - 203, cy - 203, cx + 203, cy + 203), 310, 342, fill=RED, width=22)
    arrow(d, (405, 255), (500, 185), fill=RED, width=5)
    d.text((514, 178), "局部剥落", font=font(30, bold=True), fill=RED, anchor="lm")
    d.text((320, 640), "滚动体周期经过缺陷位置\n激发冲击并调制结构共振", font=font(31), fill=DARK, anchor="mm", align="center")
    d.text((320, 745), "冲击周期随转速与几何参数变化", font=font(27), fill=GRAY, anchor="mm")

    x0, x1 = 730, 1195
    ymid = 360
    pts = []
    rng = np.random.default_rng(20261027)
    for i in range(420):
        x = x0 + i / 419 * (x1 - x0)
        base = 12 * math.sin(i * 0.18) + rng.normal(0, 8)
        impulse = 0
        for k in range(35, 420, 78):
            if i >= k:
                impulse += 115 * math.exp(-(i - k) / 16) * math.sin((i - k) * 0.8)
        pts.append((x, ymid - base - impulse))
    d.line(pts, fill=BLUE, width=3)
    d.line([(x0, ymid), (x1, ymid)], fill=(150, 150, 150), width=2)
    for k in range(35, 420, 78):
        xx = x0 + k / 419 * (x1 - x0)
        d.line([(xx, 190), (xx, 540)], fill=(210, 210, 210), width=2)
    d.text((962, 575), "时域周期冲击与衰减振荡", font=font(31, bold=True), fill=DARK, anchor="mm")
    d.text((962, 660), "包络谱中的转频、保持架、滚动体、\n外圈和内圈候选阶次", font=font(30), fill=DARK, anchor="mm", align="center")
    d.text((962, 755), "强度、对比度、重复性、相位集中及形状统计", font=font(27), fill=GRAY, anchor="mm")

    rounded_box(d, (1390, 170, 1800, 285), "4 个频带", fill=PALE_LAV, fnt=font(34, bold=True))
    rounded_box(d, (1390, 335, 1800, 480), "6 类候选周期\n× 8 类周期描述", fill=PALE_PEACH, fnt=font(30, bold=True))
    rounded_box(d, (1390, 530, 1800, 650), "每频带 5 类形状统计", fill=PALE_BLUE, fnt=font(29, bold=True))
    arrow(d, (1595, 285), (1595, 335), fill=PINK)
    arrow(d, (1595, 480), (1595, 530), fill=PINK)
    rounded_box(d, (1450, 700, 1740, 790), "212 维物理描述", fill=PALE_GREEN, fnt=font(32, bold=True), outline=GREEN)
    arrow(d, (1595, 650), (1595, 700), fill=PINK)
    arrow(d, (600, 410), (665, 410), fill=DARK)
    arrow(d, (1260, 410), (1325, 410), fill=DARK)
    return save_image(img, "fig01_mechanism.png")


def create_descriptor_figure() -> Path:
    img = Image.new("RGB", (1900, 760), "white")
    d = ImageDraw.Draw(img)
    dashed_rect(d, (35, 55, 405, 700), fill=PALE_GRAY)
    dashed_rect(d, (455, 55, 910, 700), fill=PALE_BLUE)
    dashed_rect(d, (960, 55, 1430, 700), fill=PALE_PINK)
    dashed_rect(d, (1480, 55, 1865, 700), fill=PALE_GREEN)
    draw_centered(d, (220, 88), "信号处理", font(34, bold=True))
    draw_centered(d, (682, 88), "候选物理阶次", font(34, bold=True))
    draw_centered(d, (1195, 88), "描述量", font(34, bold=True))
    draw_centered(d, (1672, 88), "训练内标准化", font(34, bold=True))
    rounded_box(d, (90, 150, 350, 245), "48 kHz 振动窗口", fill="white", fnt=font(29, bold=True))
    rounded_box(d, (90, 310, 350, 420), "四频带解析包络", fill=PALE_LAV, fnt=font(30, bold=True))
    rounded_box(d, (90, 500, 350, 610), "重采样至 3 kHz\n去除边缘 25 ms", fill=PALE_PEACH, fnt=font(27, bold=True))
    arrow(d, (220, 245), (220, 310), fill=PINK)
    arrow(d, (220, 420), (220, 500), fill=PINK)
    orders = ["转频 fr", "保持架 FTF", "滚动体 BSF", "二倍滚动体 2BSF", "外圈 BPFO", "内圈 BPFI"]
    for i, text in enumerate(orders):
        y = 145 + i * 82
        rounded_box(d, (510, y, 855, y + 60), text, fill="white", fnt=font(25, bold=(i >= 4)), radius=10, width=2)
    d.text((682, 660), "阶次由当前记录转速与轴承几何参数确定", font=font(23), fill=GRAY, anchor="mm")
    desc = ["谐波强度、覆盖率、对比度", "周期重复性与重复对比度", "冲击相位集中度与相位对比度", "事件率", "顶部能量、峭度、峰值、RMS、平坦度"]
    colors = [PALE_BLUE, PALE_LAV, PALE_PINK, PALE_PEACH, (244, 244, 230)]
    for i, text in enumerate(desc):
        y = 150 + i * 101
        rounded_box(d, (1015, y, 1375, y + 72), text, fill=colors[i], fnt=font(24, bold=True), radius=12, width=2)
    rounded_box(d, (1525, 170, 1820, 285), "中位数中心化", fill="white", fnt=font(29, bold=True))
    rounded_box(d, (1525, 345, 1820, 460), "IQR 缩放\n尺度下限 0.05", fill="white", fnt=font(27, bold=True))
    rounded_box(d, (1525, 520, 1820, 635), "截断至 [-8, 8]", fill="white", fnt=font(29, bold=True))
    arrow(d, (1672, 285), (1672, 345), fill=PINK)
    arrow(d, (1672, 460), (1672, 520), fill=PINK)
    arrow(d, (405, 380), (455, 380))
    arrow(d, (910, 380), (960, 380))
    arrow(d, (1430, 380), (1480, 380))
    return save_image(img, "fig02_descriptor.png")


def create_architecture_figure() -> Path:
    img = Image.new("RGB", (2200, 970), "white")
    d = ImageDraw.Draw(img)
    dashed_rect(d, (30, 65, 345, 900), fill=PALE_GRAY)
    dashed_rect(d, (390, 65, 1200, 900), fill=PALE_BLUE)
    dashed_rect(d, (1245, 65, 1725, 900), fill=PALE_PINK)
    dashed_rect(d, (1770, 65, 2170, 900), fill=PALE_GREEN)
    draw_centered(d, (188, 100), "输入", font(38, bold=True))
    draw_centered(d, (795, 100), "双流表示学习", font(38, bold=True))
    draw_centered(d, (1485, 100), "门控残差融合", font(38, bold=True))
    draw_centered(d, (1970, 100), "共享监督", font(38, bold=True))
    rounded_box(d, (85, 270, 290, 395), "振动窗口\n24,000 点", fill="white", fnt=font(31, bold=True))
    rounded_box(d, (85, 575, 290, 700), "转速与\n轴承参数", fill=PALE_PEACH, fnt=font(31, bold=True))

    rounded_box(d, (450, 180, 750, 315), "数据编码器\n扩张残差卷积", fill=PALE_LAV, fnt=font(30, bold=True))
    rounded_box(d, (800, 180, 1125, 315), "均值与标准差池化\nLayerNorm", fill="white", fnt=font(27, bold=True))
    rounded_box(d, (845, 355, 1080, 445), "d ∈ R^32", fill=PALE_BLUE, fnt=font(30, bold=True), outline=BLUE)
    arrow(d, (290, 330), (450, 250))
    arrow(d, (750, 248), (800, 248))
    arrow(d, (962, 315), (962, 355), fill=BLUE)

    rounded_box(d, (450, 555, 750, 690), "物理描述提取\n212 维", fill=PALE_PEACH, fnt=font(30, bold=True))
    rounded_box(d, (800, 555, 1125, 690), "MLP 212→64→32\nLayerNorm / ReLU", fill="white", fnt=font(27, bold=True))
    rounded_box(d, (845, 735, 1080, 825), "p ∈ R^32", fill=PALE_PINK, fnt=font(30, bold=True), outline=RED)
    arrow(d, (290, 638), (450, 625))
    arrow(d, (750, 625), (800, 625))
    arrow(d, (962, 690), (962, 735), fill=RED)

    rounded_box(d, (1300, 250, 1660, 375), "g = sigmoid[MLP(d,p)]", fill="white", fnt=font(29, bold=True))
    rounded_box(d, (1300, 490, 1660, 630), "z = LN[d + g ⊙ p]", fill=PALE_LAV, fnt=font(34, bold=True), outline=PURPLE)
    d.text((1480, 705), "逐通道门控只决定物理表示的注入强度，\n不等同于因果贡献率", font=font(27), fill=PINK, anchor="mm", align="center")
    arrow(d, (1050, 400), (1300, 305), fill=BLUE)
    arrow(d, (1050, 780), (1300, 330), fill=RED)
    arrow(d, (1480, 375), (1480, 490), fill=PINK)

    rounded_box(d, (1825, 220, 2115, 340), "共享分类头\n32→32→4", fill="white", fnt=font(30, bold=True))
    rounded_box(d, (1825, 420, 2115, 535), "融合主损失\nCE(z, y)", fill=PALE_GREEN, fnt=font(29, bold=True), outline=GREEN)
    rounded_box(d, (1825, 610, 2115, 725), "数据辅助损失\nλ CE(d, y)", fill=PALE_PEACH, fnt=font(28, bold=True), outline=ORANGE)
    rounded_box(d, (1870, 790, 2070, 865), "λ = 0.3", fill=PALE_PINK, fnt=font(31, bold=True), outline=PINK)
    arrow(d, (1660, 560), (1825, 280), fill=PURPLE)
    arrow(d, (1970, 340), (1970, 420), fill=GREEN)
    arrow(d, (1125, 248), (1825, 668), fill=ORANGE)
    arrow(d, (1970, 725), (1970, 790), fill=PINK)
    return save_image(img, "fig03_architecture.png")


def create_experiment_flow_figure() -> Path:
    img = Image.new("RGB", (2000, 920), "white")
    d = ImageDraw.Draw(img)
    dashed_rect(d, (30, 60, 420, 860), fill=PALE_GRAY)
    dashed_rect(d, (465, 60, 1040, 860), fill=PALE_BLUE)
    dashed_rect(d, (1085, 60, 1530, 860), fill=PALE_PEACH)
    dashed_rect(d, (1575, 60, 1970, 860), fill=PALE_GREEN)
    draw_centered(d, (225, 98), "固定数据资源", font(36, bold=True))
    draw_centered(d, (752, 98), "两类划分", font(36, bold=True))
    draw_centered(d, (1307, 98), "训练与选择", font(36, bold=True))
    draw_centered(d, (1772, 98), "统一评价", font(36, bold=True))
    rounded_box(d, (85, 205, 365, 330), "CWRU 40 条记录\n687 个不重叠窗口", fill="white", fnt=font(29, bold=True))
    rounded_box(d, (85, 420, 365, 545), "4 类状态\nN / IR / OR / B", fill=PALE_LAV, fnt=font(30, bold=True))
    rounded_box(d, (85, 635, 365, 755), "4 个负载\n0 / 1 / 2 / 3 HP", fill=PALE_PEACH, fnt=font(30, bold=True))
    arrow(d, (225, 330), (225, 420), fill=PINK)
    arrow(d, (225, 545), (225, 635), fill=PINK)
    rounded_box(d, (515, 170, 990, 330), "记录内随机窗口识别\n481 训练 / 103 验证 / 103 测试", fill="white", fnt=font(30, bold=True), outline=BLUE)
    rounded_box(d, (515, 500, 990, 680), "单一来源负载直接迁移\n来源训练与验证，其他三个负载仅测试\n共 12 个迁移方向", fill="white", fnt=font(29, bold=True), outline=RED)
    d.text((752, 770), "两种划分回答不同问题，不能互相替代", font=font(27), fill=PINK, anchor="mm")
    arrow(d, (420, 450), (515, 250))
    arrow(d, (420, 500), (515, 585))
    rounded_box(d, (1135, 160, 1480, 280), "预算\n10% / 20% / 50% / 全量", fill="white", fnt=font(27, bold=True))
    rounded_box(d, (1135, 340, 1480, 460), "协议\n纯干净 / AWGN 混合", fill=PALE_LAV, fnt=font(27, bold=True))
    rounded_box(d, (1135, 520, 1480, 640), "重复\n3 子集 × 5 训练种子", fill="white", fnt=font(27, bold=True))
    rounded_box(d, (1135, 700, 1480, 810), "验证选模\n最大 200 epoch", fill=PALE_PINK, fnt=font(27, bold=True))
    for y in [280, 460, 640]:
        arrow(d, (1307, y), (1307, y + 60), fill=PINK)
    arrow(d, (1040, 440), (1085, 440))
    rounded_box(d, (1620, 170, 1925, 300), "测试条件\n干净及 -4 至 +4 dB", fill="white", fnt=font(28, bold=True))
    rounded_box(d, (1620, 375, 1925, 505), "指标\nACC / BA / Macro-F1\n类别召回率", fill=PALE_BLUE, fnt=font(27, bold=True))
    rounded_box(d, (1620, 580, 1925, 710), "报告层级\n整体、方向、类别、成本", fill="white", fnt=font(27, bold=True))
    d.text((1772, 795), "全部模型锁定后统一测试", font=font(29, bold=True), fill=PINK, anchor="mm")
    arrow(d, (1530, 440), (1575, 440))
    return save_image(img, "fig04_experiment_flow.png")


def create_selection_figure() -> Path:
    data = json.loads(SEL_JSON.read_text(encoding="utf-8"))["candidates"]
    lambdas = [float(x["aux_weight"]) for x in data]
    xs = np.arange(len(lambdas), dtype=float)
    ys = np.array([float(x["mean_best_validation_ce"]) * 1e6 for x in data])
    es = np.array([float(x["sample_sd"]) * 1e6 for x in data])
    img = Image.new("RGBA", (1350, 820), "white")
    d = ImageDraw.Draw(img)
    tx, ty = axes(
        img,
        d,
        (170, 95, 1240, 650),
        (-0.35, 4.35),
        (4.2, 10.1),
        xs,
        [5, 6, 7, 8, 9, 10],
        "辅助损失系数 λ",
        "最佳验证交叉熵（×10⁻⁶）",
        xlabels=["0.01", "0.03", "0.10", "0.30", "1.00"],
    )
    points = [(tx(x), ty(y)) for x, y in zip(xs, ys)]
    d.line(points, fill=BLUE, width=5)
    for x, y, e in zip(xs, ys, es):
        px, py = tx(x), ty(y)
        d.line([(px, ty(y - e)), (px, ty(y + e))], fill=DARK, width=3)
        d.line([(px - 10, ty(y - e)), (px + 10, ty(y - e))], fill=DARK, width=3)
        d.line([(px - 10, ty(y + e)), (px + 10, ty(y + e))], fill=DARK, width=3)
        fill = RED if abs(lambdas[int(x)] - 0.3) < 1e-9 else BLUE
        d.ellipse((px - 10, py - 10, px + 10, py + 10), fill=fill, outline="white", width=2)
        d.text((px, py - 20), f"{y:.2f}", font=font(23, en=True), fill=fill, anchor="mb")
    d.text((tx(3), ty(5.702) + 35), "按预定均值规则选定", font=font(24, bold=True), fill=RED, anchor="mt")
    d.text((695, 750), "误差线为 5 个训练种子的样本标准差", font=font(24), fill=GRAY, anchor="mm")
    return save_image(img.convert("RGB"), "fig05_lambda_selection.png")


def create_training_figure() -> Path:
    b = pd.read_csv(SMALL_DIR / "models" / "m0011" / "training.csv")
    f = pd.read_csv(SMALL_DIR / "models" / "m0016" / "training.csv")
    img = Image.new("RGBA", (1900, 900), "white")
    d = ImageDraw.Draw(img)
    panels = [
        ((150, 75, 900, 355), b, "数据单分支损失", "loss"),
        ((1080, 75, 1830, 355), f, "融合方案损失", "loss"),
        ((150, 505, 900, 785), b, "数据单分支准确率", "acc"),
        ((1080, 505, 1830, 785), f, "融合方案准确率", "acc"),
    ]
    for rect, df, title, kind in panels:
        epochs = df["epoch"].to_numpy(float)
        if kind == "loss":
            ymax = max(1.5, float(np.nanmax([df["train_loss"].max(), df["validation_nll"].max()])) * 1.05)
            tx, ty = axes(img, d, rect, (1, max(epochs)), (0, ymax), np.linspace(1, max(epochs), 5), np.linspace(0, ymax, 5), "Epoch", "Loss", title=title, y_fmt=lambda v: f"{v:.1f}")
            s1 = list(zip([tx(x) for x in epochs], [ty(y) for y in df["train_loss"].to_numpy(float)]))
            s2 = list(zip([tx(x) for x in epochs], [ty(y) for y in df["validation_nll"].to_numpy(float)]))
            d.line(s1, fill=BLUE, width=4)
            d.line(s2, fill=RED, width=4)
            legend(d, (rect[2] - 250, rect[1] + 20), [("训练损失", BLUE), ("验证 NLL", RED)], width=225)
        else:
            tx, ty = axes(img, d, rect, (1, max(epochs)), (0, 1.02), np.linspace(1, max(epochs), 5), [0, .25, .5, .75, 1], "Epoch", "Accuracy / BA", title=title, y_fmt=lambda v: f"{v:.2f}")
            train = df["train_acc_sampled_augmented"].to_numpy(float)
            valc = df["validation_clean_balanced_acc"].to_numpy(float)
            valn = df["validation_white_-4_balanced_acc"].to_numpy(float)
            d.line(list(zip([tx(x) for x in epochs], [ty(y) for y in train])), fill=BLUE, width=4)
            d.line(list(zip([tx(x) for x in epochs], [ty(y) for y in valc])), fill=GREEN, width=4)
            d.line(list(zip([tx(x) for x in epochs], [ty(y) for y in valn])), fill=RED, width=4)
            legend(d, (rect[2] - 270, rect[1] + 20), [("训练 ACC", BLUE), ("验证干净 BA", GREEN), ("验证 -4 dB BA", RED)], width=250)
    d.text((950, 442), "(a) 损失曲线", font=font(27), fill=DARK, anchor="mm")
    d.text((950, 875), "(b) 训练与验证性能", font=font(27), fill=DARK, anchor="mm")
    return save_image(img.convert("RGB"), "fig06_training_curves.png")


def _error_line_chart(
    name: str,
    rows: Sequence[tuple[str, Sequence[float], Sequence[float], tuple[int, int, int]]],
    *,
    xlabels: Sequence[str],
    ylabel: str,
    ylim: tuple[float, float],
    note: str,
    title: str = "",
) -> Path:
    img = Image.new("RGBA", (1450, 870), "white")
    d = ImageDraw.Draw(img)
    xs = np.arange(len(xlabels), dtype=float)
    yticks = np.linspace(ylim[0], ylim[1], 6)
    tx, ty = axes(img, d, (170, 90, 1335, 680), (-0.25, len(xs) - 0.75), ylim, xs, yticks, "训练样本比例", ylabel, title=title, xlabels=xlabels, y_fmt=lambda v: f"{v:.0f}")
    for label, values, errors, color in rows:
        pts = []
        for x, y, e in zip(xs, values, errors):
            px, py = tx(x), ty(y)
            pts.append((px, py))
            d.line([(px, ty(y - e)), (px, ty(y + e))], fill=color, width=3)
            d.line([(px - 9, ty(y - e)), (px + 9, ty(y - e))], fill=color, width=3)
            d.line([(px - 9, ty(y + e)), (px + 9, ty(y + e))], fill=color, width=3)
            d.ellipse((px - 9, py - 9, px + 9, py + 9), fill=color, outline="white", width=2)
            d.text((px, py - 17), f"{y:.2f}", font=font(22, en=True), fill=color, anchor="mb")
        d.line(pts, fill=color, width=5)
    legend(d, (1010, 112), [(r[0], r[3]) for r in rows], width=300)
    d.text((725, 790), note, font=font(23), fill=GRAY, anchor="mm")
    return save_image(img.convert("RGB"), name)


def create_random_window_figure() -> Path:
    df = pd.read_csv(WIN_DIR / "evaluation" / "between_subset_summary.csv")
    sub = df[(df.protocol == "awgn_mixed") & (df.condition == "white_-4") & (df.ratio.isin([.1, .2, .5]))]
    rows = []
    for variant, label, color in [("baseline_data", "数据单分支", BLUE), ("data_aux_ce", "融合方案", RED)]:
        x = sub[sub.variant == variant].sort_values("ratio")
        rows.append((label, x.balanced_acc_mean.to_numpy() * 100, x.balanced_acc_std.to_numpy() * 100, color))
    return _error_line_chart(
        "fig07_random_small.png",
        rows,
        xlabels=["10%", "20%", "50%"],
        ylabel="均衡准确率 BA（%）",
        ylim=(70, 102),
        note="记录内随机窗口划分，-4 dB AWGN；误差线为 3 个固定子集均值之间的样本标准差",
    )


def create_full_transfer_bar() -> Path:
    df = pd.read_csv(FULL_DIR / "evaluation" / "transfer_equal_direction_mean_std.csv")
    specs = [("clean_only", "clean", "纯干净 / 干净"), ("awgn_mixed", "clean", "AWGN / 干净"), ("awgn_mixed", "white_-4", "AWGN / -4 dB")]
    labels = [s[2] for s in specs]
    base, base_e, fus, fus_e = [], [], [], []
    for protocol, cond, _ in specs:
        for variant, vals, errs in [("baseline_data", base, base_e), ("data_aux_ce", fus, fus_e)]:
            r = df[(df.protocol == protocol) & (df.condition == cond) & (df.variant == variant)].iloc[0]
            vals.append(float(r.balanced_acc_mean) * 100)
            errs.append(float(r.balanced_acc_std) * 100)
    img = Image.new("RGBA", (1550, 760), "white")
    d = ImageDraw.Draw(img)
    xs = np.arange(3)
    tx, ty = axes(img, d, (170, 75, 1430, 565), (-0.6, 2.6), (75, 100), xs, [75, 80, 85, 90, 95, 100], "训练 / 测试条件", "12 方向等权 BA（%）", xlabels=labels, y_fmt=lambda v: f"{v:.0f}")
    w = 0.28
    for vals, errs, off, color, label in [(base, base_e, -w / 1.6, BLUE, "数据单分支"), (fus, fus_e, w / 1.6, ORANGE, "融合方案")]:
        for i, (v, e) in enumerate(zip(vals, errs)):
            left, right = tx(i + off - w / 2), tx(i + off + w / 2)
            d.rectangle((left, ty(v), right, ty(75)), fill=color, outline=DARK, width=2)
            cx = (left + right) / 2
            d.line([(cx, ty(v - e)), (cx, ty(v + e))], fill=DARK, width=3)
            d.line([(cx - 9, ty(v - e)), (cx + 9, ty(v - e))], fill=DARK, width=3)
            d.line([(cx - 9, ty(v + e)), (cx + 9, ty(v + e))], fill=DARK, width=3)
            d.text((cx, ty(v + e) - 8), f"{v:.2f}±{e:.2f}", font=font(21, en=True), fill=DARK, anchor="mb")
    legend(d, (1080, 115), [("数据单分支", BLUE), ("融合方案", ORANGE)], width=305)
    d.text((775, 710), "均值和误差分别对应 5 个训练种子；每个种子内先对 12 个方向等权", font=font(23), fill=GRAY, anchor="mm")
    return save_image(img.convert("RGB"), "fig08_full_transfer.png")


def create_direction_heatmap() -> Path:
    df = pd.read_csv(FULL_DIR / "evaluation" / "paired_mean_std.csv")
    sub = df[(df.protocol == "awgn_mixed") & (df.condition == "white_-4")]
    mat = np.full((4, 4), np.nan)
    for _, r in sub.iterrows():
        mat[int(r.source_load_hp), int(r.target_load_hp)] = float(r.balanced_acc_mean) * 100
    img = Image.new("RGB", (1200, 970), "white")
    d = ImageDraw.Draw(img)
    x0, y0, cell = 250, 120, 160
    f = font(29, bold=True)
    d.text((570, 55), "融合方案相对数据单分支的 BA 差值", font=font(36, bold=True), fill=DARK, anchor="mm")
    for i in range(4):
        d.text((x0 - 30, y0 + i * cell + cell / 2), f"{i} HP", font=f, fill=DARK, anchor="rm")
        d.text((x0 + i * cell + cell / 2, y0 - 22), f"{i} HP", font=f, fill=DARK, anchor="mb")
    for i in range(4):
        for j in range(4):
            box = (x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell, y0 + (i + 1) * cell)
            if i == j or np.isnan(mat[i, j]):
                d.rectangle(box, fill=(244, 244, 244), outline="white", width=4)
                d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), "—", font=f, fill=GRAY, anchor="mm")
            else:
                v = mat[i, j]
                if v >= 0:
                    color = tuple(int(255 - (255 - c) * min(v / 18, 1)) for c in BLUE)
                else:
                    color = tuple(int(255 - (255 - c) * min(abs(v) / 4, 1)) for c in RED)
                d.rectangle(box, fill=color, outline="white", width=4)
                text_fill = "white" if abs(v) > 7 else DARK
                d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), f"{v:+.2f}", font=f, fill=text_fill, anchor="mm")
    d.text((570, 800), "目标负载", font=font(32, bold=True), fill=DARK, anchor="mm")
    rotated_text(img.convert("RGBA"), (120, 440), "来源负载", font(32, bold=True), DARK, 90)
    # Redraw the label directly on RGB to keep compatibility.
    lab = Image.new("RGBA", (380, 90), (255, 255, 255, 0))
    ld = ImageDraw.Draw(lab)
    ld.text((190, 45), "来源负载", font=font(32, bold=True), fill=DARK, anchor="mm")
    lab = lab.rotate(90, expand=True)
    img.paste(lab, (55, 250), lab)
    d = ImageDraw.Draw(img)
    d.text((570, 880), "AWGN 训练，-4 dB 测试；正值表示融合方案更高，负值表示局部退化", font=font(25), fill=GRAY, anchor="mm")
    return save_image(img, "fig09_direction_heatmap.png")


def create_small_transfer_figure() -> Path:
    df = pd.read_csv(SMALL_DIR / "evaluation" / "transfer_between_subset_summary.csv")
    sub = df[(df.protocol == "awgn_mixed") & (df.condition == "white_-4")]
    rows = []
    for variant, label, color in [("baseline_data", "数据单分支", BLUE), ("data_aux_ce", "融合方案", RED)]:
        x = sub[sub.variant == variant].sort_values("ratio")
        rows.append((label, x.balanced_acc_mean.to_numpy() * 100, x.balanced_acc_std.to_numpy() * 100, color))
    return _error_line_chart(
        "fig10_small_transfer.png",
        rows,
        xlabels=["10%", "20%", "50%"],
        ylabel="12 方向等权 BA（%）",
        ylim=(25, 100),
        note="单一来源小样本直接迁移，-4 dB AWGN；误差线为 3 个固定子集均值之间的样本标准差",
    )


def create_recall_figure() -> Path:
    budgets = ["10%", "20%", "50%"]
    baseline = np.array([[24.90, 26.60, 76.33, 16.87], [68.03, 62.35, 85.94, 38.30], [87.28, 83.09, 91.15, 55.14]])
    fusion = np.array([[99.15, 94.73, 80.84, 69.13], [99.94, 94.73, 84.20, 71.78], [99.62, 96.58, 91.30, 68.67]])
    img = Image.new("RGBA", (1750, 1000), "white")
    d = ImageDraw.Draw(img)
    colors = [BLUE, GREEN, ORANGE, RED]
    labels = ["N", "IR", "OR", "B"]
    panels = [(120, 105, 825, 825, baseline, "数据单分支"), (1010, 105, 1715, 825, fusion, "融合方案")]
    for x0, y0, x1, y1, arr, title in panels:
        tx, ty = axes(img, d, (x0, y0, x1, y1), (-0.6, 2.6), (0, 105), [0, 1, 2], [0, 20, 40, 60, 80, 100], "训练样本比例", "类别召回率（%）", xlabels=budgets, title=title, y_fmt=lambda v: f"{v:.0f}")
        w = 0.15
        for c in range(4):
            off = (c - 1.5) * w
            for i in range(3):
                left, right = tx(i + off - w * .45), tx(i + off + w * .45)
                d.rectangle((left, ty(arr[i, c]), right, ty(0)), fill=colors[c], outline=DARK, width=1)
    d.rectangle((545, 850, 1205, 915), fill="white", outline=(80, 80, 80), width=2)
    for i, (label, color) in enumerate(zip(labels, colors)):
        xx = 575 + i * 158
        d.rectangle((xx, 873, xx + 30, 893), fill=color, outline=DARK, width=1)
        d.text((xx + 42, 883), label, font=font(24, bold=True, en=True), fill=DARK, anchor="lm")
    d.text((875, 963), "AWGN 训练，-4 dB 测试；类别均值先按方向等权汇总", font=font(24), fill=GRAY, anchor="mm")
    return save_image(img.convert("RGB"), "fig11_class_recall.png")


def create_confusion_figure() -> Path:
    df = pd.read_csv(FULL_DIR / "evaluation" / "pooled_confusion_matrices.csv")
    sub = df[(df.source_load_hp == 2) & (df.target_load_hp == 1) & (df.protocol == "awgn_mixed") & (df.condition == "white_-4")]
    labels = ["N", "IR", "OR", "B"]
    img = Image.new("RGB", (1700, 920), "white")
    d = ImageDraw.Draw(img)
    panels = [(80, "baseline_data", "数据单分支"), (900, "data_aux_ce", "融合方案")]
    for xoff, variant, title in panels:
        part = sub[sub.variant == variant]
        counts = np.zeros((4, 4), dtype=int)
        pct = np.zeros((4, 4), dtype=float)
        for _, r in part.iterrows():
            i, j = int(r.true_label), int(r.predicted_label)
            counts[i, j] = int(r["count"])
            pct[i, j] = float(r.row_percent)
        x0, y0, cell = xoff + 140, 170, 130
        acc = counts.trace() / max(counts.sum(), 1) * 100
        d.text((x0 + 2 * cell, 75), title, font=font(35, bold=True), fill=DARK, anchor="mm")
        d.text((x0 + 2 * cell, 115), f"合并预测 ACC = {acc:.2f}%", font=font(26), fill=GRAY, anchor="mm")
        for i in range(4):
            d.text((x0 - 18, y0 + i * cell + cell / 2), labels[i], font=font(28, bold=True), fill=DARK, anchor="rm")
            d.text((x0 + i * cell + cell / 2, y0 - 15), labels[i], font=font(28, bold=True), fill=DARK, anchor="mb")
        for i in range(4):
            for j in range(4):
                box = (x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell, y0 + (i + 1) * cell)
                col = blues(pct[i, j] / 100)
                d.rectangle(box, fill=col, outline="white", width=4)
                tf = "white" if pct[i, j] > 58 else DARK
                d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2 - 15), f"{counts[i,j]}", font=font(25, bold=True, en=True), fill=tf, anchor="mm")
                d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2 + 22), f"{pct[i,j]:.1f}%", font=font(22, en=True), fill=tf, anchor="mm")
        d.text((x0 + 2 * cell, y0 + 4 * cell + 55), "预测类别", font=font(29, bold=True), fill=DARK, anchor="mm")
        lab = Image.new("RGBA", (300, 80), (255, 255, 255, 0))
        ld = ImageDraw.Draw(lab)
        ld.text((150, 40), "真实类别", font=font(29, bold=True), fill=DARK, anchor="mm")
        lab = lab.rotate(90, expand=True)
        img.paste(lab, (xoff + 20, y0 + 95), lab)
    d.text((850, 875), "2 HP→1 HP，AWGN 训练，-4 dB；数值为合并计数与行比例，重复预测不构成新增独立样本", font=font(24), fill=GRAY, anchor="mm")
    return save_image(img, "fig12_confusion.png")


def create_cost_figure() -> Path:
    # Reported means are stable across three result stages. Use the small-transfer stage for a unified comparison.
    df = pd.read_csv(SMALL_DIR / "computational_cost.csv")
    g = df.groupby("variant", as_index=True).agg(parameters=("parameters", "mean"), latency=("latency_ms", "mean"), memory=("peak_gpu_mib", "mean"))
    variants = ["baseline_data", "data_aux_ce"]
    params = [g.loc[v, "parameters"] / 1000 for v in variants]
    latency = [g.loc[v, "latency"] for v in variants]
    img = Image.new("RGB", (1450, 850), "white")
    d = ImageDraw.Draw(img)
    rect = (180, 80, 1280, 670)
    tx, ty1 = axes(img.convert("RGBA"), d, rect, (-0.7, 1.7), (0, 60), [0, 1], [0, 10, 20, 30, 40, 50, 60], "方法", "", xlabels=["数据单分支", "融合方案"], y_fmt=lambda v: f"{v:.0f}")
    x0, y0, x1, y1 = rect
    # Right axis.
    def ty2(v):
        return y1 - v / 9 * (y1 - y0)
    for v in [0, 1.5, 3, 4.5, 6, 7.5, 9]:
        d.text((x1 + 15, ty2(v)), f"{v:g}", font=font(23, en=True), fill=RED, anchor="lm")
    lab = Image.new("RGBA", (430, 90), (255, 255, 255, 0))
    ld = ImageDraw.Draw(lab)
    ld.text((215, 45), "单样本推理延迟（ms）", font=font(28, bold=True), fill=RED, anchor="mm")
    lab = lab.rotate(270, expand=True)
    img.paste(lab, (1320, 160), lab)
    left_lab = Image.new("RGBA", (330, 90), (255, 255, 255, 0))
    lld = ImageDraw.Draw(left_lab)
    lld.text((165, 45), "参数量（千）", font=font(28, bold=True), fill=BLUE, anchor="mm")
    left_lab = left_lab.rotate(90, expand=True)
    img.paste(left_lab, (28, 205), left_lab)
    for i in range(2):
        c = tx(i)
        d.rectangle((c - 105, ty1(params[i]), c - 10, ty1(0)), fill=BLUE, outline=DARK, width=2)
        d.rectangle((c + 10, ty2(latency[i]), c + 105, ty2(0)), fill=ORANGE, outline=DARK, width=2)
        d.text((c - 57, ty1(params[i]) - 8), f"{params[i]:.1f}k", font=font(24, en=True), fill=BLUE, anchor="mb")
        d.text((c + 57, ty2(latency[i]) - 8), f"{latency[i]:.2f}", font=font(24, en=True), fill=RED, anchor="mb")
    legend(d, (880, 110), [("参数量", BLUE), ("实测延迟", ORANGE)], width=300)
    d.text((725, 835), "融合方案峰值显存 1343.52 MiB，数据单分支 1195.58 MiB；延迟包含在线物理特征提取", font=font(23), fill=GRAY, anchor="mm")
    return save_image(img, "fig13_cost.png")


def set_cell_border(cell, **kwargs) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_borders = tc_pr.first_child_found_in("w:tcBorders")
    if tc_borders is None:
        tc_borders = OxmlElement("w:tcBorders")
        tc_pr.append(tc_borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        if edge in kwargs:
            tag = "w:" + edge
            element = tc_borders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tc_borders.append(element)
            for key, value in kwargs[edge].items():
                element.set(qn("w:" + key), str(value))


def set_cell_margins(cell, top=55, start=65, bottom=55, end=65) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_row_cant_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit")
    tr_pr.append(cant)


def set_east_asia_font(run, name: str) -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def configure_section(section, *, columns: int = 1) -> None:
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.58)
    section.left_margin = Inches(0.68)
    section.right_margin = Inches(0.68)
    section.header_distance = Inches(0.22)
    section.footer_distance = Inches(0.25)
    sect_pr = section._sectPr
    cols = sect_pr.xpath("./w:cols")
    if cols:
        col = cols[0]
    else:
        col = OxmlElement("w:cols")
        sect_pr.append(col)
    col.set(qn("w:num"), str(columns))
    col.set(qn("w:space"), "320")
    if columns == 2:
        col.set(qn("w:equalWidth"), "1")


def add_columns_section(doc: Document, columns: int) -> None:
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    configure_section(section, columns=columns)


def set_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)
    run.font.size = Pt(8)


def setup_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(9.5)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    normal.paragraph_format.space_after = Pt(1.5)
    normal.paragraph_format.first_line_indent = Pt(18)

    for style_name, size, bold, italic, align, before, after in [
        ("Title", 20.5, True, False, WD_ALIGN_PARAGRAPH.CENTER, 0, 7),
        ("Heading 1", 10.5, True, False, WD_ALIGN_PARAGRAPH.CENTER, 9, 4),
        ("Heading 2", 10, True, True, WD_ALIGN_PARAGRAPH.LEFT, 6, 2),
        ("Heading 3", 9.5, True, True, WD_ALIGN_PARAGRAPH.LEFT, 4, 1),
    ]:
        s = styles[style_name]
        s.font.name = "Times New Roman"
        s._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        s.font.size = Pt(size)
        s.font.bold = bold
        s.font.italic = italic
        s.font.color.rgb = RGBColor(0, 0, 0)
        s.paragraph_format.alignment = align
        s.paragraph_format.space_before = Pt(before)
        s.paragraph_format.space_after = Pt(after)
        s.paragraph_format.keep_with_next = True
        ppr = s._element.get_or_add_pPr()
        p_bdr = ppr.find(qn("w:pBdr"))
        if p_bdr is not None:
            ppr.remove(p_bdr)

    for name in ["Caption Figure", "Caption Table", "Abstract Text", "Keywords", "Reference"]:
        if name not in styles:
            styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    cap = styles["Caption Figure"]
    cap.font.name = "Times New Roman"
    cap._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    cap.font.size = Pt(8)
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cap.paragraph_format.first_line_indent = Pt(0)
    cap.paragraph_format.space_before = Pt(1)
    cap.paragraph_format.space_after = Pt(4)
    cap.paragraph_format.keep_with_previous = True

    capt = styles["Caption Table"]
    capt.font.name = "Times New Roman"
    capt._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    capt.font.size = Pt(8)
    capt.font.bold = False
    capt.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    capt.paragraph_format.first_line_indent = Pt(0)
    capt.paragraph_format.space_before = Pt(4)
    capt.paragraph_format.space_after = Pt(1)
    capt.paragraph_format.keep_with_next = True

    abst = styles["Abstract Text"]
    abst.font.name = "Times New Roman"
    abst._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    abst.font.size = Pt(9)
    abst.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    abst.paragraph_format.first_line_indent = Pt(0)
    abst.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    abst.paragraph_format.space_after = Pt(3)

    kw = styles["Keywords"]
    kw.font.name = "Times New Roman"
    kw._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    kw.font.size = Pt(9)
    kw.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    kw.paragraph_format.first_line_indent = Pt(0)
    kw.paragraph_format.space_after = Pt(4)

    ref = styles["Reference"]
    ref.font.name = "Times New Roman"
    ref._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    ref.font.size = Pt(7.2)
    ref.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    ref.paragraph_format.first_line_indent = Pt(-15)
    ref.paragraph_format.left_indent = Pt(15)
    ref.paragraph_format.space_after = Pt(0)


def add_body(doc: Document, text: str, *, bold_prefix: str | None = None, no_indent: bool = False) -> None:
    p = doc.add_paragraph()
    if no_indent:
        p.paragraph_format.first_line_indent = Pt(0)
    if bold_prefix and text.startswith(bold_prefix):
        r1 = p.add_run(bold_prefix)
        r1.bold = True
        set_east_asia_font(r1, "黑体")
        r2 = p.add_run(text[len(bold_prefix):])
        set_east_asia_font(r2, "宋体")
    else:
        r = p.add_run(text)
        set_east_asia_font(r, "宋体")


def add_heading(doc: Document, text: str, level: int) -> None:
    p = doc.add_heading(text, level=level)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.keep_together = True


def add_equation(doc: Document, equation: str, number: str) -> None:
    table = doc.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(0.22)
    table.columns[1].width = Inches(2.75)
    table.columns[2].width = Inches(0.35)
    for cell in table.rows[0].cells:
        set_cell_border(cell, top={"val": "nil"}, bottom={"val": "nil"}, left={"val": "nil"}, right={"val": "nil"})
        set_cell_margins(cell, top=0, bottom=0, start=0, end=0)
    p = table.cell(0, 1).paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    math_para = OxmlElement("m:oMathPara")
    math_obj = OxmlElement("m:oMath")
    mr = OxmlElement("m:r")
    mrpr = OxmlElement("m:rPr")
    mr.append(mrpr)
    mt = OxmlElement("m:t")
    mt.text = equation
    mr.append(mt)
    math_obj.append(mr)
    math_para.append(math_obj)
    p._p.append(math_para)
    pn = table.cell(0, 2).paragraphs[0]
    pn.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    pn.paragraph_format.first_line_indent = Pt(0)
    rn = pn.add_run(number)
    rn.font.size = Pt(9)
    set_east_asia_font(rn, "Times New Roman")


def add_figure(doc: Document, path: Path, caption: str, *, width: float, center=True) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run()
    r.add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph(caption, style="Caption Figure")
    cap.paragraph_format.keep_with_previous = True


def add_wide_figure(doc: Document, path: Path, caption: str, *, width: float = 7.0) -> None:
    add_columns_section(doc, 1)
    add_figure(doc, path, caption, width=width)
    add_columns_section(doc, 2)


def add_table_caption(doc: Document, number: int, title: str) -> None:
    p = doc.add_paragraph(style="Caption Table")
    p.add_run(f"表 {number}\n").bold = False
    r = p.add_run(title)
    r.bold = False
    set_east_asia_font(r, "宋体")


def add_three_line_table(
    doc: Document,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    widths: Sequence[float] | None = None,
    font_size: float = 7.4,
    bold_best: set[tuple[int, int]] | None = None,
) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = widths is None
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    set_repeat_table_header(table.rows[0])
    if widths:
        for j, w in enumerate(widths):
            table.columns[j].width = Inches(w)
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_margins(cell)
        set_cell_border(cell, top={"val": "single", "sz": "12", "color": "000000"}, bottom={"val": "single", "sz": "6", "color": "000000"}, left={"val": "nil"}, right={"val": "nil"})
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(str(h))
        r.bold = True
        r.font.size = Pt(font_size)
        set_east_asia_font(r, "宋体")
    for i, row_data in enumerate(rows):
        row = table.add_row()
        set_row_cant_split(row)
        for j, value in enumerate(row_data):
            cell = row.cells[j]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            bottom = {"val": "single", "sz": "12", "color": "000000"} if i == len(rows) - 1 else {"val": "nil"}
            set_cell_border(cell, top={"val": "nil"}, bottom=bottom, left={"val": "nil"}, right={"val": "nil"})
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.keep_with_next = i < len(rows) - 1
            r = p.add_run(str(value))
            r.font.size = Pt(font_size)
            r.bold = bool(bold_best and (i, j) in bold_best)
            set_east_asia_font(r, "宋体")


def add_wide_table(
    doc: Document,
    number: int,
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    widths: Sequence[float] | None = None,
    font_size: float = 7.4,
    bold_best: set[tuple[int, int]] | None = None,
) -> None:
    add_columns_section(doc, 1)
    add_table_caption(doc, number, title)
    add_three_line_table(doc, headers, rows, widths=widths, font_size=font_size, bold_best=bold_best)
    add_columns_section(doc, 2)


def build_document(figs: dict[str, Path]) -> Path:
    doc = Document()
    configure_section(doc.sections[0], columns=1)
    setup_styles(doc)

    # Header and footer are intentionally restrained. IEEE publication notices from the reference are not reproduced.
    header = doc.sections[0].header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    header.paragraph_format.first_line_indent = Pt(0)
    hr = header.add_run("面向噪声与跨负载小样本场景的物理引导轴承故障诊断")
    hr.font.size = Pt(7)
    hr.font.color.rgb = RGBColor(110, 110, 110)
    set_east_asia_font(hr, "宋体")
    set_page_number(doc.sections[0].footer.paragraphs[0])

    title = doc.add_paragraph(style="Title")
    title.add_run("面向噪声与跨负载小样本场景的\n物理引导双流门控融合轴承故障诊断方法")
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.paragraph_format.first_line_indent = Pt(0)
    sub.paragraph_format.space_after = Pt(7)
    sr = sub.add_run("基于 CWRU 固定划分与 V3 完整实验结果")
    sr.font.size = Pt(10)
    set_east_asia_font(sr, "宋体")

    p = doc.add_paragraph(style="Abstract Text")
    r = p.add_run("摘  要—")
    r.bold = True
    set_east_asia_font(r, "黑体")
    p.add_run(
        "针对滚动轴承故障诊断中训练样本不足、加性噪声和负载变化同时存在时数据驱动模型稳定性下降的问题，"
        "本文提出一种由原始振动数据表示和确定性物理描述共同构成的双流门控融合模型。数据分支采用扩张残差卷积编码原始振动窗口。"
        "物理分支围绕转频、保持架、滚动体、外圈和内圈候选阶次，在四个包络频带内提取谐波、周期、相位和形状信息，形成212维描述，"
        "并仅用当前训练子集拟合稳健标准化参数。两类32维表示通过逐通道门控以残差形式融合。共享分类头同时承担融合主任务和数据分支辅助任务，"
        "辅助损失系数按五个训练种子的验证交叉熵预先锁定为0.3。实验使用CWRU全部40条记录形成的687个不重叠窗口，分别考察记录内随机窗口识别、"
        "全样本单源跨负载迁移和小样本单源跨负载迁移。V3各阶段共训练1020个新模型。结果表明，在随机窗口10%训练预算和-4 dB条件下，"
        "融合方案的均衡准确率为94.72%，数据单分支为82.66%。在全样本12方向跨负载评价中，两者为89.51%和84.31%。"
        "在10%小样本跨负载评价中，两者为85.96%和36.17%。优势主要集中在低样本和分布变化同时存在的条件。"
        "当随机窗口训练预算达到20%以上时，性能接近饱和，融合增益明显收缩；2 HP到1 HP方向及外圈故障召回仍出现局部退化。"
        "因此，现有证据支持完整融合方案在本数据划分下提高低样本跨负载诊断的经验稳定性，但不能单独识别物理分支、转速信息、门控结构和辅助损失各自的因果贡献。"
    )
    for rr in p.runs[1:]:
        set_east_asia_font(rr, "宋体")
    kw = doc.add_paragraph(style="Keywords")
    kr = kw.add_run("关键词—")
    kr.bold = True
    set_east_asia_font(kr, "黑体")
    k2 = kw.add_run("滚动轴承；物理引导学习；小样本故障诊断；跨负载迁移；门控融合；噪声鲁棒性")
    set_east_asia_font(k2, "宋体")

    add_columns_section(doc, 2)

    add_heading(doc, "1  研究背景和引言", 1)
    add_body(doc, "滚动轴承是旋转机械中承担载荷传递和运动约束的关键部件。局部剥落、点蚀和裂纹会诱发重复冲击，并可能进一步造成停机或连带损伤。有效诊断不仅要求识别正常、内圈、外圈和滚动体状态，还要求模型在噪声、负载变化和有限标注条件下保持可用。后一个要求比固定工况上的高准确率更接近实际维护场景，也更难从常用实验数据中得到充分证明。")
    add_body(doc, "传统轴承诊断依据故障特征频率、包络谱、谱峭度及统计量建立判据。这类方法的物理含义较清晰，但效果依赖频带选择、转速信息和人工阈值。深度学习将特征提取与分类统一起来，减少了手工设计。然而，当训练和测试来自相同记录时，模型可能同时利用故障模式、记录背景和采集条件。高准确率只能说明在该划分下具有较强判别能力，不能自动证明模型获得了可迁移的故障规律[1]-[4]。")
    add_body(doc, "在当前研究对象中，三个困难同时存在。第一，极低样本预算下每类故障的覆盖有限。例如0 HP来源的10%跨负载训练集只有9个窗口，其中正常类只有1个窗口。第二，AWGN会降低冲击和包络调制的可辨识度。第三，负载变化会改变幅值、转速和背景成分，使原始波形分布发生偏移。仅在记录内随机窗口划分上评价，无法回答单一负载模型能否直接迁移到其他负载。仅在全样本上评价，也无法判断方法是否真正缓解了标注不足。")
    add_body(doc, "本文据此把研究问题限定为：在不使用目标负载训练数据的前提下，将同一振动信号派生的确定性物理描述作为第二信息来源，能否提高小样本、噪声和跨负载条件下的诊断稳定性。这里的双流不是多传感器融合。第一流是原始振动的学习表示，第二流是由同一振动窗口、转速与轴承参数构造的212维物理描述。该限定避免把本研究误写为多模态传感器贡献，也明确了转速元数据属于融合方案的一部分。")
    add_body(doc, "围绕这一问题，本文完成三项工作。第一，构造覆盖四个包络频带和六类候选周期的物理描述，使谐波强度、周期重复性、相位集中和冲击形状进入统一表示。第二，提出逐通道门控残差融合，并用共享分类头的数据分支辅助损失约束原始表征。第三，在固定数据划分下，用5个训练种子、3个小样本子集、两种训练协议和12个迁移方向评价完整方案。本文不把这些工作表述为首次引入物理知识，也不把门控值解释为物理因果贡献。实验对照只能说明完整方案相对数据单分支的总体差异。")
    add_body(doc, "全文按以下顺序展开。第2节沿研究历程梳理机理特征、端到端学习、抗噪与迁移、小样本学习以及物理引导融合，并据此定位研究不足。第3节从轴承故障周期出发，说明物理描述、基础数据模型和门控融合模型。第4节给出数据划分、训练规则、统计口径及完整结果。第5节总结可支持的结论，并说明不能外推的范围。")

    add_heading(doc, "2  国内外研究现状及不足", 1)
    add_heading(doc, "2.1  从机理指标到端到端表征", 2)
    add_body(doc, "滚动轴承局部损伤的经典解释是滚动体周期性通过缺陷位置并激发结构共振。Randall和Antoni系统整理了故障特征频率、共振带选择和包络分析之间的关系[1]。这一阶段的研究重点是把机械结构和信号响应建立对应。方法通常具有明确的指标含义，但需要专业人员选择频带和阈值。Smith和Randall对CWRU数据逐记录分析后指出，不同记录的诊断难度并不一致，部分故障信号未呈现理想的典型特征[2]。这意味着数据集平均准确率不能替代对具体记录和迁移方向的分析。")
    add_body(doc, "随后，研究重心转向从原始信号中自动学习表示。Ince等以一维卷积网络直接处理原始电机信号，将特征提取和分类合并为端到端模型[3]。Zhang等提出WDCNN，通过宽首层卷积核增强对原始振动和噪声的适应，并利用AdaBN处理工况差异[4]。这一类模型降低了人工特征依赖，但仍主要从训练分布学习统计相关性。只要训练和测试共享记录背景，模型就可能获得较乐观的结果。")
    add_heading(doc, "2.2  抗噪、跨工况与小样本研究的演进", 2)
    add_body(doc, "为提高强噪声条件下的特征选择能力，Zhao等提出深度残差收缩网络，在残差结构中引入可学习软阈值[5]。该思路强化了数据驱动的噪声抑制，但软阈值本身不保证保留成分与故障机理一致。另一方面，负载和转速变化被逐渐表述为领域偏移问题。Ma等通过分布对齐构造跨域诊断框架[6]，Zou等结合Wasserstein对抗学习和类内约束处理不同工况[7]。这类研究建立了迁移诊断的技术路线，但许多方法在训练阶段需要目标域无标签数据或少量目标域标签。它们与完全不接触目标负载的单源直接泛化不同。")
    add_body(doc, "样本不足又推动了元学习、生成扩充和二维编码。Zhang等使用MAML学习可快速适应的新任务初始化[8]，Li等把不同工况组织为N-way K-shot任务[9]，Yang等使用生成模型扩充小样本[10]。焦孟萱等将振动信号转换为马尔可夫转移场图像，并结合条纹池化和注意力机制考察变工况和加噪诊断[11]。这些方法缓解了样本不足，但通常依赖额外任务、生成数据或适应阶段。若研究问题限定为仅用单一来源负载的少量真实窗口从头训练，既有结论不能直接套用。")
    add_body(doc, "随着结果可比性受到关注，Hendriks等指出CWRU研究中常见的数据构造可能弱化真实域偏移，并提出更严格的基准思路[12]。因此，随机窗口小样本与跨负载小样本应分开报告。前者反映共享记录背景下的窗口识别，后者反映来源负载到目标负载的直接迁移。即使采用跨负载划分，只要来源训练和验证仍来自相同记录，也不能称为独立轴承验证。")
    add_heading(doc, "2.3  物理引导学习与双流融合", 2)
    add_body(doc, "物理信息机器学习将方程、物理变量或机理约束加入输入、结构或损失，以降低模型对纯统计相关性的依赖[13]。在多源故障诊断中，Cui等提出多任务多传感器融合网络，利用振动和电流的互补信息提高判别性[14]。不过，多传感器方法增加了硬件信息量，性能改善可能同时来自传感器数量、模型容量和训练目标。")
    add_body(doc, "在轴承诊断中，Lu等依据故障特征频率设计物理引导的特征加权，使网络关注机理相关频带[15]。Ni等提出PIResNet，通过物理属性和双通道残差结构处理变速、变负载数据[16]。Wu等的PG-RDGFN进一步用原始振动和包络谱形成两路表示，并以物理置信度监督门控[17]。用户提供的Huang等研究则针对感应电机定子匝间短路和断条，联合编码电流与振动，并将负序电流和包络指标写入物理一致性损失[18]。这些工作说明物理知识能够进入输入、结构或优化目标，但并未留下“尚无物理引导诊断”的空白。")
    add_body(doc, "本研究与上述工作的真实差异位于更窄的交叉条件。其一，研究同时固定噪声协议、小样本预算和跨负载方向，并报告重复训练与子集间波动。其二，物理分支不是单一谱峰或标量置信度，而是在四个频带内统一组织谐波、周期、相位和形状描述。其三，跨负载评价不在训练阶段使用目标域数据。其四，结果不仅报告平均准确率，还保留方向级差值和类别召回。这样的定位能够支持一篇硕士论文，但不足以支持“首次”“全面优于”或“已实现物理可解释性”的强主张。")

    add_wide_table(
        doc,
        1,
        "相关技术演进、解决问题及其证据限制",
        ["阶段", "代表工作", "主要解决问题", "仍然存在的限制", "与本文关系"],
        [
            ["机理与信号分析", "Randall和Antoni[1]；Smith和Randall[2]", "建立冲击周期、包络谱与故障位置对应", "依赖频带、阈值与工况；记录难度不一", "提供确定性描述依据"],
            ["端到端一维网络", "Ince等[3]；WDCNN[4]", "自动从原始波形学习表示", "可能利用记录背景；低样本时不稳定", "形成数据单分支基础"],
            ["抗噪与迁移", "DRSN[5]；领域适应[6]-[7]", "抑制噪声并缩小域差异", "部分方法需要目标域数据；物理含义有限", "本文采用目标域不可见评价"],
            ["小样本学习", "MAML[8]；元学习[9]；生成扩充[10]-[11]", "用任务先验或生成样本缓解数据不足", "依赖额外任务、生成质量或适应过程", "本文只用来源真实窗口从头训练"],
            ["物理引导融合", "PIML[13]；M2FN[14]；PIResNet[16]；PG-RDGFN[17]-[18]", "把物理变量、特征或损失纳入学习", "组件贡献常难分离；不等同于因果解释", "构造212维显式分支并整体评价"],
        ],
        widths=[1.0, 1.45, 1.45, 1.55, 1.45],
        font_size=7.1,
    )

    add_heading(doc, "2.4  研究不足与本文切入点", 2)
    add_body(doc, "综合以上研究历程，本文聚焦四个可检验的不足。第一，噪声、小样本和负载迁移常被分别验证，缺少统一划分和统一统计口径。第二，部分物理引导方法只使用少量频率响应，没有形成覆盖多频带、多阶次和多类冲击描述的固定表示。第三，许多迁移方法依赖目标域数据，无法回答目标域完全不可见时的直接泛化。第四，单次平均准确率容易掩盖特定方向和类别退化。本文据此将贡献限定为具体描述构造、门控残差融合方式和多条件实证设计。")

    add_heading(doc, "3  理论部分和基础模型的介绍", 1)
    add_heading(doc, "3.1  轴承局部损伤与候选周期", 2)
    add_body(doc, "当滚动体经过局部缺陷时，接触刚度和载荷发生突变，振动中出现短时冲击及衰减共振。冲击重复频率取决于轴转频、滚动体数量、滚动体直径、节圆直径和接触角。设滚动体数量为n_b，轴转频为f_r，滚动体直径为d_b，节圆直径为d_p，接触角为θ，则保持架、外圈、内圈与滚动体候选频率可写为以下形式[1]。")
    add_equation(doc, "FTF = (f_r / 2)(1 − d_b cosθ / d_p)", "(1)")
    add_equation(doc, "BPFO = (n_b f_r / 2)(1 − d_b cosθ / d_p)", "(2)")
    add_equation(doc, "BPFI = (n_b f_r / 2)(1 + d_b cosθ / d_p)", "(3)")
    add_equation(doc, "BSF = (d_p f_r / 2d_b)[1 − (d_b cosθ / d_p)²]", "(4)")
    add_body(doc, "真实信号中的滑移、载荷变化和结构传递会使谱线偏离理想频率，因此本文没有把单一谱峰幅值作为硬判据，而是围绕转频、FTF、BSF、2BSF、BPFO和BPFI计算邻域统计。物理分支由确定性描述构成，但这些描述仍受转速误差、频带选择和噪声影响。它们属于机理启发特征，不等于完整动力学模型。")
    add_wide_figure(doc, figs["mechanism"], "图 1  轴承局部损伤机理、振动响应与212维物理描述之间的映射。图中候选周期用于构造邻域描述，不作为硬分类规则。", width=6.65)

    add_heading(doc, "3.2  多频带物理描述", 2)
    add_body(doc, "每个24,000点振动窗口首先在四个固定频带内形成解析包络，随后重采样至3 kHz，并去除两端25 ms以减弱滤波边缘效应。对每个频带的六类候选周期，计算8类描述，包括谐波强度、谐波覆盖率、谐波对比度、周期重复性、重复对比度、相位集中度、相位对比度和事件率。每个频带再补充5类形状统计，分别为顶部1%能量占比、对数峭度、对数峰值因子、相对均方根和谱平坦度。特征维数为4×(6×8+5)=212。")
    add_body(doc, "为了避免验证和测试信息进入物理特征尺度，稳健中心与尺度只用当前训练子集拟合。设第j维原始物理特征为u_j，训练子集的中位数为m_j，四分位距为q_j，则标准化结果为式(5)。尺度下限取0.05，最终数值截断至[-8,8]。这种处理降低极端窗口对尺度的影响，但不消除物理特征对转速和数据集的依赖。")
    add_equation(doc, "ũ_j = clip[(u_j − m_j) / max(q_j, 0.05), −8, 8]", "(5)")
    add_wide_figure(doc, figs["descriptor"], "图 2  多频带物理描述的构造过程。所有稳健标准化参数只由当前训练子集拟合。", width=6.55)
    add_wide_table(
        doc,
        2,
        "212维物理描述的组成",
        ["层级", "数量", "描述内容", "维数"],
        [
            ["频带", "4", "在不同共振范围内形成解析包络", "—"],
            ["候选周期", "6/频带", "f_r、FTF、BSF、2BSF、BPFO、BPFI", "—"],
            ["周期描述", "8/候选周期", "谐波强度、覆盖率、对比度、重复性、重复对比度、相位集中度、相位对比度、事件率", "4×6×8=192"],
            ["形状统计", "5/频带", "顶部能量、对数峭度、对数峰值因子、相对RMS、谱平坦度", "4×5=20"],
            ["合计", "—", "确定性物理描述", "212"],
        ],
        widths=[1.0, 1.0, 4.35, .65],
        font_size=7.4,
    )

    add_heading(doc, "3.3  数据单分支基础模型", 2)
    add_body(doc, "基础模型只接收原始振动。输入先经1×1卷积映射至32通道，再经过通道注意模块和10层扩张残差卷积。第i层扩张率为2^i，激活函数为tanh。末端通过1×1卷积和长度16的平均池化得到时序表示。对时间维计算均值和标准差，并经LayerNorm形成32维数据向量d。分类头为32→32→4的多层感知机。基础模型共33,964个可训练参数。它与融合方案共享数据编码器和分类头规格，因此能够作为主要数据驱动对照。")

    add_heading(doc, "3.4  双流门控残差融合模型", 2)
    add_body(doc, "物理向量经212→64→32的多层感知机映射为32维向量p。门控网络接收[d,p]，输出32维逐通道门控g。融合不直接替换数据表示，而是把受门控的物理表示作为残差加入数据表示。其目的是在物理描述不足时保留数据路径，在物理描述有用时允许通道级补充。")
    add_equation(doc, "g = sigmoid{MLP([d, p])}", "(6)")
    add_equation(doc, "z = LayerNorm[d + g ⊙ p]", "(7)")
    add_body(doc, "分类阶段使用同一个共享头分别处理融合向量z和数据向量d。训练损失由融合主交叉熵和数据辅助交叉熵构成。辅助项的目的不是增加新的推理分支，而是避免数据表示完全依赖物理路径。验证和测试只使用融合输出。")
    add_equation(doc, "L = CE(h(z), y) + λ CE(h(d), y)", "(8)")
    add_body(doc, "辅助系数λ由完整训练预算、纯干净协议下的五个训练种子验证交叉熵均值确定，并在后续实验前锁定。该过程使用全负载随机窗口验证数据，因此跨负载实验并不是完全独立于目标负载的模型开发。模型结构如图3所示。")
    add_wide_figure(doc, figs["architecture"], "图 3  物理引导双流门控残差融合模型。粉红色箭头表示物理信息和辅助监督路径；推理阶段仅使用融合主输出。", width=7.1)
    add_wide_table(
        doc,
        3,
        "基础模型与融合模型的主要模块",
        ["模块", "输入", "结构", "输出", "是否用于推理"],
        [
            ["数据编码器", "1×24,000振动", "1×1卷积、SE、10层扩张残差卷积、平均池化", "32维d", "是"],
            ["物理描述器", "振动、转速、轴承参数", "四频带包络与212维确定性计算", "212维u", "融合模型使用"],
            ["物理投影", "标准化u", "212→64→32，LayerNorm、ReLU、Dropout", "32维p", "融合模型使用"],
            ["门控融合", "d与p", "64→32→32，sigmoid；残差加和与LayerNorm", "32维z", "融合模型使用"],
            ["共享分类头", "d或z", "32→32→4", "四类logits", "主输出使用"],
            ["辅助分类", "d", "复用共享分类头，λ=0.3", "辅助CE", "仅训练"],
        ],
        widths=[1.0, 1.35, 2.9, .9, .85],
        font_size=7.2,
    )

    add_heading(doc, "3.5  方法能够支持的贡献", 2)
    add_body(doc, "从方法学上看，本文贡献不是证明某个物理方程直接决定网络输出，而是把多频带、多阶次的确定性描述以显式分支加入原始波形模型，并在低样本与跨负载条件下检验完整方案。门控值只反映模型在当前训练目标下学到的通道缩放，不能被解释为物理特征的真实贡献率。由于没有分别删除物理分支、门控、转速信息和辅助损失，实验不能把性能增益分解到单个组件。论文的有效主张应保持在完整方案层面。")

    add_heading(doc, "4  实验部分", 1)
    add_heading(doc, "4.1  研究问题与实验框架", 2)
    add_body(doc, "实验围绕三个递进问题展开。问题一是在训练与测试共享原始记录时，融合方案能否缓解极低训练预算下的初始化和子集敏感性。问题二是在使用来源负载全部训练窗口时，融合方案能否提高对其他负载的直接迁移。问题三是在单一来源负载只有10%、20%或50%训练窗口时，该优势是否仍然存在。三类实验采用同一模型、同一辅助系数和相同优化规则，但它们的证据含义不同。")
    add_wide_figure(doc, figs["experiment_flow"], "图 4  实验设计框架。记录内随机窗口识别与单源跨负载迁移分别回答同分布识别和分布变化下直接泛化问题。", width=7.1)

    add_heading(doc, "4.2  数据集与固定划分", 2)
    add_body(doc, "实验使用CWRU轴承数据中的40条记录，共形成687个不重叠振动窗口。每个窗口含24,000个采样点，采样率为48 kHz。四类状态为正常N、内圈IR、外圈OR和滚动体B。负载为0、1、2和3 HP。本文只使用驱动端振动及记录转速信息。数据来源、窗口划分和随机种子在实验前固定，不在测试结果出现后重抽。")
    add_body(doc, "记录内随机窗口划分包含481个训练窗口、103个验证窗口和103个测试窗口。10%、20%和50%训练预算分别使用49、98和242个窗口，每个预算设3个固定嵌套子集。跨负载划分以一个负载作为来源，来源内数据划分为训练和验证，其余三个负载作为目标测试，因此形成12个有向迁移任务。不同来源可用窗口不同，不能把所有方向难度差异都归因于负载本身。")
    add_wide_table(
        doc,
        4,
        "固定数据划分和小样本预算",
        ["划分", "来源或用途", "全量训练", "10%", "20%", "50%", "验证", "测试"],
        [
            ["随机窗口", "全部负载混合", "481", "49", "98", "242", "103", "103"],
            ["跨负载", "0 HP来源", "73", "9", "16", "37", "19", "其他负载195/200/200"],
            ["跨负载", "1 HP来源", "156", "17", "33", "78", "39", "其他负载92/200/200"],
            ["跨负载", "2 HP来源", "160", "17", "34", "80", "40", "其他负载92/195/200"],
            ["跨负载", "3 HP来源", "160", "17", "34", "80", "40", "其他负载92/195/200"],
        ],
        widths=[1.0, 1.25, .75, .55, .55, .55, .55, 1.8],
        font_size=7.2,
    )

    add_heading(doc, "4.3  训练协议、评价指标与统计口径", 2)
    add_body(doc, "设置纯干净和AWGN混合两种训练协议。纯干净协议不人工加噪。AWGN协议中，每次窗口出现时有50%概率保持干净，50%概率加入信噪比在[-4,4] dB连续均匀采样的白噪声。验证和测试均包含干净以及-4、-2、0、+2、+4 dB条件。每个测试噪声档使用5个固定噪声实例。噪声实例只是对同一窗口的重复观测，不作为新的独立样本。")
    add_body(doc, "每个配置使用5个训练种子1706至1710。优化器为AdamW，初始学习率5×10⁻⁴，权重衰减10⁻³，批量32，梯度裁剪阈值5。训练上限为200 epoch，余弦退火至10⁻⁵，验证损失连续20个epoch无改善时早停。物理标准化参数只在对应训练子集拟合。")
    add_body(doc, "评价指标包括准确率ACC、四类等权的均衡准确率BA、Macro-F1和各类召回率。跨负载结果先在同一噪声档内平均5个噪声实例，再对12个方向等权，最后对5个训练种子计算均值和样本标准差。小样本表中的±不是15次独立重复的标准差，而是3个固定子集各自五种子均值之间的样本标准差。")
    add_equation(doc, "BA = (1/K) Σₖ TPₖ / (TPₖ + FNₖ),   K = 4", "(9)")
    add_wide_table(
        doc,
        5,
        "主要训练和评价设置",
        ["项目", "设置", "项目", "设置"],
        [
            ["训练种子", "1706—1710", "最大epoch", "200"],
            ["优化器", "AdamW", "早停耐心", "20"],
            ["初始学习率", "5×10⁻⁴", "最小学习率", "1×10⁻⁵"],
            ["权重衰减", "1×10⁻³", "批量", "32"],
            ["训练噪声", "50%干净；50% U[-4,4] dB", "测试噪声", "干净及-4、-2、0、+2、+4 dB"],
            ["随机小样本", "3个固定嵌套子集", "跨负载", "4来源×3目标，共12方向"],
        ],
        widths=[1.0, 2.2, 1.0, 2.8],
        font_size=7.3,
    )

    add_heading(doc, "4.4  辅助损失系数选择", 2)
    add_body(doc, "候选λ为0.01、0.03、0.10、0.30和1.00。每个候选在全样本纯干净配置下训练5个种子，按各自最佳验证交叉熵的五种子均值选择。λ=0.30的均值最低，因此按预定规则锁定。需要注意，0.30与1.00的均值差仅1.10×10⁻⁷，小于训练种子波动。结果只能支持按规则选取0.30，不能支持其具有明确统计优势或普适最优性。")
    add_figure(doc, figs["selection"], "图 5  辅助损失系数的验证选择。误差线为5个训练种子的样本标准差。", width=3.15)
    add_wide_table(
        doc,
        6,
        "辅助损失系数的五种子验证结果",
        ["λ", "最佳验证CE均值", "样本标准差"],
        [["0.01", "8.2013×10⁻⁶", "1.3984×10⁻⁶"], ["0.03", "7.2375×10⁻⁶", "0.9157×10⁻⁶"], ["0.10", "6.4669×10⁻⁶", "0.6030×10⁻⁶"], ["0.30", "5.7021×10⁻⁶", "0.8531×10⁻⁶"], ["1.00", "5.8118×10⁻⁶", "0.5921×10⁻⁶"]],
        widths=[1.0, 2.5, 2.5],
        font_size=7.4,
        bold_best={(3, 0), (3, 1)},
    )

    add_heading(doc, "4.5  训练过程与可重复性", 2)
    add_body(doc, "图6给出0 HP来源、10%预算、子集4101、AWGN协议和种子1706的代表性训练曲线。数据单分支在第23个epoch取得最佳验证点并于第43个epoch停止。融合方案在第169个epoch取得最佳点并训练至第189个epoch。两者优化速度明显不同，因此不能用相同epoch位置比较瞬时性能。曲线的作用是说明保存权重对应验证选择过程，不足以证明所有配置均具有相同收敛形态。")
    add_body(doc, "V3实验分四个独立阶段执行。基础选参训练30个模型。随机窗口阶段训练190个新模型并只读引用10个已选模型。全样本跨负载阶段训练80个模型。小样本跨负载阶段训练720个模型。合计为1020个新模型，但不是单一入口连续完成。第312个小样本模型在原进程中断后从头补训，原不完整现场保留且不计入完成模型。全部指标均可由保存的混淆计数复算，最大绝对差不超过1.11×10⁻¹⁶。")
    add_wide_figure(doc, figs["training"], "图 6  代表性小样本跨负载训练曲线。数据单分支为m0011，融合方案为m0016；两者均使用0 HP来源、10%预算、子集4101、AWGN协议和种子1706。", width=6.4)

    add_heading(doc, "4.6  记录内随机窗口小样本结果", 2)
    add_body(doc, "在10%训练预算下，融合方案在三种代表条件中均明显高于数据单分支。AWGN训练和-4 dB测试时，BA由82.66%提高到94.72%，子集间标准差由7.65个百分点降至0.28个百分点。该结果说明完整方案减少了极低样本下的子集敏感性。不过，数据单分支10%结果包含两个退化训练实例。因此，12.06个百分点的差值不能全部归因于物理知识提高了一般判别能力，其中一部分来自融合方案避免了对照模型的优化失败。")
    add_body(doc, "当预算增加至20%和50%时，随机窗口任务接近饱和。AWGN、-4 dB条件下的BA增益分别缩小为1.21和0.83个百分点。纯干净训练时，融合方案在20%和50%预算分别低0.06和0.44个百分点。全样本纯干净结果也低0.16个百分点。因此，本组实验支持低样本稳定性优势，不支持融合方案在所有预算和所有条件下全面更优。")
    add_body(doc, "从统计口径看，7.65个百分点是三个固定子集均值之间的波动，不是单个训练种子的标准差。它反映所抽训练窗口组合对结果的影响。融合方案在10%条件下同时提高均值并缩小该波动，这比只比较单次最佳准确率更有信息。不过，三个子集存在嵌套关系，波动下降仍不能按独立重复实验进行显著性检验。")
    add_body(doc, "随机窗口任务在20%以上快速饱和，也说明该划分的区分度有限。训练、验证和测试来自相同原始记录，模型可以利用稳定的记录背景。因而，本组结果主要用于检查优化和样本敏感性，不能作为跨工况贡献的核心证据。跨负载实验承担更强的外推检验。")
    add_wide_figure(doc, figs["random"], "图 7  记录内随机窗口小样本在-4 dB下的BA。误差线为3个固定子集五种子均值之间的样本标准差。", width=6.4)
    add_wide_table(
        doc,
        7,
        "记录内随机窗口小样本结果",
        ["预算", "训练与测试条件", "数据单分支BA（%）", "融合方案BA（%）", "差值（百分点）"],
        [
            ["10%", "纯干净 / 干净", "91.68 ± 8.54", "98.84 ± 0.28", "+7.16"],
            ["10%", "AWGN / 干净", "83.24 ± 8.35", "98.79 ± 0.25", "+15.55"],
            ["10%", "AWGN / -4 dB", "82.66 ± 7.65", "94.72 ± 0.28", "+12.06"],
            ["20%", "纯干净 / 干净", "99.17 ± 0.59", "99.11 ± 0.09", "-0.06"],
            ["20%", "AWGN / 干净", "98.90 ± 0.33", "99.17 ± 0.00", "+0.27"],
            ["20%", "AWGN / -4 dB", "97.29 ± 0.12", "98.50 ± 0.24", "+1.21"],
            ["50%", "纯干净 / 干净", "99.50 ± 0.17", "99.06 ± 0.10", "-0.44"],
            ["50%", "AWGN / 干净", "98.94 ± 0.19", "99.17 ± 0.00", "+0.22"],
            ["50%", "AWGN / -4 dB", "97.98 ± 0.35", "98.81 ± 0.06", "+0.83"],
        ],
        widths=[.65, 1.7, 1.45, 1.45, 1.05],
        font_size=7.2,
    )

    add_heading(doc, "4.7  全样本跨负载结果", 2)
    add_body(doc, "全样本跨负载结果对12个有向迁移任务等权。纯干净训练和干净测试时，融合方案BA为95.69%，数据单分支为84.21%，差值11.48个百分点。AWGN训练和干净测试时，差值为6.79个百分点。AWGN训练和-4 dB测试时，融合方案BA为89.51%，数据单分支为84.31%，差值5.20个百分点。ACC和Macro-F1呈现相同总体方向。")
    add_body(doc, "融合方案在-4 dB下的绝对BA更高，但从AWGN干净条件到-4 dB的下降为4.30个百分点，数据单分支下降2.71个百分点。因此，结果不能被写成融合方案具有更小的相对噪声退化。它说明融合方案在两个测试条件下均达到更高绝对水平。")
    add_body(doc, "BA增益大于ACC增益，表明改善并非只由样本较多的故障类别推动。Macro-F1同步提高则说明精确率和召回率的综合关系总体改善。但三个汇总指标仍来自同一批预测，不能被当作三份独立证据。它们用于描述性能的不同侧面。")
    add_body(doc, "五种子标准差在三种代表条件下均未超过2.20个百分点，说明平均优势不依赖单一初始化。不过，训练种子只改变参数初始化和随机增强，不改变原始记录、来源负载或目标负载。模型对数据采集变化的稳定性仍需跨平台验证。")
    add_wide_figure(doc, figs["full"], "图 8  全样本跨负载12方向等权BA。均值和误差分别对应5个训练种子。", width=6.7)
    add_wide_table(
        doc,
        8,
        "全样本跨负载12方向等权结果",
        ["训练与测试条件", "方法", "ACC（%）", "BA（%）", "Macro-F1（%）"],
        [
            ["纯干净 / 干净", "数据单分支", "82.41 ± 2.82", "84.21 ± 1.71", "82.31 ± 2.45"],
            ["纯干净 / 干净", "融合方案", "94.71 ± 1.19", "95.69 ± 0.98", "95.13 ± 1.37"],
            ["AWGN / 干净", "数据单分支", "84.60 ± 2.02", "87.02 ± 1.93", "85.19 ± 2.62"],
            ["AWGN / 干净", "融合方案", "92.40 ± 2.20", "93.81 ± 1.84", "92.77 ± 2.56"],
            ["AWGN / -4 dB", "数据单分支", "82.42 ± 3.05", "84.31 ± 2.87", "82.43 ± 3.83"],
            ["AWGN / -4 dB", "融合方案", "87.23 ± 1.60", "89.51 ± 1.29", "87.92 ± 2.06"],
        ],
        widths=[1.65, 1.15, 1.35, 1.35, 1.45],
        font_size=7.3,
        bold_best={(1, 2), (1, 3), (1, 4), (3, 2), (3, 3), (3, 4), (5, 2), (5, 3), (5, 4)},
    )
    add_body(doc, "方向级结果揭示了平均值掩盖的局部问题。AWGN、-4 dB条件下，2 HP到1 HP方向的融合BA低0.46个百分点，外圈召回低2.27个百分点。3 HP到1 HP的滚动体召回低8.20个百分点，0 HP到3 HP的滚动体召回低6.80个百分点。图9用差值热图保留了这些例外。")
    add_figure(doc, figs["heatmap"], "图 9  全样本跨负载AWGN、-4 dB条件下融合方案相对数据单分支的方向级BA差值。", width=3.15)

    add_heading(doc, "4.8  小样本跨负载结果", 2)
    add_body(doc, "小样本跨负载是本研究最具区分度的实验。10%预算下，来源模型只用9至17个窗口训练。AWGN、-4 dB条件下，数据单分支BA为36.17%，融合方案为85.96%，差值49.79个百分点。20%预算下差值为24.01个百分点，50%预算下仍为9.88个百分点。纯干净与AWGN干净测试也呈现相同的预算趋势。随着训练样本增加，数据单分支逐步恢复，融合增益相应收缩。这说明方案的主要价值位于训练信息最受限时，而不是在所有预算上保持固定增益。")
    add_body(doc, "融合方案的子集间标准差在-4 dB条件下分别为0.84、0.48和0.25个百分点，均低于数据单分支的2.88、4.64和0.37个百分点。该结果可以支持稳定性改善，但三个子集来自同一训练池，不能被当作三个独立数据集，也不能据此给出总体置信区间。")
    add_wide_figure(doc, figs["small"], "图 10  小样本跨负载AWGN、-4 dB条件下的12方向等权BA。误差线为3个固定子集五种子均值之间的样本标准差。", width=6.5)
    add_wide_table(
        doc,
        9,
        "小样本跨负载12方向等权结果",
        ["预算", "训练与测试条件", "数据单分支BA（%）", "融合方案BA（%）", "差值（百分点）"],
        [
            ["10%", "纯干净 / 干净", "49.88 ± 3.88", "89.97 ± 1.15", "+40.09"],
            ["10%", "AWGN / 干净", "38.10 ± 3.08", "90.02 ± 0.55", "+51.92"],
            ["10%", "AWGN / -4 dB", "36.17 ± 2.88", "85.96 ± 0.84", "+49.79"],
            ["20%", "纯干净 / 干净", "68.87 ± 1.86", "93.15 ± 0.12", "+24.28"],
            ["20%", "AWGN / 干净", "66.56 ± 6.19", "91.41 ± 0.37", "+24.85"],
            ["20%", "AWGN / -4 dB", "63.66 ± 4.64", "87.66 ± 0.48", "+24.01"],
            ["50%", "纯干净 / 干净", "81.76 ± 1.48", "95.08 ± 0.28", "+13.32"],
            ["50%", "AWGN / 干净", "81.89 ± 0.68", "93.66 ± 0.21", "+11.76"],
            ["50%", "AWGN / -4 dB", "79.16 ± 0.37", "89.04 ± 0.25", "+9.88"],
        ],
        widths=[.65, 1.7, 1.45, 1.45, 1.05],
        font_size=7.2,
        bold_best={(i, 3) for i in range(9)},
    )

    add_heading(doc, "4.9  类别差异与失败案例", 2)
    add_body(doc, "类别召回显示，低样本增益主要来自正常、内圈和滚动体故障。10%预算、AWGN、-4 dB时，融合方案的正常和内圈召回分别为99.15%和94.73%，数据单分支只有24.90%和26.60%。滚动体召回由16.87%提高到69.13%。外圈召回的改善较小，为76.33%到80.84%。当预算为20%时，融合方案外圈召回反而低1.74个百分点。")
    add_body(doc, "进一步检查756个固定方向、子集、协议和测试条件组合，BA差值为负的配置有37个。其中30个集中在2 HP到1 HP，7个集中在1 HP到0 HP。最小差值为-3.35个百分点，发生在2 HP到1 HP、20%预算、子集4103、AWGN训练和+4 dB测试。外圈召回差值为负的配置达到369个。这一模式说明总体BA优势并未解决外圈类别在特定方向的系统性不足。")
    add_wide_figure(doc, figs["recall"], "图 11  小样本跨负载AWGN、-4 dB条件下的类别召回率。类别结果先在12个方向内等权。", width=7.0)
    add_wide_table(
        doc,
        10,
        "小样本跨负载AWGN、-4 dB条件下的类别召回率",
        ["预算", "方法", "N（%）", "IR（%）", "OR（%）", "B（%）"],
        [
            ["10%", "数据单分支", "24.90", "26.60", "76.33", "16.87"],
            ["10%", "融合方案", "99.15", "94.73", "80.84", "69.13"],
            ["20%", "数据单分支", "68.03", "62.35", "85.94", "38.30"],
            ["20%", "融合方案", "99.94", "94.73", "84.20", "71.78"],
            ["50%", "数据单分支", "87.28", "83.09", "91.15", "55.14"],
            ["50%", "融合方案", "99.62", "96.58", "91.30", "68.67"],
        ],
        widths=[.7, 1.4, .9, .9, .9, .9],
        font_size=7.3,
    )
    add_heading(doc, "4.9.1  失败模式的替代解释", 3)
    add_body(doc, "负差集中于2 HP到1 HP和1 HP到0 HP，说明问题不是均匀随机误差。一个合理解释是，物理描述对正常和内圈周期响应提供了较稳定的补充，而外圈响应更容易受载荷路径、传递函数和故障方位影响。负载改变后，同一候选阶次附近的幅值和对比度可能发生系统变化，使固定描述不能完全保持域不变。现有实验没有直接测量传递路径，因此这一解释只能作为待检验假设。")
    add_body(doc, "另一个替代解释来自训练预算和类别覆盖。四个来源负载的全量训练窗口为73至160个，验证窗口为19至40个。0 HP的10%条件只有1个正常窗口。即使按类别配额抽样，不同来源仍具有不同的记录数量和可观测变化范围。方向差异可能同时由负载、样本量和记录特性造成，不能全部归因于机械负载。")
    add_body(doc, "平均BA对四类召回等权，因此能够减轻类别支持数不平衡的影响，但不会消除方向间复用模型和目标数据带来的相关性。图12选择负差方向展示混淆结构，其目的不是证明融合方案失败，而是说明总体均值无法替代类别级诊断。若后续只增加更多总体平均结果而不处理这一方向，论文的证据不会实质增强。")
    add_body(doc, "图12给出全样本2 HP到1 HP方向的合并混淆矩阵。每个单元格同时保留预测计数和按真实类别归一化的比例。合并计数包含训练种子与固定噪声实例的重复预测，只用于观察错误结构，不代表新增独立测试样本。")
    add_wide_figure(doc, figs["confusion"], "图 12  全样本2 HP→1 HP、AWGN训练和-4 dB测试下的合并混淆矩阵。每格上行为计数，下行为真实类别行比例。", width=7.0)

    add_heading(doc, "4.10  参数、延迟与显存开销", 2)
    add_body(doc, "数据单分支含33,964个参数，融合方案含53,068个参数，增加19,104个，增幅56.25%。在NVIDIA GeForce RTX 5060 Laptop GPU上，融合方案的单样本推理延迟约为7.01至7.52 ms，数据单分支约为2.38至2.59 ms。融合方案峰值显存为1343.52 MiB，比数据单分支增加147.94 MiB。延迟计时包含归一化、在线物理特征提取和网络推理，因此能够反映完整方案的端到端代价。")
    add_body(doc, "现有Conv1d和Linear FLOPs统计没有覆盖物理FFT、激活和逐元素运算，不能用于声称两种方法计算量几乎相同。对部署而言，融合方案的主要代价是在线物理描述计算和近3倍实测延迟。该代价是否可接受取决于目标设备的采样周期和诊断时限，当前实验没有完成嵌入式部署验证。")
    add_wide_figure(doc, figs["cost"], "图 13  参数量与单样本实测推理延迟。延迟包含在线物理特征提取；峰值显存见表11。", width=6.6)
    add_wide_table(
        doc,
        11,
        "三个实验阶段的计算成本",
        ["阶段", "方法", "参数量", "平均延迟（ms）", "峰值显存（MiB）"],
        [
            ["随机窗口", "数据单分支", "33,964", "2.38", "1195.58"],
            ["随机窗口", "融合方案", "53,068", "7.01", "1343.52"],
            ["全样本跨负载", "数据单分支", "33,964", "2.59", "1195.58"],
            ["全样本跨负载", "融合方案", "53,068", "7.52", "1343.52"],
            ["小样本跨负载", "数据单分支", "33,964", "2.47", "1195.58"],
            ["小样本跨负载", "融合方案", "53,068", "7.27", "1343.52"],
        ],
        widths=[1.4, 1.4, 1.1, 1.35, 1.45],
        font_size=7.3,
    )

    add_heading(doc, "4.11  有效性威胁与证据边界", 2)
    add_body(doc, "实验结果在当前固定划分内具有内部一致性，但外部效度有限。随机窗口训练、验证和测试共享原始记录，只能支持记录内窗口识别。跨负载来源训练和验证也共享记录，不能视为独立轴承验证。辅助系数使用全负载随机窗口验证数据选择，目标负载已参与模型开发。项目还存在历史测试访问，因此本研究不是在从未暴露的确认性测试集上完成。")
    add_body(doc, "小样本实验仍使用19至40个完整验证窗口，训练窗口数量不等于总标注预算。五个训练种子只反映优化随机性，三个子集来自同一训练池，12个方向复用来源模型和目标数据。AWGN只是可控加性噪声，不能代表非平稳冲击、电磁干扰、转速波动和传感器松动。")
    add_body(doc, "在因果归因上，融合方案同时增加212维物理描述、转速元数据、19,104个参数、逐通道门控和辅助损失。当前对照没有把这些因素分开。因此，可以得出的结论是完整方案相对数据单分支具有经验优势，不能断言优势专门来自物理知识，也不能把门控输出作为机理解释证据。")
    add_wide_table(
        doc,
        12,
        "实验事实、可支持结论及不可支持结论",
        ["实验事实", "可支持的解释", "不能支持的解释"],
        [
            ["随机窗口10%下BA和稳定性明显提高", "完整方案降低极低样本的子集与训练敏感性", "模型已学到可跨设备推广的故障规律"],
            ["12方向跨负载平均BA提高", "在当前CWRU负载划分下具有较强直接迁移能力", "对所有方向、所有类别全面优于对照"],
            ["212维物理描述和门控共同使用", "显式物理描述可作为第二信息来源", "门控值是物理因果贡献率"],
            ["AWGN -4 dB绝对性能较高", "对可控白噪声具有一定鲁棒性", "对真实工业噪声已经验证"],
            ["5种子和3子集重复", "结果不是单次初始化偶然值", "形成独立数据集层面的统计推断"],
        ],
        widths=[2.05, 2.4, 2.45],
        font_size=7.2,
    )

    add_heading(doc, "5  结论与不足", 1)
    add_heading(doc, "5.1  主要结论", 2)
    add_body(doc, "本文针对噪声、负载变化和小样本同时存在的轴承诊断问题，提出由原始振动数据表示和212维确定性物理描述构成的双流门控残差融合模型。物理描述围绕四个频带和六类候选周期组织谐波、周期、相位与形状信息，并只用当前训练子集拟合稳健标准化参数。模型以数据表示为主路径，通过逐通道门控注入物理表示，并用共享分类头的数据辅助损失维持原始分支判别能力。")
    add_body(doc, "固定实验结果表明，完整方案的主要优势集中在低样本和跨负载同时存在的条件。记录内随机窗口10%预算、AWGN训练和-4 dB测试时，BA由82.66%提高到94.72%。全样本12方向跨负载时，BA由84.31%提高到89.51%。10%小样本跨负载时，BA由36.17%提高到85.96%。随着训练预算增加，差值逐步缩小，说明方案更像是有限数据下的稳定补充，而不是高预算条件下必然提高上限的机制。")
    add_body(doc, "类别和方向分析同时否定了全面占优的表述。2 HP到1 HP是主要负差方向，外圈召回在大量配置中出现负差。融合方案从干净到-4 dB的相对下降也不小于数据单分支。因而，论文结论应限定为当前CWRU固定划分下的平均经验优势，并保留局部失败条件。")

    add_heading(doc, "5.2  研究不足", 2)
    add_body(doc, "第一，实验仅使用CWRU单一平台，并且训练、验证和测试在记录或轴承实例层面并非完全独立。随机窗口结果容易受到记录背景影响，跨负载结果也不能代表跨设备或跨轴承泛化。第二，AWGN无法覆盖真实工业噪声。第三，物理分支使用转速元数据，而数据单分支不使用，现有设计不能区分转速信息和物理描述本身的贡献。第四，融合模型同时改变特征、结构、参数量和训练目标，缺少严格组件消融。第五，三个小样本子集和五个训练种子不足以形成强统计推断。第六，在线物理特征导致推理延迟接近数据单分支的3倍，尚未进行边缘设备部署。")

    add_heading(doc, "5.3  后续改进路径", 2)
    add_body(doc, "后续工作应按优先级推进。首先，在模型结构不变的前提下补充最小必要消融，包括仅增加等参数容量、仅加入转速、物理分支但无门控、门控但无辅助损失，以及随机打乱物理描述的负对照。该组实验能够直接改善贡献归因。其次，应建立记录级或轴承实例级完全隔离的开发与确认性测试，并在新的轴承数据集或实际台架上复现。再次，应加入非平稳背景噪声、冲击干扰、转速误差和传感器安装变化，检查物理描述是否在输入偏离时失效。最后，应针对2 HP到1 HP和外圈类别设计方向级误差分析，而不是继续增加总体平均图表。")
    add_body(doc, "在硕士论文层面，最优先的改进是完成最小消融和收紧结论语言。它们会实质性提高证据与结论的匹配程度。继续增加修辞、三维可视化或更多同源随机划分只能改善呈现，不能弥补贡献无法分解和外部验证不足。")

    add_heading(doc, "参考文献", 1)
    refs = [
        "[1] Randall R B, Antoni J. Rolling element bearing diagnostics—A tutorial. Mechanical Systems and Signal Processing, 2011, 25(2): 485–520. DOI: 10.1016/j.ymssp.2010.07.017.",
        "[2] Smith W A, Randall R B. Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study. Mechanical Systems and Signal Processing, 2015, 64–65: 100–131. DOI: 10.1016/j.ymssp.2015.04.021.",
        "[3] Ince T, Kiranyaz S, Eren L, Askar M, Gabbouj M. Real-Time Motor Fault Detection by 1-D Convolutional Neural Networks. IEEE Transactions on Industrial Electronics, 2016, 63(11): 7067–7075. DOI: 10.1109/TIE.2016.2582729.",
        "[4] Zhang W, Peng G, Li C, Chen Y, Zhang Z. A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals. Sensors, 2017, 17(2): 425. DOI: 10.3390/s17020425.",
        "[5] Zhao M, Zhong S, Fu X, Tang B, Pecht M. Deep Residual Shrinkage Networks for Fault Diagnosis. IEEE Transactions on Industrial Informatics, 2020, 16(7): 4681–4690. DOI: 10.1109/TII.2019.2943898.",
        "[6] Ma P, Zhang H, Fan W, Wang C. A diagnosis framework based on domain adaptation for bearing fault diagnosis across diverse domains. ISA Transactions, 2020, 99: 465–478. DOI: 10.1016/j.isatra.2019.08.040.",
        "[7] Zou Y, Liu Y, Deng J, Jiang Y, Zhang W. A novel transfer learning method for bearing fault diagnosis under different working conditions. Measurement, 2021, 171: 108767. DOI: 10.1016/j.measurement.2020.108767.",
        "[8] Zhang S, Ye F, Wang B, Habetler T G. Few-Shot Bearing Fault Diagnosis Based on Model-Agnostic Meta-Learning. IEEE Transactions on Industry Applications, 2021, 57(5): 4754–4764. DOI: 10.1109/TIA.2021.3091958.",
        "[9] Li C, Li S, Zhang A, He Q, Liao Z, Hu J. Meta-learning for few-shot bearing fault diagnosis under complex working conditions. Neurocomputing, 2021, 439: 197–211. DOI: 10.1016/j.neucom.2021.01.099.",
        "[10] Yang X, Liu B, Xiang L, Hu A, Xu Y. A novel intelligent fault diagnosis method of rolling bearings with small samples. Measurement, 2022, 203: 111899. DOI: 10.1016/j.measurement.2022.111899.",
        "[11] 焦孟萱, 雷春丽, 马淑珍, 等. 基于MTF-SPCNN的小样本滚动轴承变工况故障诊断方法. 北京航空航天大学学报, 2024, 50(12): 3696–3708. DOI: 10.13700/j.bh.1001-5965.2022.0927.",
        "[12] Hendriks J, Dumond P, Knox D A. Towards better benchmarking using the CWRU bearing fault dataset. Mechanical Systems and Signal Processing, 2022, 169: 108732. DOI: 10.1016/j.ymssp.2021.108732.",
        "[13] Karniadakis G E, Kevrekidis I G, Lu L, Perdikaris P, Wang S, Yang L. Physics-informed machine learning. Nature Reviews Physics, 2021, 3: 422–440. DOI: 10.1038/s42254-021-00314-5.",
        "[14] Cui J, Xie P, Wang X, Wang J, He Q, Jiang G. M2FN: An end-to-end multi-task and multi-sensor fusion network for intelligent fault diagnosis. Measurement, 2022, 204: 112085. DOI: 10.1016/j.measurement.2022.112085.",
        "[15] Lu H, Nemani V P, Barzegar V, et al. A physics-informed feature weighting method for bearing fault diagnostics. Mechanical Systems and Signal Processing, 2023, 191: 110171. DOI: 10.1016/j.ymssp.2023.110171.",
        "[16] Ni Q, Ji J C, Halkon B, Feng K, Nandi A K. Physics-Informed Residual Network for rolling element bearing fault diagnostics. Mechanical Systems and Signal Processing, 2023, 200: 110544. DOI: 10.1016/j.ymssp.2023.110544.",
        "[17] Wu C, Jiao G, Jin K, Zhang Y, Li Z, Zhang Z. A Physics-Guided Raw-Dominant Gated Fusion Method for Fine-Grained Bearing Fault Diagnosis. Applied Sciences, 2026, 16(16): 8337. DOI: 10.3390/app16168337.",
        "[18] Huang X, Ruzimov S, Zhang J, Cheng M. Physics-Guided Dual-Stream Fusion Network for Fault Detection of Induction Motors Using Multi-Sensor Signals. IEEE Transactions on Energy Conversion, 2026, early access: 1–11. DOI: 10.1109/TEC.2026.3726800.",
        "[19] Case Western Reserve University Bearing Data Center. Bearing vibration data and apparatus description. Accessed 2026-10-27.",
    ]
    for ref in refs:
        p = doc.add_paragraph(ref, style="Reference")
        for r in p.runs:
            set_east_asia_font(r, "宋体")

    # A final continuous one-column section forces Word to balance the preceding reference columns.
    add_columns_section(doc, 1)
    tail = doc.add_paragraph()
    tail.paragraph_format.space_after = Pt(0)
    tail.paragraph_format.line_spacing = Pt(1)
    tail.paragraph_format.first_line_indent = Pt(0)
    tr = tail.add_run(" ")
    tr.font.size = Pt(1)

    # Core properties are kept generic for privacy and portability.
    doc.core_properties.title = "物理引导双流门控融合轴承故障诊断研究"
    doc.core_properties.subject = "硕士学位论文正文草稿"
    doc.core_properties.author = ""
    doc.core_properties.keywords = "轴承故障诊断; 物理引导; 小样本; 跨负载"
    doc.core_properties.comments = ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_PATH)
    return OUT_PATH


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs = {
        "mechanism": create_mechanism_figure(),
        "descriptor": create_descriptor_figure(),
        "architecture": create_architecture_figure(),
        "experiment_flow": create_experiment_flow_figure(),
        "selection": create_selection_figure(),
        "training": create_training_figure(),
        "random": create_random_window_figure(),
        "full": create_full_transfer_bar(),
        "heatmap": create_direction_heatmap(),
        "small": create_small_transfer_figure(),
        "recall": create_recall_figure(),
        "confusion": create_confusion_figure(),
        "cost": create_cost_figure(),
    }
    result = build_document(figs)
    print(result)


if __name__ == "__main__":
    main()

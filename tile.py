"""
专业 Tile 编辑器
支持模块化设计、自定义图片输入、专业 UI 风格
"""

import tkinter as tk
from tkinter import ttk, colorchooser, messagebox, filedialog
import random
from typing import List, Optional, Tuple, Set, TYPE_CHECKING
from dataclasses import dataclass, asdict
from PIL import Image, ImageTk, ImageDraw
import os
import json
from collections import deque

# ==========================================
# 1. 数据模型
# ==========================================

@dataclass
class SocketConfig:
    """插槽配置类"""
    direction: str
    inset_start: float
    inset_end: float

    def __post_init__(self):
        self.inset_start = min(self.inset_start, self.inset_end)
        self.inset_end = max(self.inset_start, self.inset_end)

    @property
    def length(self) -> float:
        return abs(self.inset_end - self.inset_start)

    def get_opposite_dir(self) -> str:
        opposite_map = {'U': 'D', 'D': 'U', 'L': 'R', 'R': 'L'}
        return opposite_map[self.direction]


@dataclass
class TileConfig:
    """Tile 配置类"""
    name: str
    width: float
    height: float
    color: str
    image_path: Optional[str] = None
    sockets: List[SocketConfig] = None

    def __post_init__(self):
        self.width = max(20, self.width)
        self.height = max(20, self.height)
        if self.sockets is None:
            self.sockets = []

    def get_image(self, width: int = 100, height: int = 100) -> Optional[ImageTk.PhotoImage]:
        """获取 tile 的图像"""
        if self.image_path and os.path.exists(self.image_path):
            try:
                img = Image.open(self.image_path)
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                return ImageTk.PhotoImage(img)
            except:
                pass
        return None


@dataclass
class PlacedTile:
    """已放置的 tile"""
    config: TileConfig
    x: float
    y: float
    scale: float = 1.0
    rotation: float = 0.0

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """获取 tile 的边界 (x1, y1, x2, y2)"""
        return (self.x, self.y, self.x + self.config.width, self.y + self.config.height)

    def intersects(self, other: 'PlacedTile') -> bool:
        """检查是否与另一个 tile 相交"""
        b1, b2 = self.get_bounds(), other.get_bounds()
        return not (b1[2] <= b2[0] + 0.1 or
                   b1[0] >= b2[2] - 0.1 or
                   b1[3] <= b2[1] + 0.1 or
                   b1[1] >= b2[3] - 0.1)


# ==========================================
# 2. 辅助工具类
# ==========================================

class EdgeDetector:
    """边检测器 - 自动识别图片的垂直边"""

    @staticmethod
    def detect_edges(image_path: str) -> List[SocketConfig]:
        """检测图片边缘，生成插槽配置"""
        try:
            img = Image.open(image_path).convert("RGBA")
            width, height = img.size
            data = img.load()

            sockets = []

            # 检测四个方向的边
            edges = EdgeDetector._detect_all_edges(data, width, height)

            for direction, segments in edges.items():
                # 合并相邻的段
                merged_segments = EdgeDetector._merge_segments(segments)

                for start, end in merged_segments:
                    if end > start:  # 确保有效的边
                        sockets.append(SocketConfig(direction, float(start), float(end)))

            return sockets
        except Exception as e:
            print(f"边检测错误: {e}")
            return []

    @staticmethod
    def _detect_all_edges(data, width: int, height: int):
        """检测所有边 - 找到连续的非透明像素段"""
        edges = {'U': [], 'D': [], 'L': [], 'R': []}

        # 检测上边 - 找所有连续的非透明段
        in_edge = False
        start = 0
        for x in range(width):
            is_opaque = data[x, 0][3] > 0
            if is_opaque and not in_edge:
                in_edge = True
                start = x
            elif not is_opaque and in_edge:
                in_edge = False
                edges['U'].append((start, x - 1))
        if in_edge:
            edges['U'].append((start, width - 1))

        # 检测下边
        in_edge = False
        for x in range(width):
            is_opaque = data[x, height - 1][3] > 0
            if is_opaque and not in_edge:
                in_edge = True
                start = x
            elif not is_opaque and in_edge:
                in_edge = False
                edges['D'].append((start, x - 1))
        if in_edge:
            edges['D'].append((start, width - 1))

        # 检测左边
        in_edge = False
        for y in range(height):
            is_opaque = data[0, y][3] > 0
            if is_opaque and not in_edge:
                in_edge = True
                start = y
            elif not is_opaque and in_edge:
                in_edge = False
                edges['L'].append((start, y - 1))
        if in_edge:
            edges['L'].append((start, height - 1))

        # 检测右边
        in_edge = False
        for y in range(height):
            is_opaque = data[width - 1, y][3] > 0
            if is_opaque and not in_edge:
                in_edge = True
                start = y
            elif not is_opaque and in_edge:
                in_edge = False
                edges['R'].append((start, y - 1))
        if in_edge:
            edges['R'].append((start, height - 1))

        return edges

    @staticmethod
    def _merge_segments(segments: List[Tuple[int, int]], min_gap: int = 3) -> List[Tuple[int, int]]:
        """合并相邻的段"""
        if not segments:
            return []

        sorted_segments = sorted(segments)
        merged = [sorted_segments[0]]

        for current in sorted_segments[1:]:
            last = merged[-1]
            if current[0] - last[1] <= min_gap:
                # 合并
                merged[-1] = (last[0], current[1])
            else:
                merged.append(current)

        return merged


class ProjectManager:
    """项目管理器 - 保存和加载项目"""

    def __init__(self, project_file: str = "tile_project.json"):
        self.project_file = project_file

    def save(self, templates: List['TileConfig'], tile_positions: dict, tile_scales: dict):
        """保存项目"""
        data = {
            "templates": [],
            "positions": {str(k): v for k, v in tile_positions.items()},
            "scales": {str(k): v for k, v in tile_scales.items()}
        }

        for tile in templates:
            tile_data = {
                "name": tile.name,
                "width": tile.width,
                "height": tile.height,
                "color": tile.color,
                "image_path": tile.image_path,
                "sockets": []
            }

            for socket in tile.sockets:
                tile_data["sockets"].append({
                    "direction": socket.direction,
                    "inset_start": socket.inset_start,
                    "inset_end": socket.inset_end
                })

            data["templates"].append(tile_data)

        try:
            with open(self.project_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False

    def load(self) -> Optional[Tuple[List[TileConfig], dict, dict]]:
        """加载项目"""
        if not os.path.exists(self.project_file):
            return None

        try:
            with open(self.project_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            templates = []
            for tile_data in data["templates"]:
                tile = TileConfig(
                    tile_data["name"],
                    tile_data["width"],
                    tile_data["height"],
                    tile_data["color"],
                    tile_data.get("image_path")
                )

                for socket_data in tile_data.get("sockets", []):
                    socket = SocketConfig(
                        socket_data["direction"],
                        socket_data["inset_start"],
                        socket_data["inset_end"]
                    )
                    tile.sockets.append(socket)

                templates.append(tile)

            positions = {int(k): v for k, v in data.get("positions", {}).items()}
            scales = {int(k): v for k, v in data.get("scales", {}).items()}

            return templates, positions, scales
        except Exception as e:
            print(f"加载失败: {e}")
            return None


# ==========================================
# 2. 主题配置
# ==========================================

class UITheme:
    """统一的 UI 主题配置"""
    # 调色板 - 现代专业风格
    PRIMARY = "#2563EB"        # 主蓝色
    PRIMARY_HOVER = "#1D4ED8"  # 悬停蓝色
    SECONDARY = "#64748B"      # 次要色
    SUCCESS = "#10B981"        # 成功绿
    WARNING = "#F59E0B"        # 警告橙
    DANGER = "#EF4444"         # 危险红
    INFO = "#06B6D4"           # 信息青

    # 背景色
    BG_PRIMARY = "#FFFFFF"     # 主背景
    BG_SECONDARY = "#F8FAFC"   # 次要背景
    BG_ACCENT = "#F1F5F9"      # 强调背景

    # 文本色
    TEXT_PRIMARY = "#0F172A"   # 主文本
    TEXT_SECONDARY = "#475569" # 次要文本
    TEXT_MUTE = "#94A3B8"      # 弱化文本

    # 边框
    BORDER = "#E2E8F0"
    BORDER_HOVER = "#CBD5E1"

    # 画布
    CANVAS_BG = "#F8FAFC"
    CANVAS_GRID = "#E2E8F0"

    # 字体
    FONT_FAMILY = "Segoe UI"

    # 插槽颜色
    SOCKET_DEFAULT = "#10B981"
    SOCKET_SELECTED = "#F59E0B"
    SOCKET_HANDLE = "#FFFFFF"


# ==========================================
# 3. 生成器
# ==========================================

class CastleGenerator:
    """城堡生成器 - 封装生成逻辑"""

    @staticmethod
    def generate(templates: List[TileConfig], max_tiles: int, seed: int) -> List[PlacedTile]:
        """生成城堡布局"""
        random.seed(seed)
        valid_templates = [t for t in templates if t.sockets]
        if not valid_templates:
            return []

        placed_tiles = [PlacedTile(random.choice(valid_templates), 0, 0)]
        open_sockets = [(placed_tiles[0], s) for s in placed_tiles[0].config.sockets]

        attempts = 0
        max_attempts = max_tiles * 20

        while (len(placed_tiles) < max_tiles and
               open_sockets and
               attempts < max_attempts):
            attempts += 1
            idx = random.randint(0, len(open_sockets) - 1)
            parent_tile, parent_socket = open_sockets.pop(idx)
            target_dir = parent_socket.get_opposite_dir()

            candidates = [
                (t, s) for t in valid_templates
                for s in t.sockets
                if s.direction == target_dir and
                abs(s.length - parent_socket.length) < 1.0
            ]

            if not candidates:
                continue

            new_template, new_socket = random.choice(candidates)
            new_pos = CastleGenerator._calculate_position(
                parent_tile, parent_socket, new_template, new_socket
            )

            candidate = PlacedTile(new_template, new_pos[0], new_pos[1])
            if not any(candidate.intersects(pt) for pt in placed_tiles):
                placed_tiles.append(candidate)
                open_sockets.extend([
                    (candidate, s) for s in new_template.sockets
                    if s != new_socket
                ])

        return placed_tiles

    @staticmethod
    def _calculate_position(
        parent_tile: PlacedTile,
        parent_socket: SocketConfig,
        new_template: TileConfig,
        new_socket: SocketConfig
    ) -> Tuple[float, float]:
        """计算新 tile 的位置"""
        if parent_socket.direction == 'U':
            nx = parent_tile.x + parent_socket.inset_start - new_socket.inset_start
            ny = parent_tile.y - new_template.height
        elif parent_socket.direction == 'D':
            nx = parent_tile.x + parent_socket.inset_start - new_socket.inset_start
            ny = parent_tile.y + parent_tile.config.height
        elif parent_socket.direction == 'L':
            nx = parent_tile.x - new_template.width
            ny = parent_tile.y + parent_socket.inset_start - new_socket.inset_start
        else:  # 'R'
            nx = parent_tile.x + parent_tile.config.width
            ny = parent_tile.y + parent_socket.inset_start - new_socket.inset_start

        return nx, ny


# ==========================================
# 4. 渲染器
# ==========================================

class CanvasRenderer:
    """画布渲染器 - 统一渲染逻辑"""

    def __init__(self, canvas: tk.Canvas, theme: UITheme = None):
        self.canvas = canvas
        self.theme = theme or UITheme()
        self._image_cache = {}

    def draw_grid(self, cell_size: int = 20):
        """绘制背景网格"""
        self.canvas.delete("grid")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        for x in range(0, w, cell_size):
            self.canvas.create_line(x, 0, x, h, fill=self.theme.CANVAS_GRID, tags="grid")
        for y in range(0, h, cell_size):
            self.canvas.create_line(0, y, w, y, fill=self.theme.CANVAS_GRID, tags="grid")
        self.canvas.tag_lower("grid")

    def draw_tile_with_sockets(
        self,
        tile: TileConfig,
        offset_x: float,
        offset_y: float,
        selected_socket: Optional[SocketConfig] = None,
        show_handles: bool = True,
        scale: float = 1.0
    ):
        """绘制 tile 和插槽"""
        # 绘制 tile 主体
        x1, y1 = offset_x, offset_y
        x2, y2 = offset_x + tile.width * scale, offset_y + tile.height * scale

        # 尝试使用自定义图片
        img = tile.get_image(int(tile.width), int(tile.height))
        if img:
            self.canvas.create_image(
                (x1 + x2) / 2, (y1 + y2) / 2,
                image=img, tags=("tile", "tile_body"),
                anchor=tk.CENTER
            )
            # 缓存图片防止被垃圾回收
            cache_key = f"{tile.name}_{offset_x}_{offset_y}"
            self._image_cache[cache_key] = img
            outline = self.theme.PRIMARY
            width = 2
        else:
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=tile.color,
                outline=self.theme.TEXT_PRIMARY,
                width=2,
                tags=("tile", "tile_body")
            )
            outline = tile.color
            width = 3

        # 绘制调整手柄
        if show_handles:
            handle_size = 8 * scale
            self.canvas.create_rectangle(
                x2 - handle_size, y2 - handle_size,
                x2 + handle_size, y2 + handle_size,
                fill=self.theme.DANGER,
                outline=self.theme.BG_PRIMARY,
                width=2,
                tags=("tile", "resize_handle")
            )

        # 绘制插槽
        for socket in tile.sockets:
            self._draw_socket(socket, tile, offset_x, offset_y, selected_socket, scale)

    def _draw_socket(
        self,
        socket: SocketConfig,
        tile: TileConfig,
        offset_x: float,
        offset_y: float,
        selected_socket: Optional[SocketConfig],
        scale: float = 1.0
    ):
        """绘制单个插槽"""
        ox, oy = offset_x, offset_y
        t = tile
        s = socket

        # 颜色和宽度
        color = self.theme.SOCKET_SELECTED if s == selected_socket else self.theme.SOCKET_DEFAULT
        line_width = 10 if s == selected_socket else 6
        handle_size = 6

        # 计算起点和终点
        coords = self._get_socket_coords(s, t, ox, oy, scale)

        # 绘制线条
        line_tag = f"socket_line_{id(s)}"
        self.canvas.create_line(
            *coords,
            fill=color,
            width=line_width,
            capstyle=tk.ROUND,
            tags=("socket_line", line_tag)
        )

        # 绘制控制手柄
        for idx in [0, 1]:
            handle_type = 'start' if idx == 0 else 'end'
            hx, hy = coords[idx * 2], coords[idx * 2 + 1]
            handle_tag = f"handle_{id(s)}_{handle_type}"
            self.canvas.create_rectangle(
                hx - handle_size, hy - handle_size,
                hx + handle_size, hy + handle_size,
                fill=self.theme.SOCKET_HANDLE,
                outline=self.theme.TEXT_PRIMARY,
                width=2,
                tags=("socket_handle", handle_tag)
            )

    def _get_socket_coords(
        self,
        socket: SocketConfig,
        tile: TileConfig,
        offset_x: float,
        offset_y: float,
        scale: float = 1.0
    ) -> Tuple[float, float, float, float]:
        """获取插槽的坐标"""
        ox, oy = offset_x, offset_y
        t = tile
        s = socket

        if s.direction == 'U':
            return (ox + s.inset_start * scale, oy,
                    ox + s.inset_end * scale, oy)
        elif s.direction == 'D':
            return (ox + s.inset_start * scale, oy + t.height * scale,
                    ox + s.inset_end * scale, oy + t.height * scale)
        elif s.direction == 'L':
            return (ox, oy + s.inset_start * scale,
                    ox, oy + s.inset_end * scale)
        else:  # 'R'
            return (ox + t.width * scale, oy + s.inset_start * scale,
                    ox + t.width * scale, oy + s.inset_end * scale)

    def clear(self):
        """清空画布"""
        self.canvas.delete("all")
        self._image_cache.clear()


# ==========================================
# 5. 样式管理器
# ==========================================

class StyleManager:
    """统一管理所有 ttk 样式"""

    @staticmethod
    def apply_styles(root: tk.Tk):
        """应用所有样式"""
        theme = UITheme()
        style = ttk.Style()

        # 配置主样式
        style.theme_use('clam')

        # Frame 样式
        style.configure("TFrame", background=theme.BG_SECONDARY)
        style.configure("Card.TFrame", background=theme.BG_PRIMARY, relief=tk.RAISED)

        # Label 样式
        style.configure("TLabel",
                       background=theme.BG_SECONDARY,
                       foreground=theme.TEXT_PRIMARY,
                       font=(theme.FONT_FAMILY, 9))
        style.configure("Header.TLabel",
                       font=(theme.FONT_FAMILY, 12, "bold"),
                       foreground=theme.TEXT_PRIMARY)
        style.configure("Subheader.TLabel",
                       font=(theme.FONT_FAMILY, 10, "bold"),
                       foreground=theme.TEXT_SECONDARY)
        style.configure("Muted.TLabel",
                       font=(theme.FONT_FAMILY, 8),
                       foreground=theme.TEXT_MUTE)

        # Button 样式
        button_styles = {
            "Modern.primary.TButton": (theme.PRIMARY, "white"),
            "Modern.success.TButton": (theme.SUCCESS, "white"),
            "Modern.warning.TButton": (theme.WARNING, "white"),
            "Modern.danger.TButton": (theme.DANGER, "white"),
            "Modern.secondary.TButton": (theme.SECONDARY, "white"),
        }

        for style_name, (bg, fg) in button_styles.items():
            style.configure(style_name,
                          background=bg,
                          foreground=fg,
                          font=(theme.FONT_FAMILY, 9),
                          borderwidth=0,
                          focuscolor="none")
            style.map(style_name,
                     background=[('active', StyleManager._darker_color(bg))])

        # Entry 样式
        style.configure("TEntry",
                       fieldbackground=theme.BG_PRIMARY,
                       bordercolor=theme.BORDER,
                       insertcolor=theme.PRIMARY,
                       font=(theme.FONT_FAMILY, 9))

        # Treeview 样式
        style.configure("Treeview",
                       background=theme.BG_PRIMARY,
                       foreground=theme.TEXT_PRIMARY,
                       fieldbackground=theme.BG_PRIMARY,
                       font=(theme.FONT_FAMILY, 9),
                       rowheight=28)
        style.configure("Treeview.Heading",
                       background=theme.BG_ACCENT,
                       foreground=theme.TEXT_PRIMARY,
                       font=(theme.FONT_FAMILY, 9, "bold"))

        # Notebook (Tab) 样式
        style.configure("TNotebook",
                       background=theme.BG_SECONDARY,
                       borderwidth=0)
        style.configure("TNotebook.Tab",
                       background=theme.BG_ACCENT,
                       foreground=theme.TEXT_SECONDARY,
                       padding=(16, 8),
                       font=(theme.FONT_FAMILY, 10))
        style.map("TNotebook.Tab",
                 background=[('selected', theme.BG_PRIMARY)],
                 foreground=[('selected', theme.TEXT_PRIMARY)],
                 expand=[('selected', [1, 1, 1, 0])])
        style.configure("TNotebook.Frame",
                       background=theme.BG_PRIMARY)

    @staticmethod
    def _darker_color(hex_color: str, factor: float = 0.85) -> str:
        """返回更深的颜色"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darker = tuple(int(c * factor) for c in rgb)
        return f'#{darker[0]:02x}{darker[1]:02x}{darker[2]:02x}'


# ==========================================
# 6. 检查器面板
# ==========================================

class InspectorPanel:
    """属性检查器面板"""

    def __init__(self, parent, on_apply_callback):
        self.parent = parent
        self.on_apply_callback = on_apply_callback
        self.theme = UITheme()
        self._current_tile = None
        self._selected_socket = None
        self._setup_ui()

    def _setup_ui(self):
        """设置 UI"""
        # 主容器
        container = ttk.Frame(self.parent, style="Card.TFrame", padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        # 标题
        ttk.Label(container, text="属性检查器", style="Header.TLabel").pack(fill=tk.X, pady=(0, 12))
        ttk.Separator(container, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 12))

        # Tile 属性
        self._create_tile_properties(container)

        ttk.Separator(container, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        # 插槽属性
        self._create_socket_properties(container)

        ttk.Separator(container, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        # 应用按钮
        ttk.Button(
            container,
            text="✓ 应用更改",
            command=self._on_apply,
            style="Modern.success.TButton"
        ).pack(fill=tk.X, pady=8)

        ttk.Label(
            container,
            text="💡 提示：在画布中点击插槽可选中编辑",
            style="Muted.TLabel",
            wraplength=240
        ).pack(anchor=tk.W)

    def _create_tile_properties(self, parent):
        """创建 Tile 属性区域"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(0, 8))

        # 宽度
        ttk.Label(frame, text="宽度:", style="TLabel").grid(row=0, column=0, sticky=tk.W, pady=4)
        self._var_width = tk.StringVar()
        ttk.Entry(frame, textvariable=self._var_width, width=8).grid(row=0, column=1, padx=4, pady=4)

        # 高度
        ttk.Label(frame, text="高度:", style="TLabel").grid(row=0, column=2, sticky=tk.W, pady=4)
        self._var_height = tk.StringVar()
        ttk.Entry(frame, textvariable=self._var_height, width=8).grid(row=0, column=3, padx=4, pady=4)

        # 颜色按钮
        self._btn_color = tk.Button(
            frame,
            text="🎨",
            command=self._on_color_pick,
            bg=self.theme.SECONDARY,
            fg="white",
            width=3,
            relief=tk.FLAT
        )
        self._btn_color.grid(row=0, column=4, padx=4, pady=4)

        # 图片按钮
        ttk.Button(
            frame,
            text="🖼️",
            command=self._on_image_pick,
            style="Modern.secondary.TButton",
            width=4
        ).grid(row=0, column=5, padx=4, pady=4)

    def _create_socket_properties(self, parent):
        """创建插槽属性区域"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(0, 8))

        # 起点
        ttk.Label(frame, text="起点:", style="TLabel").grid(row=0, column=0, sticky=tk.W, pady=4)
        self._var_socket_start = tk.StringVar()
        self._entry_socket_start = ttk.Entry(frame, textvariable=self._var_socket_start, width=8, state="disabled")
        self._entry_socket_start.grid(row=0, column=1, padx=4, pady=4)

        # 终点
        ttk.Label(frame, text="终点:", style="TLabel").grid(row=0, column=2, sticky=tk.W, pady=4)
        self._var_socket_end = tk.StringVar()
        self._entry_socket_end = ttk.Entry(frame, textvariable=self._var_socket_end, width=8, state="disabled")
        self._entry_socket_end.grid(row=0, column=3, padx=4, pady=4)

    def _on_color_pick(self):
        """颜色选择"""
        color = colorchooser.askcolor()[1]
        if color and self._current_tile:
            self._current_tile.color = color
            self._btn_color.config(bg=color)
            self.on_apply_callback()

    def _on_image_pick(self):
        """图片选择"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("PNG 图片", "*.png"),
                ("JPEG 图片", "*.jpg;*.jpeg"),
                ("所有文件", "*.*")
            ]
        )
        if file_path and self._current_tile:
            # 设置图片路径
            self._current_tile.image_path = file_path

            # 获取图片尺寸
            try:
                img = Image.open(file_path)
                self._current_tile.width = img.width
                self._current_tile.height = img.height
            except Exception:
                pass

            # 自动检测边
            sockets = EdgeDetector.detect_edges(file_path)
            if sockets:
                self._current_tile.sockets = sockets

            self.on_apply_callback()

    def _on_apply(self):
        """应用更改"""
        if self.on_apply_callback:
            self.on_apply_callback()

    def update(self, tile: Optional[TileConfig], selected_socket: Optional[SocketConfig]):
        """更新面板数据"""
        self._current_tile = tile
        self._selected_socket = selected_socket

        if tile:
            self._var_width.set(f"{tile.width:.1f}")
            self._var_height.set(f"{tile.height:.1f}")
            self._btn_color.config(bg=tile.color)
        else:
            self._var_width.set("")
            self._var_height.set("")

        if selected_socket:
            self._entry_socket_start.config(state="normal")
            self._entry_socket_end.config(state="normal")
            self._var_socket_start.set(f"{selected_socket.inset_start:.1f}")
            self._var_socket_end.set(f"{selected_socket.inset_end:.1f}")
        else:
            self._entry_socket_start.config(state="disabled")
            self._entry_socket_end.config(state="disabled")
            self._var_socket_start.set("")
            self._var_socket_end.set("")

    def get_values(self) -> Optional[Tuple[float, float, Optional[Tuple[float, float]]]]:
        """获取当前面板的值"""
        if not self._current_tile:
            return None

        try:
            width = max(20.0, float(self._var_width.get()))
            height = max(20.0, float(self._var_height.get()))

            socket_values = None
            if self._selected_socket:
                start = float(self._var_socket_start.get())
                end = float(self._var_socket_end.get())
                socket_values = (start, end)

            return (width, height, socket_values)
        except ValueError:
            return None


# ==========================================
# 7. Tile 列表面板
# ==========================================

class TileLibraryPanel:
    """Tile 库面板"""

    def __init__(self, parent, on_tile_select, on_add, on_delete):
        self.parent = parent
        self.on_tile_select = on_tile_select
        self.on_add = on_add
        self.on_delete = on_delete
        self._setup_ui()

    def _setup_ui(self):
        """设置 UI"""
        container = ttk.Frame(self.parent, style="Card.TFrame", padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        # 标题
        ttk.Label(container, text="Tile 库", style="Header.TLabel").pack(fill=tk.X, pady=(0, 8))

        # 树形列表
        self._tree = ttk.Treeview(container, columns=("name", "size"), show="headings", height=10)
        self._tree.heading("name", text="名称")
        self._tree.heading("size", text="尺寸")
        self._tree.column("name", width=120)
        self._tree.column("size", width=60)
        self._tree.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        # 按钮组
        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="➕ 新建", command=self.on_add, style="Modern.primary.TButton").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(btn_frame, text="🗑️ 删除", command=self.on_delete, style="Modern.danger.TButton").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

    def _on_select(self, event):
        """选择事件"""
        selection = self._tree.selection()
        if selection:
            index = int(selection[0].replace("tile_", ""))
            if self.on_tile_select:
                self.on_tile_select(index)

    def update(self, templates: List[TileConfig]):
        """更新列表"""
        self._tree.delete(*self._tree.get_children())

        for i, tile in enumerate(templates):
            image_indicator = " 🖼️" if tile.image_path else ""
            self._tree.insert("", tk.END, iid=f"tile_{i}", values=(
                f"{tile.name}{image_indicator}",
                f"{tile.width:.0f}x{tile.height:.0f}"
            ))


# ==========================================
# 8. 主应用
# ==========================================

class TileEditorApp:
    """Tile 编辑器主应用"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("专业 Tile 编辑器")
        self.root.geometry("1600x900")

        # 应用样式
        StyleManager.apply_styles(root)

        # 初始化项目管理器
        self._project_manager = ProjectManager()

        # 初始化数据
        self._init_data()

        # 尝试加载项目
        self._load_project()

        # 设置 UI
        self._setup_ui()

        # 初始渲染
        self._refresh_ui()

    def _init_data(self):
        """初始化数据"""
        self._templates = [
            TileConfig("基础建筑", 100, 100, "#34495E")
        ]
        self._templates[0].sockets = [
            SocketConfig('U', 20, 80),
            SocketConfig('D', 20, 80),
            SocketConfig('L', 20, 80),
            SocketConfig('R', 20, 80)
        ]

        # 每个模板的位置和缩放状态
        self._tile_positions = {id(self._templates[0]): [100, 100]}
        self._tile_scales = {id(self._templates[0]): 1.0}

        self._current_tile = self._templates[0]
        self._selected_socket = None

        self._drag_mode = None
        self._active_socket = None
        self._panning = False
        self._pan_mode = None
        self._pan_start = None

        self._editor_offset = [0, 0]  # 画布全局偏移
        self._seed = 42

        # 鼠标悬停状态
        self._hovered_tile = None
        self._save_pending = False

    def _setup_ui(self):
        """设置 UI"""
        # 主容器
        main_container = tk.Frame(self.root, bg=UITheme.BG_SECONDARY)
        main_container.pack(fill=tk.BOTH, expand=True)

        # 侧边栏容器
        sidebar = tk.Frame(main_container, bg=UITheme.BG_SECONDARY, width=320)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # Tile 库面板
        self._tile_library = TileLibraryPanel(
            sidebar,
            on_tile_select=self._on_tile_select,
            on_add=self._add_tile,
            on_delete=self._delete_tile
        )

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # 检查器面板
        self._inspector = InspectorPanel(sidebar, on_apply_callback=self._apply_properties)

        # 主内容区 - 使用 Notebook 实现 Tab 切换
        content = tk.Frame(main_container, bg=UITheme.BG_SECONDARY)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 创建 Notebook (Tab 控件)
        self._notebook = ttk.Notebook(content)
        self._notebook.pack(fill=tk.BOTH, expand=True)

        # 编辑器 Tab
        editor_frame = tk.Frame(self._notebook, bg=UITheme.BG_SECONDARY)
        self._notebook.add(editor_frame, text="📐 交互编辑器")
        self._create_editor_panel(editor_frame)

        # 预览 Tab
        preview_frame = tk.Frame(self._notebook, bg=UITheme.BG_SECONDARY)
        self._notebook.add(preview_frame, text="🏰 实时预览")
        self._create_preview_panel(preview_frame)

        # 绑定 Tab 切换事件
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _create_editor_panel(self, parent):
        """创建编辑器面板"""
        container = tk.Frame(parent, bg=UITheme.BG_PRIMARY)
        container.pack(fill=tk.BOTH, expand=True)

        # 工具栏
        toolbar = tk.Frame(container, bg=UITheme.BG_ACCENT, height=40)
        toolbar.pack(fill=tk.X)
        toolbar.pack_propagate(False)

        tk.Label(
            toolbar,
            text="交互编辑器",
            bg=UITheme.BG_ACCENT,
            fg=UITheme.TEXT_PRIMARY,
            font=(UITheme.FONT_FAMILY, 11, "bold")
        ).pack(side=tk.LEFT, padx=12, pady=8)

        tk.Label(
            toolbar,
            text="双击边界添加插槽 | 右键删除插槽 | 按住空格或中键拖动画布",
            bg=UITheme.BG_ACCENT,
            fg=UITheme.TEXT_MUTE,
            font=(UITheme.FONT_FAMILY, 8)
        ).pack(side=tk.RIGHT, padx=12, pady=8)

        # 画布
        self._editor_canvas = tk.Canvas(container, bg=UITheme.CANVAS_BG)
        self._editor_canvas.pack(fill=tk.BOTH, expand=True)

        # 创建渲染器
        self._editor_renderer = CanvasRenderer(self._editor_canvas)

        # 绑定事件
        self._editor_canvas.bind("<Button-1>", self._on_editor_click)
        self._editor_canvas.bind("<B1-Motion>", self._on_editor_drag)
        self._editor_canvas.bind("<ButtonRelease-1>", self._on_editor_release)
        self._editor_canvas.bind("<Double-Button-1>", self._on_editor_double_click)
        self._editor_canvas.bind("<Button-3>", self._on_editor_right_click)
        self._editor_canvas.bind("<Button-2>", self._on_pan_start)
        self._editor_canvas.bind("<B2-Motion>", self._on_pan_motion)
        self._editor_canvas.bind("<ButtonRelease-2>", self._on_pan_release)
        self._editor_canvas.bind("<Motion>", self._on_editor_mouse_move)
        self._editor_canvas.bind("<Configure>", lambda e: self._draw_editor())

        # 键盘事件 - 空格键拖拽
        self.root.bind("<space>", self._on_space_press)
        self.root.bind("<KeyRelease-space>", self._on_space_release)

    def _create_preview_panel(self, parent):
        """创建预览面板"""
        container = tk.Frame(parent, bg=UITheme.BG_PRIMARY)
        container.pack(fill=tk.BOTH, expand=True)

        # 工具栏
        toolbar = tk.Frame(container, bg=UITheme.BG_ACCENT, height=40)
        toolbar.pack(fill=tk.X)
        toolbar.pack_propagate(False)

        tk.Label(
            toolbar,
            text="实时预览",
            bg=UITheme.BG_ACCENT,
            fg=UITheme.TEXT_PRIMARY,
            font=(UITheme.FONT_FAMILY, 11, "bold")
        ).pack(side=tk.LEFT, padx=12, pady=8)

        ttk.Button(
            toolbar,
            text="🎲 刷新",
            command=self._randomize_seed,
            style="Modern.primary.TButton"
        ).pack(side=tk.RIGHT, padx=12, pady=4)

        # 种子标签
        self._seed_label = tk.Label(
            toolbar,
            text=f"种子: {self._seed}",
            bg=UITheme.BG_ACCENT,
            fg=UITheme.TEXT_SECONDARY,
            font=(UITheme.FONT_FAMILY, 9)
        )
        self._seed_label.pack(side=tk.RIGHT, padx=12)

        # 画布
        self._preview_canvas = tk.Canvas(container, bg=UITheme.CANVAS_BG)
        self._preview_canvas.pack(fill=tk.BOTH, expand=True)

        # 创建渲染器
        self._preview_renderer = CanvasRenderer(self._preview_canvas)
        self._preview_canvas.bind("<Configure>", lambda e: self._draw_preview())

    # ==================== 渲染逻辑 ====================

    def _draw_editor(self):
        """绘制编辑器"""
        self._editor_renderer.clear()
        self._editor_renderer.draw_grid()

        # 绘制所有 Tile
        for tile in self._templates:
            tile_id = id(tile)
            pos = self._tile_positions.get(tile_id, [100 + len(self._templates) * 50, 100 + len(self._templates) * 50])
            scale = self._tile_scales.get(tile_id, 1.0)

            # 只为当前选中的 Tile 显示手柄和插槽高亮
            show_handles = (tile == self._current_tile)
            selected_socket = self._selected_socket if tile == self._current_tile else None

            self._editor_renderer.draw_tile_with_sockets(
                tile,
                pos[0] + self._editor_offset[0],
                pos[1] + self._editor_offset[1],
                selected_socket,
                show_handles,
                scale
            )

    def _on_tab_changed(self, event):
        """Tab 切换事件"""
        # 切换到预览 Tab 时自动刷新
        current_tab = self._notebook.index(self._notebook.select())
        if current_tab == 1:  # 预览 Tab
            self._draw_preview()

    def _draw_preview(self):
        """绘制预览"""
        self._preview_renderer.clear()
        self._preview_renderer.draw_grid()

        tiles = CastleGenerator.generate(self._templates, 50, self._seed)
        if not tiles:
            return

        # 计算缩放和平移
        min_x = min(t.x for t in tiles)
        min_y = min(t.y for t in tiles)
        max_x = max(t.x + t.config.width for t in tiles)
        max_y = max(t.y + t.config.height for t in tiles)

        cw = self._preview_canvas.winfo_width() or 600
        ch = self._preview_canvas.winfo_height() or 400

        # 居中显示
        ox = (cw - (max_x - min_x)) / 2 - min_x
        oy = (ch - (max_y - min_y)) / 2 - min_y

        # 绘制所有 tile
        for pt in tiles:
            self._preview_renderer.draw_tile_with_sockets(
                pt.config,
                pt.x + ox,
                pt.y + oy,
                show_handles=False
            )

    def _refresh_ui(self):
        """刷新 UI"""
        self._tile_library.update(self._templates)
        self._inspector.update(self._current_tile, self._selected_socket)
        self._draw_editor()
        self._draw_preview()

        # 自动保存
        self._save_project()

    def _save_project(self):
        """自动保存项目"""
        if not self._save_pending:
            self._save_pending = True
            # 延迟保存，避免频繁写入
            self.root.after(2000, self._do_save_project)

    def _do_save_project(self):
        """执行保存"""
        self._project_manager.save(self._templates, self._tile_positions, self._tile_scales)
        self._save_pending = False

    def _load_project(self):
        """加载项目"""
        result = self._project_manager.load()
        if result:
            templates, positions, scales = result
            self._templates = templates
            self._tile_positions = positions
            self._tile_scales = scales
            if templates:
                self._current_tile = templates[0]

    def _find_tile_at_position(self, x: int, y: int) -> Optional[TileConfig]:
        """查找鼠标位置下的 Tile"""
        # 从上到下查找（反向遍历，找到最上面的）
        for tile in reversed(self._templates):
            tile_id = id(tile)
            pos = self._tile_positions.get(tile_id, [100, 100])
            ox, oy = pos[0] + self._editor_offset[0], pos[1] + self._editor_offset[1]

            if (ox <= x <= ox + tile.width and
                oy <= y <= oy + tile.height):
                return tile
        return None

    # ==================== 事件处理 ====================

    def _on_tile_select(self, index: int):
        """Tile 选择事件"""
        if 0 <= index < len(self._templates):
            self._current_tile = self._templates[index]
            self._selected_socket = None
            # 更新树形列表的选中状态
            self._tile_library._tree.selection_set(f"tile_{index}")
            self._refresh_ui()

    def _add_tile(self):
        """添加 Tile"""
        index = len(self._templates) + 1
        colors = ["#34495E", "#95A5A6", "#3498DB", "#9B59B6", "#1ABC9C"]
        color = random.choice(colors)
        new_tile = TileConfig(f"Tile {index}", 80, 80, color)
        self._templates.append(new_tile)

        # 初始化新 Tile 的位置（错开显示）
        offset_x = 150 + (len(self._templates) - 1) * 120
        offset_y = 150 + (len(self._templates) - 1) * 50
        self._tile_positions[id(new_tile)] = [offset_x, offset_y]
        self._tile_scales[id(new_tile)] = 1.0

        self._refresh_ui()

    def _delete_tile(self):
        """删除 Tile"""
        if len(self._templates) > 1:
            if self._current_tile in self._templates:
                tile_id = id(self._current_tile)
                # 清理位置数据
                if tile_id in self._tile_positions:
                    del self._tile_positions[tile_id]
                if tile_id in self._tile_scales:
                    del self._tile_scales[tile_id]

                index = self._templates.index(self._current_tile)
                self._templates.remove(self._current_tile)
                self._current_tile = self._templates[0]
                self._selected_socket = None
                self._refresh_ui()

    def _apply_properties(self):
        """应用属性更改"""
        values = self._inspector.get_values()
        if values and self._current_tile:
            width, height, socket_values = values
            self._current_tile.width = width
            self._current_tile.height = height

            if socket_values and self._selected_socket:
                start, end = socket_values
                max_len = (self._current_tile.width if self._selected_socket.direction in ['U', 'D']
                          else self._current_tile.height)
                self._selected_socket.inset_start = max(0, min(start, max_len))
                self._selected_socket.inset_end = max(0, min(end, max_len))

            self._refresh_ui()

    def _randomize_seed(self):
        """随机化种子"""
        self._seed = random.randint(1, 9999)
        self._seed_label.config(text=f"种子: {self._seed}")
        self._draw_preview()

    def _on_editor_click(self, event):
        """编辑器点击事件"""
        x, y = event.x, event.y

        # 查找鼠标位置下的 Tile
        clicked_tile = self._find_tile_at_position(x, y)

        # 如果按住 Alt，准备拖动悬停的 Tile
        if event.state & 0x20000:  # Alt 键被按下
            if clicked_tile:
                self._current_tile = clicked_tile
                self._panning = True
                self._pan_mode = "tile"
                self._pan_start = (x, y)
                self._editor_canvas.config(cursor="fleur")
                self._tile_library.update(self._templates)  # 更新选中状态
            return

        # 检查是否点击调整手柄（当前 Tile）
        tile = self._current_tile
        if not tile:
            return

        tile_id = id(tile)
        tile_pos = self._tile_positions.get(tile_id, [100, 100])
        ox, oy = tile_pos[0] + self._editor_offset[0], tile_pos[1] + self._editor_offset[1]

        # 检查是否点击调整手柄
        if (abs(x - (ox + tile.width)) < 15 and
            abs(y - (oy + tile.height)) < 15):
            self._drag_mode = 'resize_tile'
            return

        # 检查点击的元素
        items = self._editor_canvas.find_overlapping(x - 3, y - 3, x + 3, y + 3)

        for item in items:
            tags = self._editor_canvas.gettags(item)

            # 检查手柄
            if "socket_handle" in tags:
                tag_id = next((t for t in tags if t.startswith("handle_")), None)
                if tag_id:
                    socket_id, handle_type = tag_id.split("_")[1:3]
                    socket = self._find_socket_by_id(int(socket_id))
                    if socket:
                        self._active_socket = socket
                        self._drag_mode = handle_type
                        self._selected_socket = socket
                        self._refresh_ui()
                        return

            # 检查插槽线条
            if "socket_line" in tags:
                tag_id = next((t for t in tags if t.startswith("socket_line_")), None)
                if tag_id:
                    socket_id = tag_id.split("_")[2]
                    socket = self._find_socket_by_id(int(socket_id))
                    if socket:
                        self._selected_socket = socket
                        self._refresh_ui()
                        return

            # 检查 tile 主体
            if "tile_body" in tags:
                return

        # 点击空白处，取消选择
        self._selected_socket = None
        self._refresh_ui()

    def _on_editor_drag(self, event):
        """编辑器拖拽事件"""
        # Tile 平移 (Alt + 拖动 Tile 主体)
        if self._panning and self._pan_start:
            dx = event.x - self._pan_start[0]
            dy = event.y - self._pan_start[1]

            # 只移动当前选中的 Tile
            tile_id = id(self._current_tile)
            if tile_id in self._tile_positions:
                self._tile_positions[tile_id][0] += dx
                self._tile_positions[tile_id][1] += dy

            self._pan_start = (event.x, event.y)
            self._draw_editor()
            return

        # 元素拖拽
        if not self._drag_mode or not self._current_tile:
            return

        x, y = event.x, event.y
        tile = self._current_tile
        tile_id = id(tile)
        tile_pos = self._tile_positions.get(tile_id, [100, 100])
        ox, oy = tile_pos[0] + self._editor_offset[0], tile_pos[1] + self._editor_offset[1]

        if self._drag_mode == 'resize_tile':
            tile.width = max(30, x - ox)
            tile.height = max(30, y - oy)

        elif self._active_socket and self._drag_mode in ['start', 'end']:
            socket = self._active_socket
            if socket.direction in ['U', 'D']:
                local_val = max(0, min(x - ox, tile.width))
            else:
                local_val = max(0, min(y - oy, tile.height))

            if self._drag_mode == 'start':
                socket.inset_start = local_val
            else:
                socket.inset_end = local_val

            self._inspector.update(tile, self._selected_socket)

        self._draw_editor()

    def _on_editor_release(self, event):
        """编辑器释放事件"""
        self._drag_mode = None
        self._active_socket = None
        self._panning = False
        self._pan_start = None
        self._editor_canvas.config(cursor="")
        self._refresh_ui()

    # ==================== 画布平移事件 ====================

    def _on_pan_start(self, event):
        """开始平移 - 中键"""
        self._panning = True
        self._pan_mode = "canvas"  # 标记为画布平移
        self._pan_start = (event.x, event.y)
        self._editor_canvas.config(cursor="fleur")

    def _on_pan_motion(self, event):
        """平移中 - 中键拖动"""
        if self._panning and self._pan_start and self._pan_mode == "canvas":
            dx = event.x - self._pan_start[0]
            dy = event.y - self._pan_start[1]
            self._editor_offset[0] += dx
            self._editor_offset[1] += dy
            self._pan_start = (event.x, event.y)
            self._draw_editor()

    def _on_pan_release(self, event):
        """结束平移 - 中键释放"""
        self._panning = False
        self._pan_mode = None
        self._pan_start = None
        self._editor_canvas.config(cursor="")

    def _on_space_press(self, event):
        """空格键按下 - 暂时启用平移"""
        self._editor_canvas.config(cursor="fleur")

    def _on_space_release(self, event):
        """空格键释放"""
        self._editor_canvas.config(cursor="")

    def _on_editor_mouse_move(self, event):
        """鼠标移动事件 - 检测悬停的 Tile"""
        hovered = self._find_tile_at_position(event.x, event.y)

        if hovered != self._hovered_tile:
            self._hovered_tile = hovered
            # 更新光标
            if hovered and (event.state & 0x20000):  # Alt 按下且悬停在 Tile 上
                self._editor_canvas.config(cursor="fleur")
            else:
                self._editor_canvas.config(cursor="")

    def _on_editor_double_click(self, event):
        """编辑器双击事件"""
        x, y = event.x, event.y
        ox, oy = self._editor_offset
        tile = self._current_tile

        if not tile:
            return

        tolerance = 15

        if abs(y - oy) < tolerance and ox < x < ox + tile.width:
            socket = SocketConfig('U', x - ox - 15, x - ox + 15)
        elif abs(y - (oy + tile.height)) < tolerance and ox < x < ox + tile.width:
            socket = SocketConfig('D', x - ox - 15, x - ox + 15)
        elif abs(x - ox) < tolerance and oy < y < oy + tile.height:
            socket = SocketConfig('L', y - oy - 15, y - oy + 15)
        elif abs(x - (ox + tile.width)) < tolerance and oy < y < oy + tile.height:
            socket = SocketConfig('R', y - oy - 15, y - oy + 15)
        else:
            return

        tile.sockets.append(socket)
        self._refresh_ui()

    def _on_editor_right_click(self, event):
        """编辑器右键事件"""
        items = self._editor_canvas.find_overlapping(event.x - 3, event.y - 3, event.x + 3, event.y + 3)

        for item in items:
            tags = self._editor_canvas.gettags(item)
            if "socket_line" in tags:
                tag_id = next((t for t in tags if t.startswith("socket_line_")), None)
                if tag_id:
                    socket_id = tag_id.split("_")[2]
                    socket = self._find_socket_by_id(int(socket_id))
                    if socket and socket in self._current_tile.sockets:
                        self._current_tile.sockets.remove(socket)
                        if self._selected_socket == socket:
                            self._selected_socket = None
                        self._refresh_ui()
                        return

    def _find_socket_by_id(self, socket_id: int) -> Optional[SocketConfig]:
        """根据 ID 查找 socket"""
        for tile in self._templates:
            for socket in tile.sockets:
                if id(socket) == socket_id:
                    return socket
        return None


# ==========================================
# 9. 入口点
# ==========================================

def main():
    """主入口"""
    root = tk.Tk()
    app = TileEditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

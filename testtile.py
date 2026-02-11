import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene, 
                             QGraphicsRectItem, QWidget, QVBoxLayout, QHBoxLayout, 
                             QListWidget, QListWidgetItem, QLabel, QDockWidget, QFrame)
from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QPoint
from PyQt6.QtGui import QColor, QPen, QBrush, QPainter, QAction, QTransform

# ==========================================
# 1. CONSTANTS & CONFIG
# ==========================================
TILE_SIZE = 64
GRID_WIDTH = 20
GRID_HEIGHT = 15
DARK_THEME = """
QMainWindow { background-color: #2b2b2b; }
QDockWidget { color: #dddddd; font-weight: bold; }
QGraphicsView { background-color: #202020; border: none; }
QListWidget { background-color: #333333; color: #dddddd; border: 1px solid #444; }
QListWidget::item:selected { background-color: #4a90e2; }
QLabel { color: #aaaaaa; }
"""

# ==========================================
# 2. LOGIC: TILE DEFINITIONS
# ==========================================
class TileType:
    EMPTY = 0
    DIRT = 1    # Basic static tile
    WALL = 2    # "Smart" Rule tile

class TileData:
    """Stores the actual data for a coordinate"""
    def __init__(self, t_type=TileType.EMPTY):
        self.type = t_type
        self.bitmask = 0  # For rule tiles (0-15 usually)

# ==========================================
# 3. VIEW: THE TILE ITEM (RENDERING)
# ==========================================
class GraphicsTileItem(QGraphicsRectItem):
    """The visual representation of a tile in the scene"""
    def __init__(self, x, y, tile_data):
        super().__init__(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        self.grid_x = x
        self.grid_y = y
        self.update_appearance(tile_data)
        
        # Performance: Don't draw outline for every tile to save calls
        self.setPen(QPen(Qt.PenStyle.NoPen)) 

    def update_appearance(self, tile_data):
        if tile_data.type == TileType.EMPTY:
            self.setBrush(QBrush(Qt.GlobalColor.transparent))
        
        elif tile_data.type == TileType.DIRT:
            self.setBrush(QBrush(QColor("#5d4037"))) # Brown
            
        elif tile_data.type == TileType.WALL:
            # Rule Tile Visualization based on bitmask
            # In a real app, you would select a specific sprite UV here
            color = QColor("#607d8b") # Base Grey
            
            # Simple visualization: Darker if it has connections
            if tile_data.bitmask > 0:
                color = color.darker(100 + (tile_data.bitmask * 5))
            
            self.setBrush(QBrush(color))

# ==========================================
# 4. CONTROLLER: THE EDITOR CANVAS
# ==========================================
class EditorScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.setSceneRect(0, 0, GRID_WIDTH * TILE_SIZE, GRID_HEIGHT * TILE_SIZE)
        self.grid_data = {} # (x,y) -> TileData
        self.visuals = {}   # (x,y) -> GraphicsTileItem
        self.current_tool_tile = TileType.DIRT
        self.is_drawing = False

        # Draw Grid Lines
        self._draw_grid()

    def _draw_grid(self):
        pen = QPen(QColor("#333333"), 1)
        # Vertical lines
        for x in range(GRID_WIDTH + 1):
            self.addLine(x * TILE_SIZE, 0, x * TILE_SIZE, GRID_HEIGHT * TILE_SIZE, pen)
        # Horizontal lines
        for y in range(GRID_HEIGHT + 1):
            self.addLine(0, y * TILE_SIZE, GRID_WIDTH * TILE_SIZE, y * TILE_SIZE, pen)

    def paint_tile(self, gx, gy):
        if not (0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT):
            return

        # 1. Create Data
        t_data = TileData(self.current_tool_tile)
        self.grid_data[(gx, gy)] = t_data

        # 2. Update Visuals
        if (gx, gy) in self.visuals:
            self.removeItem(self.visuals[(gx, gy)])
        
        item = GraphicsTileItem(gx, gy, t_data)
        self.addItem(item)
        self.visuals[(gx, gy)] = item

        # 3. Trigger Smart Tile Update (Autotiling)
        if self.current_tool_tile == TileType.WALL:
            self.update_surrounding_rules(gx, gy)

    def update_surrounding_rules(self, x, y):
        """Updates this tile and its neighbors to handle connections"""
        check_list = [(x, y), (x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        for cx, cy in check_list:
            if (cx, cy) not in self.grid_data: continue
            
            data = self.grid_data[(cx, cy)]
            if data.type != TileType.WALL: continue

            # Calculate Bitmask (4-bit: Top, Right, Bottom, Left)
            mask = 0
            # TOP
            if self.is_same_tile(cx, cy-1, TileType.WALL): mask += 1
            # RIGHT
            if self.is_same_tile(cx+1, cy, TileType.WALL): mask += 2
            # BOTTOM
            if self.is_same_tile(cx, cy+1, TileType.WALL): mask += 4
            # LEFT
            if self.is_same_tile(cx-1, cy, TileType.WALL): mask += 8
            
            data.bitmask = mask
            # Refresh visual
            self.visuals[(cx, cy)].update_appearance(data)

    def is_same_tile(self, x, y, type_to_check):
        return (x, y) in self.grid_data and self.grid_data[(x, y)].type == type_to_check

    # --- Mouse Events handled in Scene for cleaner logic ---
    def mousePressEvent(self, event):
        self.is_drawing = True
        pos = event.scenePos()
        gx, gy = int(pos.x() // TILE_SIZE), int(pos.y() // TILE_SIZE)
        self.paint_tile(gx, gy)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            pos = event.scenePos()
            gx, gy = int(pos.x() // TILE_SIZE), int(pos.y() // TILE_SIZE)
            self.paint_tile(gx, gy)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.is_drawing = False
        super().mouseReleaseEvent(event)

# ==========================================
# 5. UI: MAIN WINDOW
# ==========================================
class EditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python Game Tile Editor")
        self.resize(1200, 800)
        self.setStyleSheet(DARK_THEME)

        # -- Central Canvas --
        self.scene = EditorScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing, False) # Pixel art style
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag) # We handle drag manually
        self.setCentralWidget(self.view)

        # -- Palette Dock (Left) --
        self.palette_dock = QDockWidget("Tile Palette", self)
        self.palette_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.palette_list = QListWidget()
        self.palette_list.setIconSize(QPoint(32, 32))
        
        # Add Items to Palette
        self.add_palette_item("Eraser", TileType.EMPTY, "#202020")
        self.add_palette_item("Dirt (Static)", TileType.DIRT, "#5d4037")
        self.add_palette_item("Wall (Smart/Rule)", TileType.WALL, "#607d8b")
        
        self.palette_list.currentRowChanged.connect(self.change_tool)
        self.palette_dock.setWidget(self.palette_list)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.palette_dock)

        # -- Status Bar --
        self.statusBar().showMessage("Ready. Select a tile and drag on grid.")

    def add_palette_item(self, name, type_id, hex_color):
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, type_id)
        # Create a simple icon for the UI
        item.setBackground(QColor(hex_color))
        self.palette_list.addItem(item)

    def change_tool(self, index):
        item = self.palette_list.item(index)
        tool_id = item.data(Qt.ItemDataRole.UserRole)
        self.scene.current_tool_tile = tool_id
        self.statusBar().showMessage(f"Selected: {item.text()}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EditorWindow()
    window.show()
    sys.exit(app.exec())
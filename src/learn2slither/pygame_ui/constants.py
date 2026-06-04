# UI Constants
CELL_SIZE = 40  # Cell size for standard square layout
GRID_WIDTH = 10
GRID_HEIGHT = 10
BOARD_WIDTH = GRID_WIDTH * CELL_SIZE
BOARD_HEIGHT = GRID_HEIGHT * CELL_SIZE
HEADER_HEIGHT = 80

WINDOW_WIDTH = BOARD_WIDTH
WINDOW_HEIGHT = BOARD_HEIGHT + HEADER_HEIGHT

# Palette (Rustic Retro Monochrome/Green theme - no fancy gradients or slate-indigo)
COLOR_BG_DARK = (20, 20, 20)  # Near black
COLOR_HEADER_BG = (35, 35, 35)  # Dark gray
COLOR_GRID_LINE = (45, 45, 45)  # Muted grid line
COLOR_DIVIDER = (70, 70, 70)  # Gray divider line
COLOR_TEXT_PRIMARY = (230, 230, 230)  # Off-white / light gray
COLOR_TEXT_MUTED = (160, 160, 160)  # Muted gray

# Snake & Apple (Solid, simple colors - no gloss, reflections or fancy gradients)
COLOR_SNAKE_HEAD = (30, 100, 200)  # Darker rustic blue for head
COLOR_SNAKE_BODY = (70, 150, 220)  # Lighter rustic blue for body
COLOR_SNAKE_TAIL = COLOR_SNAKE_BODY  # Keep reference for compatibility
COLOR_GREEN_APPLE = (0, 200, 0)  # Solid green apple
COLOR_RED_APPLE = (200, 0, 0)  # Solid red apple
COLOR_ALERT = (200, 0, 0)  # Red alert

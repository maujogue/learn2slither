import math
import sys
import pygame
from learn2slither.models import Direction, GameOverReason, create_initial_game
from learn2slither.cli import print_vision_grid

# UI Constants
CELL_SIZE = 40  # Increased cell size for a gorgeous compact window
GRID_WIDTH = 10
GRID_HEIGHT = 10
BOARD_WIDTH = GRID_WIDTH * CELL_SIZE
BOARD_HEIGHT = GRID_HEIGHT * CELL_SIZE
HEADER_HEIGHT = 80

WINDOW_WIDTH = BOARD_WIDTH
WINDOW_HEIGHT = BOARD_HEIGHT + HEADER_HEIGHT

# Palette (Rich Slate, Teal-Indigo Gradients, Emerald/Rose highlights)
COLOR_BG_DARK = (15, 23, 42)  # Slate 900
COLOR_HEADER_BG = (30, 41, 59)  # Slate 800
COLOR_GRID_LINE = (30, 41, 59)  # Slate 800 (slightly lighter)
COLOR_DIVIDER = (51, 65, 85)  # Slate 700
COLOR_TEXT_PRIMARY = (248, 250, 252)  # Slate 50
COLOR_TEXT_MUTED = (148, 163, 184)  # Slate 400

# Snake Color Gradient (Teal to Indigo)
COLOR_SNAKE_HEAD = (6, 182, 212)  # Cyan 500
COLOR_SNAKE_TAIL = (99, 102, 241)  # Indigo 500

# Apple Colors
COLOR_GREEN_APPLE = (16, 185, 129)  # Emerald 500
COLOR_GREEN_LEAF = (4, 120, 87)  # Emerald 700
COLOR_RED_APPLE = (239, 68, 68)  # Red 500
COLOR_RED_STEM = (120, 53, 15)  # Amber 900 (Brown)
COLOR_APPLE_GLOSS = (255, 255, 255)  # White highlight

# Alert Color
COLOR_ALERT = (244, 63, 94)  # Rose 500


class Slider:
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        min_val: int,
        max_val: int,
        current_val: int,
        label: str,
    ):
        self.x = x
        self.y = y
        self.width = width
        self.min_val = min_val
        self.max_val = max_val
        self.current_val = current_val
        self.label = label
        self.is_dragging = False

    def draw(self, screen, font_label, font_val, text_color, track_color, handle_color):
        # Draw label
        lbl_surf = font_label.render(self.label, True, text_color)
        screen.blit(lbl_surf, (self.x, self.y - 22))

        # Draw current value
        val_surf = font_val.render(str(self.current_val), True, handle_color)
        screen.blit(val_surf, (self.x + self.width - val_surf.get_width(), self.y - 22))

        # Draw track
        pygame.draw.line(
            screen, track_color, (self.x, self.y), (self.x + self.width, self.y), 4
        )

        # Draw handle
        handle_x = self.x + int(
            (self.current_val - self.min_val)
            / (self.max_val - self.min_val)
            * self.width
        )
        pygame.draw.circle(screen, handle_color, (handle_x, self.y), 8)

    def handle_event(self, event, mouse_pos) -> bool:
        """Returns True if the value changed."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # Check if click is on the track/handle area
                track_rect = pygame.Rect(self.x - 10, self.y - 12, self.width + 20, 24)
                if track_rect.collidepoint(mouse_pos):
                    self.is_dragging = True
                    old_val = self.current_val
                    self.update_val_from_mouse(mouse_pos[0])
                    return self.current_val != old_val

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.is_dragging = False

        elif event.type == pygame.MOUSEMOTION:
            if self.is_dragging:
                old_val = self.current_val
                self.update_val_from_mouse(mouse_pos[0])
                return self.current_val != old_val

        return False

    def update_val_from_mouse(self, mouse_x: int):
        rel_x = max(0, min(self.width, mouse_x - self.x))
        fraction = rel_x / self.width
        raw_val = self.min_val + fraction * (self.max_val - self.min_val)
        self.current_val = int(round(raw_val))


def run_game(initial_width: int = 10, initial_height: int = 10, initial_speed: int = 6):
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("Learn2Slither - Playable Snake Game")

    # Dynamic layout constants
    SIDEBAR_WIDTH = 300

    def get_window_size(w, h):
        board_w = w * CELL_SIZE
        board_h = h * CELL_SIZE
        win_w = board_w + SIDEBAR_WIDTH
        win_h = max(board_h + HEADER_HEIGHT, 460)
        return win_w, win_h, board_w, board_h

    # Initialize current dimensions and speed
    grid_width = initial_width
    grid_height = initial_height
    speed = initial_speed

    win_w, win_h, board_w, board_h = get_window_size(grid_width, grid_height)
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()

    # Load elegant fonts with system fallbacks
    font_title = pygame.font.SysFont(
        "Inter, Helvetica, Arial, sans-serif", 28, bold=True
    )
    font_title_sm = pygame.font.SysFont(
        "Inter, Helvetica, Arial, sans-serif", 20, bold=True
    )
    font_subtitle = pygame.font.SysFont("Inter, Helvetica, Arial, sans-serif", 16)
    font_gameover = pygame.font.SysFont(
        "Inter, Helvetica, Arial, sans-serif", 38, bold=True
    )
    font_gameover_sub = pygame.font.SysFont("Inter, Helvetica, Arial, sans-serif", 18)

    # Initialize game state
    state = create_initial_game(width=grid_width, height=grid_height)
    game_started = False

    # Instantiate Sliders (Width: 5-25, Height: 5-20, Speed: 2-20)
    slider_width = Slider(
        board_w + 20,
        HEADER_HEIGHT + 90,
        SIDEBAR_WIDTH - 40,
        5,
        25,
        grid_width,
        "Board Width",
    )
    slider_height = Slider(
        board_w + 20,
        HEADER_HEIGHT + 160,
        SIDEBAR_WIDTH - 40,
        5,
        25,
        grid_height,
        "Board Height",
    )
    slider_speed = Slider(
        board_w + 20,
        HEADER_HEIGHT + 230,
        SIDEBAR_WIDTH - 40,
        1,
        20,
        speed,
        "Speed (steps/s)",
    )

    # Print initial state vision matrix
    print("\n" + "=" * 60)
    print(f"🎮 GAME START - Initial Vision ({grid_width}x{grid_height}) 🎮")
    print_vision_grid(state)
    print("=" * 60)

    # Pygame Custom Timer Event for moving the snake
    MOVE_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(MOVE_EVENT, int(1000 / speed))

    running = True
    while running:
        # Event Loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            elif event.type == MOVE_EVENT:
                if game_started and not state.is_game_over:
                    state.step()
                    # Output State Vision to terminal on every tick
                    print_vision_grid(state)
                    print(f"Snake Length: {len(state.snake.body)}")
                    if state.is_game_over:
                        print("\nGAME OVER!")

            elif event.type == pygame.KEYDOWN:
                if state.is_game_over:
                    if event.key == pygame.K_SPACE:
                        # Restart the game with the CURRENT slider values!
                        grid_width = slider_width.current_val
                        grid_height = slider_height.current_val
                        speed = slider_speed.current_val

                        win_w, win_h, board_w, board_h = get_window_size(
                            grid_width, grid_height
                        )
                        screen = pygame.display.set_mode((win_w, win_h))

                        # Reposition sliders
                        slider_width.x = board_w + 20
                        slider_height.x = board_w + 20
                        slider_speed.x = board_w + 20

                        state = create_initial_game(
                            width=grid_width, height=grid_height
                        )
                        game_started = False

                        pygame.time.set_timer(MOVE_EVENT, int(1000 / speed))

                        print("\n" + "=" * 60)
                        print(
                            f"🎮 GAME RESTARTED - {grid_width}x{grid_height} at speed {speed} moves/sec 🎮"
                        )
                        print_vision_grid(state)
                        print("=" * 60)
                else:
                    direction_map = {
                        pygame.K_UP: Direction.UP,
                        pygame.K_w: Direction.UP,
                        pygame.K_DOWN: Direction.DOWN,
                        pygame.K_s: Direction.DOWN,
                        pygame.K_LEFT: Direction.LEFT,
                        pygame.K_a: Direction.LEFT,
                        pygame.K_RIGHT: Direction.RIGHT,
                        pygame.K_d: Direction.RIGHT,
                    }
                    if event.key in direction_map:
                        requested_dir = direction_map[event.key]

                        # Prevent setting started/direction if the user presses a reversal key
                        is_reversal = False
                        if len(state.snake.body) > 1:
                            neck = state.snake.body[1]
                            head = state.snake.head
                            if head.move(requested_dir) == neck:
                                is_reversal = True

                        if not is_reversal:
                            state.change_direction(requested_dir)
                            game_started = True

            # Handle slider and apply button mouse interactions
            elif event.type in (
                pygame.MOUSEBUTTONDOWN,
                pygame.MOUSEBUTTONUP,
                pygame.MOUSEMOTION,
            ):
                mouse_pos = pygame.mouse.get_pos()

                # Handle speed slider (immediate effect)
                if slider_speed.handle_event(event, mouse_pos):
                    speed = slider_speed.current_val
                    pygame.time.set_timer(MOVE_EVENT, int(1000 / speed))

                # Handle other sliders
                slider_width.handle_event(event, mouse_pos)
                slider_height.handle_event(event, mouse_pos)

                # Check Apply Button click
                btn_rect = pygame.Rect(
                    board_w + 20, HEADER_HEIGHT + 290, SIDEBAR_WIDTH - 40, 45
                )
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_rect.collidepoint(mouse_pos):
                        # Apply width, height, and speed setting immediately
                        grid_width = slider_width.current_val
                        grid_height = slider_height.current_val
                        speed = slider_speed.current_val

                        win_w, win_h, board_w, board_h = get_window_size(
                            grid_width, grid_height
                        )
                        screen = pygame.display.set_mode((win_w, win_h))

                        # Reposition sliders
                        slider_width.x = board_w + 20
                        slider_height.x = board_w + 20
                        slider_speed.x = board_w + 20

                        state = create_initial_game(
                            width=grid_width, height=grid_height
                        )
                        game_started = False

                        pygame.time.set_timer(MOVE_EVENT, int(1000 / speed))

                        print("\n" + "=" * 60)
                        print(
                            f"🎮 SETTINGS APPLIED & RESTARTED - {grid_width}x{grid_height} at speed {speed} moves/sec 🎮"
                        )
                        print_vision_grid(state)
                        print("=" * 60)

        # Clear Screen with Slate 900
        screen.fill(COLOR_BG_DARK)

        # ----------------- DRAW HEADER -----------------
        # Header Background spanning across win_w
        pygame.draw.rect(screen, COLOR_HEADER_BG, (0, 0, win_w, HEADER_HEIGHT))
        # Divider Line
        pygame.draw.line(
            screen,
            COLOR_DIVIDER,
            (0, HEADER_HEIGHT - 1),
            (win_w, HEADER_HEIGHT - 1),
            2,
        )

        # Text: Score (Snake Length)
        score_text = font_title.render(
            f"SCORE: {len(state.snake.body)}", True, COLOR_TEXT_PRIMARY
        )
        screen.blit(score_text, (15, (HEADER_HEIGHT - score_text.get_height()) // 2))

        # Text: Green and Red Apple legends / Status (anchored relative to win_w)
        status_y = (HEADER_HEIGHT - 20) // 2
        # Green apple icon indicator
        pygame.draw.circle(screen, COLOR_GREEN_APPLE, (win_w - 180, status_y + 10), 6)
        pygame.draw.circle(screen, COLOR_APPLE_GLOSS, (win_w - 182, status_y + 8), 1.5)
        green_lbl = font_subtitle.render("Grow (+1)", True, COLOR_TEXT_MUTED)
        screen.blit(green_lbl, (win_w - 168, status_y + 1))

        # Red apple icon indicator
        pygame.draw.circle(screen, COLOR_RED_APPLE, (win_w - 90, status_y + 10), 6)
        pygame.draw.circle(screen, COLOR_APPLE_GLOSS, (win_w - 92, status_y + 8), 1.5)
        red_lbl = font_subtitle.render("Shrink (-1)", True, COLOR_TEXT_MUTED)
        screen.blit(red_lbl, (win_w - 78, status_y + 1))

        # ----------------- DRAW GRID BOARD -----------------
        # Draw background Grid Lines (within board_w and board_h bounds)
        for x in range(grid_width + 1):
            dx = x * CELL_SIZE
            pygame.draw.line(
                screen,
                COLOR_GRID_LINE,
                (dx, HEADER_HEIGHT),
                (dx, HEADER_HEIGHT + board_h),
            )
        for y in range(grid_height + 1):
            dy = HEADER_HEIGHT + y * CELL_SIZE
            pygame.draw.line(screen, COLOR_GRID_LINE, (0, dy), (board_w, dy))

        # ----------------- DRAW APPLES -----------------
        # Draw Green Apples
        for apple in state.green_apples:
            cx = apple.x * CELL_SIZE + CELL_SIZE // 2
            cy = HEADER_HEIGHT + apple.y * CELL_SIZE + CELL_SIZE // 2
            # Draw leaf
            pygame.draw.circle(screen, COLOR_GREEN_LEAF, (cx + 5, cy - 8), 5)
            # Draw main body
            pygame.draw.circle(screen, COLOR_GREEN_APPLE, (cx, cy), CELL_SIZE // 2 - 3)
            # Draw shiny gloss
            pygame.draw.circle(screen, COLOR_APPLE_GLOSS, (cx - 4, cy - 4), 4)

        # Draw Red Apples
        for apple in state.red_apples:
            cx = apple.x * CELL_SIZE + CELL_SIZE // 2
            cy = HEADER_HEIGHT + apple.y * CELL_SIZE + CELL_SIZE // 2
            # Draw stem
            pygame.draw.line(screen, COLOR_RED_STEM, (cx, cy - 8), (cx + 5, cy - 14), 2)
            # Draw main body
            pygame.draw.circle(screen, COLOR_RED_APPLE, (cx, cy), CELL_SIZE // 2 - 3)
            # Draw shiny gloss
            pygame.draw.circle(screen, COLOR_APPLE_GLOSS, (cx - 4, cy - 4), 4)

        # ----------------- DRAW SNAKE -----------------
        body_len = len(state.snake.body)
        for i, point in enumerate(state.snake.body):
            cx = point.x * CELL_SIZE + CELL_SIZE // 2
            cy = HEADER_HEIGHT + point.y * CELL_SIZE + CELL_SIZE // 2

            # Compute beautiful Cyan-Indigo gradient
            if body_len > 1:
                t = i / (body_len - 1)
            else:
                t = 0.0
            r = int(COLOR_SNAKE_HEAD[0] * (1 - t) + COLOR_SNAKE_TAIL[0] * t)
            g = int(COLOR_SNAKE_HEAD[1] * (1 - t) + COLOR_SNAKE_TAIL[1] * t)
            b = int(COLOR_SNAKE_HEAD[2] * (1 - t) + COLOR_SNAKE_TAIL[2] * t)
            segment_color = (r, g, b)

            # Draw segment as a rounded rectangle for premium styling
            rect = pygame.Rect(
                point.x * CELL_SIZE + 3,
                HEADER_HEIGHT + point.y * CELL_SIZE + 3,
                CELL_SIZE - 6,
                CELL_SIZE - 6,
            )
            pygame.draw.rect(screen, segment_color, rect, border_radius=10)

            # Draw details on the head
            if i == 0:
                # White of eyes
                if state.snake.direction == Direction.UP:
                    eye_l = (cx - 7, cy - 7)
                    eye_r = (cx + 7, cy - 7)
                elif state.snake.direction == Direction.DOWN:
                    eye_l = (cx - 7, cy + 7)
                    eye_r = (cx + 7, cy + 7)
                elif state.snake.direction == Direction.LEFT:
                    eye_l = (cx - 7, cy - 7)
                    eye_r = (cx - 7, cy + 7)
                else:  # Direction.RIGHT
                    eye_l = (cx + 7, cy - 7)
                    eye_r = (cx + 7, cy + 7)

                pygame.draw.circle(screen, (255, 255, 255), eye_l, 5)
                pygame.draw.circle(screen, (255, 255, 255), eye_r, 5)
                # Pupils (black dots)
                pygame.draw.circle(screen, (0, 0, 0), eye_l, 2)
                pygame.draw.circle(screen, (0, 0, 0), eye_r, 2)

        # ----------------- DRAW SIDEBAR -----------------
        # Sidebar background
        sidebar_rect = pygame.Rect(
            board_w, HEADER_HEIGHT, SIDEBAR_WIDTH, win_h - HEADER_HEIGHT
        )
        pygame.draw.rect(screen, COLOR_HEADER_BG, sidebar_rect)
        # Vertical divider line
        pygame.draw.line(
            screen,
            COLOR_DIVIDER,
            (board_w, HEADER_HEIGHT),
            (board_w, win_h),
            2,
        )

        # Title for Sidebar
        lbl_settings = font_title_sm.render("SETTINGS", True, COLOR_TEXT_PRIMARY)
        screen.blit(lbl_settings, (board_w + 20, HEADER_HEIGHT + 20))

        # Draw sliders
        slider_width.draw(
            screen,
            font_subtitle,
            font_subtitle,
            COLOR_TEXT_MUTED,
            COLOR_DIVIDER,
            COLOR_SNAKE_HEAD,
        )
        slider_height.draw(
            screen,
            font_subtitle,
            font_subtitle,
            COLOR_TEXT_MUTED,
            COLOR_DIVIDER,
            COLOR_SNAKE_HEAD,
        )
        slider_speed.draw(
            screen,
            font_subtitle,
            font_subtitle,
            COLOR_TEXT_MUTED,
            COLOR_DIVIDER,
            COLOR_SNAKE_HEAD,
        )

        # Draw Apply Button with hover animation
        mouse_pos = pygame.mouse.get_pos()
        btn_rect = pygame.Rect(
            board_w + 20, HEADER_HEIGHT + 290, SIDEBAR_WIDTH - 40, 45
        )
        is_hovered = btn_rect.collidepoint(mouse_pos)
        btn_color = (
            (79, 70, 229) if is_hovered else (99, 102, 241)
        )  # Slate-Indigo gradient shades

        pygame.draw.rect(screen, btn_color, btn_rect, border_radius=8)
        btn_text = font_subtitle.render("APPLY & RESTART", True, COLOR_TEXT_PRIMARY)
        btn_text_rect = btn_text.get_rect(center=btn_rect.center)
        screen.blit(btn_text, btn_text_rect)

        # ----------------- DRAW WAITING TO START OVERLAY -----------------
        if not game_started and not state.is_game_over:
            # Semi-transparent dark overlay covering the board area
            overlay = pygame.Surface((board_w, board_h))
            overlay.set_alpha(160)
            overlay.fill(COLOR_BG_DARK)
            screen.blit(overlay, (0, HEADER_HEIGHT))

            # Centered elements inside the board area
            center_x = board_w // 2
            center_y = HEADER_HEIGHT + board_h // 2

            # Main Start Prompt
            start_prompt = font_gameover_sub.render(
                "PRESS ANY DIRECTION KEY TO START", True, COLOR_SNAKE_HEAD
            )
            start_prompt_rect = start_prompt.get_rect(center=(center_x, center_y - 15))
            screen.blit(start_prompt, start_prompt_rect)

            # Pulsing sub prompt for controls
            ticks = pygame.time.get_ticks()
            pulse = int(127 + 128 * math.sin(ticks * 0.007))
            sub_color = (pulse, pulse, pulse)

            sub_prompt = font_subtitle.render("W/A/S/D or Arrow keys", True, sub_color)
            sub_prompt_rect = sub_prompt.get_rect(center=(center_x, center_y + 15))
            screen.blit(sub_prompt, sub_prompt_rect)

        # ----------------- DRAW GAME OVER OVERLAY -----------------
        if state.is_game_over:
            # Semi-transparent dark overlay covering the board area
            overlay = pygame.Surface((board_w, board_h))
            overlay.set_alpha(200)
            overlay.fill(COLOR_BG_DARK)
            screen.blit(overlay, (0, HEADER_HEIGHT))

            # Centered elements inside the board area
            center_x = board_w // 2
            center_y = HEADER_HEIGHT + board_h // 2

            # Main Game Over Title
            go_title = font_gameover.render("GAME OVER", True, COLOR_ALERT)
            go_title_rect = go_title.get_rect(center=(center_x, center_y - 40))
            screen.blit(go_title, go_title_rect)

            # Detailed game over reason
            reason_str = "Unknown collision"
            if state.game_over_reason == GameOverReason.WALL_COLLISION:
                reason_str = "Crashed into wall!"
            elif state.game_over_reason == GameOverReason.TAIL_COLLISION:
                reason_str = "Bit your own tail!"
            elif state.game_over_reason == GameOverReason.STARVATION:
                reason_str = "Starved (length hit 0)!"

            go_reason = font_gameover_sub.render(reason_str, True, COLOR_TEXT_MUTED)
            go_reason_rect = go_reason.get_rect(center=(center_x, center_y))
            screen.blit(go_reason, go_reason_rect)

            # Blinking restart call-to-action
            ticks = pygame.time.get_ticks()
            pulse = int(127 + 128 * math.sin(ticks * 0.007))
            restart_color = (pulse, pulse, pulse)

            go_restart = font_gameover_sub.render(
                "Press [SPACE] to Restart", True, restart_color
            )
            go_restart_rect = go_restart.get_rect(center=(center_x, center_y + 40))
            screen.blit(go_restart, go_restart_rect)

        # Flip screen
        pygame.display.flip()
        clock.tick(60)  # Limit rendering frames per second

    pygame.quit()
    sys.exit()

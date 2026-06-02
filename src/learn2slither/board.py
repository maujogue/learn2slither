import math
import os
import sys
import pygame
from learn2slither.models import Direction, GameOverReason, create_initial_game
from learn2slither.cli import print_vision_grid
from learn2slither.agent import QLearningAgent, StateFeatures

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
COLOR_BG_DARK = (20, 20, 20)      # Near black
COLOR_HEADER_BG = (35, 35, 35)    # Dark gray
COLOR_GRID_LINE = (45, 45, 45)    # Muted grid line
COLOR_DIVIDER = (70, 70, 70)      # Gray divider line
COLOR_TEXT_PRIMARY = (230, 230, 230) # Off-white / light gray
COLOR_TEXT_MUTED = (160, 160, 160)   # Muted gray

# Snake & Apple (Solid, simple colors - no gloss, reflections or fancy gradients)
COLOR_SNAKE_HEAD = (30, 100, 200)   # Darker rustic blue for head
COLOR_SNAKE_BODY = (70, 150, 220)   # Lighter rustic blue for body
COLOR_SNAKE_TAIL = COLOR_SNAKE_BODY  # Keep reference for compatibility
COLOR_GREEN_APPLE = (0, 200, 0)   # Solid green apple
COLOR_RED_APPLE = (200, 0, 0)     # Solid red apple
COLOR_ALERT = (200, 0, 0)         # Red alert


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

        # Draw handle as a simple full square block (no circles)
        handle_x = self.x + int(
            (self.current_val - self.min_val)
            / (self.max_val - self.min_val)
            * self.width
        )
        handle_rect = pygame.Rect(handle_x - 6, self.y - 8, 12, 16)
        pygame.draw.rect(screen, handle_color, handle_rect)

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


def run_game(
    initial_width: int = 10,
    initial_height: int = 10,
    initial_speed: int = 6,
    qtable_path: str | None = None,
    autopilot: bool = True,
    training: bool = False,
    episodes: int = 15000,
):
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("Learn2Slither - Playable Snake Game")

    # Dynamic layout constants
    SIDEBAR_WIDTH = 300

    def get_window_size(w, h):
        board_w = w * CELL_SIZE
        board_h = h * CELL_SIZE
        win_w = board_w + SIDEBAR_WIDTH
        win_h = max(board_h + HEADER_HEIGHT, 580)
        return win_w, win_h, board_w, board_h

    def get_min_green_dist(s) -> float:
        if not s.green_apples:
            return 0.0
        h = s.snake.head
        return min(abs(a.x - h.x) + abs(a.y - h.y) for a in s.green_apples)


    # Initialize current dimensions and speed
    grid_width = initial_width
    grid_height = initial_height
    speed = initial_speed

    win_w, win_h, board_w, board_h = get_window_size(grid_width, grid_height)
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()

    # Initialize agent and load pretrained Q-table
    agent = QLearningAgent()
    if qtable_path:
        agent.q_table.load_from_file(qtable_path)
    else:
        try:
            root_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            models_qtable = os.path.join(root_dir, "models", "q_table.json")
            if os.path.exists(models_qtable):
                agent.q_table.load_from_file(models_qtable)
            else:
                legacy_qtable = os.path.join(os.path.dirname(__file__), "q_table.json")
                agent.q_table.load_from_file(legacy_qtable)
        except Exception:
            pass
    ai_mode = autopilot
 

    # Load rustic monospace fonts with system fallbacks
    font_title = pygame.font.SysFont(
        "Courier New, Courier, Monospace, monospace", 24, bold=True
    )
    font_title_sm = pygame.font.SysFont(
        "Courier New, Courier, Monospace, monospace", 16, bold=True
    )
    font_subtitle = pygame.font.SysFont(
        "Courier New, Courier, Monospace, monospace", 14
    )
    font_gameover = pygame.font.SysFont(
        "Courier New, Courier, Monospace, monospace", 32, bold=True
    )
    font_gameover_sub = pygame.font.SysFont(
        "Courier New, Courier, Monospace, monospace", 16
    )

    # Initialize game state
    state = create_initial_game(width=grid_width, height=grid_height)
    game_started = True if training else False
    episode = 1
    score = len(state.snake.body)
    last_dist = get_min_green_dist(state)

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
        1000,
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
                if (game_started or ai_mode) and not state.is_game_over:
                    if training:
                        # Extract relative features
                        curr_features = StateFeatures.from_game_state(state)
                        # Choose action (training exploration)
                        action = agent.get_action(curr_features, training=True)
                        # Translate relative action to absolute direction
                        dx, dy = state.snake.direction.value
                        if action == 0:  # STRAIGHT
                            new_dir = state.snake.direction
                        elif action == 1:  # LEFT
                            new_dir = next(d for d in Direction if d.value == (dy, -dx))
                        else:  # RIGHT
                            new_dir = next(d for d in Direction if d.value == (-dy, dx))
                        state.change_direction(new_dir)

                        # Move one step
                        state.step()

                        # Observe new state and reward
                        next_features = StateFeatures.from_game_state(state)
                        done = state.is_game_over

                        # Compute reward
                        reward = 0.0
                        if done:
                            reward = -100.0  # Heavy crash penalty
                        else:
                            new_len = len(state.snake.body)
                            if new_len > score:
                                reward = 100.0  # High reward for growing
                                score = new_len
                            elif new_len < score:
                                reward = -30.0  # Penalty for shrinking (eating a red apple)
                                score = new_len
                            else:
                                # Survival reward
                                reward = 1.0

                                # Dense distance-based reward
                                new_dist = get_min_green_dist(state)
                                if new_dist < last_dist:
                                    reward += 10.0  # Reward for moving closer to green apple
                                elif new_dist > last_dist:
                                    reward -= 15.0  # Penalty for moving away from green apple
                                last_dist = new_dist

                        # Update Q-table
                        agent.update(curr_features, action, reward, next_features, done)

                        if done:
                            agent.decay_epsilon()
                            if qtable_path:
                                agent.q_table.save_to_file(qtable_path)

                            episode += 1
                            if episode > episodes:
                                print(f"🎉 Training of {episodes} episodes completed successfully!")
                                running = False
                            else:
                                # Reset for next training episode automatically
                                state = create_initial_game(width=grid_width, height=grid_height)
                                score = len(state.snake.body)
                                last_dist = get_min_green_dist(state)
                    else:
                        if ai_mode:
                            # Extract relative features
                            curr_features = StateFeatures.from_game_state(state)
                            # Choose action
                            action = agent.get_action(curr_features, training=False)
                            # Translate relative action to absolute direction
                            dx, dy = state.snake.direction.value
                            if action == 0:  # STRAIGHT
                                new_dir = state.snake.direction
                            elif action == 1:  # LEFT
                                new_dir = next(d for d in Direction if d.value == (dy, -dx))
                            else:  # RIGHT
                                new_dir = next(d for d in Direction if d.value == (-dy, dx))
                            state.change_direction(new_dir)

                        state.step()

                    # Output State Vision to terminal on every tick
                    print_vision_grid(state)
                    print(f"Snake Length: {len(state.snake.body)}")
                    if state.is_game_over and not training:
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
                        game_started = ai_mode

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

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Check Apply Button and Autopilot toggle clicks
                    btn_rect = pygame.Rect(
                        board_w + 20, HEADER_HEIGHT + 290, SIDEBAR_WIDTH - 40, 45
                    )
                    # Check Autopilot toggle click
                    toggle_rect = pygame.Rect(
                        board_w + 20, HEADER_HEIGHT + 360, SIDEBAR_WIDTH - 40, 45
                    )
                    if not training and toggle_rect.collidepoint(mouse_pos):
                        ai_mode = not ai_mode
                        if ai_mode:
                            game_started = True

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

                        # Hot reload grid size change:
                        state.config.width = grid_width
                        state.config.height = grid_height

                        # Remove out-of-bounds apples and spawn new ones
                        state.green_apples = {p for p in state.green_apples if state.is_within_bounds(p)}
                        state.red_apples = {p for p in state.red_apples if state.is_within_bounds(p)}
                        while len(state.green_apples) < 2:
                            state._spawn_green_apple()
                        while len(state.red_apples) < 1:
                            state._spawn_red_apple()

                        last_dist = get_min_green_dist(state)

                        pygame.time.set_timer(MOVE_EVENT, int(1000 / speed))

                        print("\n" + "=" * 60)
                        print(
                            f"🎮 SETTINGS APPLIED & RESTARTED - {grid_width}x{grid_height} at speed {speed} moves/sec 🎮"
                        )
                        print_vision_grid(state)
                        print("=" * 60)

        # Clear Screen
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
        # Draw Green Apples as full flat squares
        for apple in state.green_apples:
            rect = pygame.Rect(
                apple.x * CELL_SIZE + 2,
                HEADER_HEIGHT + apple.y * CELL_SIZE + 2,
                CELL_SIZE - 4,
                CELL_SIZE - 4,
            )
            pygame.draw.rect(screen, COLOR_GREEN_APPLE, rect)

        # Draw Red Apples as full flat squares
        for apple in state.red_apples:
            rect = pygame.Rect(
                apple.x * CELL_SIZE + 2,
                HEADER_HEIGHT + apple.y * CELL_SIZE + 2,
                CELL_SIZE - 4,
                CELL_SIZE - 4,
            )
            pygame.draw.rect(screen, COLOR_RED_APPLE, rect)

        # ----------------- DRAW SNAKE -----------------
        for i, point in enumerate(state.snake.body):
            segment_color = COLOR_SNAKE_HEAD if i == 0 else COLOR_SNAKE_BODY
            # Draw segment as a flat full square with small separation
            rect = pygame.Rect(
                point.x * CELL_SIZE + 1,
                HEADER_HEIGHT + point.y * CELL_SIZE + 1,
                CELL_SIZE - 2,
                CELL_SIZE - 2,
            )
            pygame.draw.rect(screen, segment_color, rect)

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

        # Draw Apply Button (simple rustic flat square button)
        mouse_pos = pygame.mouse.get_pos()
        btn_rect = pygame.Rect(
            board_w + 20, HEADER_HEIGHT + 290, SIDEBAR_WIDTH - 40, 45
        )
        is_hovered = btn_rect.collidepoint(mouse_pos)
        btn_color = (90, 90, 90) if is_hovered else (60, 60, 60)

        pygame.draw.rect(screen, btn_color, btn_rect)
        pygame.draw.rect(screen, COLOR_DIVIDER, btn_rect, 1)  # Outline border

        btn_text = font_subtitle.render("APPLY & RESTART", True, COLOR_TEXT_PRIMARY)
        btn_text_rect = btn_text.get_rect(center=btn_rect.center)
        screen.blit(btn_text, btn_text_rect)

        # Draw AUTOPILOT Toggle (no rounded corners, simple)
        if not training:
            toggle_rect = pygame.Rect(
                board_w + 20, HEADER_HEIGHT + 360, SIDEBAR_WIDTH - 40, 45
            )
            pygame.draw.rect(screen, COLOR_BG_DARK, toggle_rect)
            pygame.draw.rect(screen, COLOR_DIVIDER, toggle_rect, 1)  # Outline border

            toggle_lbl = font_subtitle.render("AUTOPILOT", True, COLOR_TEXT_PRIMARY)
            screen.blit(
                toggle_lbl,
                (toggle_rect.x + 15, toggle_rect.y + (45 - toggle_lbl.get_height()) // 2),
            )

            switch_w = 50
            switch_h = 24
            switch_x = toggle_rect.x + toggle_rect.width - switch_w - 15
            switch_y = toggle_rect.y + (45 - switch_h) // 2

            switch_bg_color = COLOR_SNAKE_HEAD if ai_mode else COLOR_DIVIDER
            switch_rect = pygame.Rect(switch_x, switch_y, switch_w, switch_h)
            pygame.draw.rect(screen, switch_bg_color, switch_rect)
            pygame.draw.rect(screen, COLOR_TEXT_MUTED, switch_rect, 1)

            knob_size = 14
            knob_y = switch_y + (switch_h - knob_size) // 2
            knob_x = switch_x + (switch_w - knob_size - 4 if ai_mode else 4)
            knob_rect = pygame.Rect(knob_x, knob_y, knob_size, knob_size)
            pygame.draw.rect(screen, (255, 255, 255), knob_rect)

        if ai_mode:
            # Telemetry header (no AI branding)
            tel_y = HEADER_HEIGHT + 360 if training else HEADER_HEIGHT + 420
            lbl_tel = font_title_sm.render("TELEMETRY", True, COLOR_TEXT_PRIMARY)
            screen.blit(lbl_tel, (board_w + 20, tel_y))

            if training:
                lbl_ep = font_subtitle.render(
                    f"Episode: {episode}/{episodes}", True, COLOR_TEXT_MUTED
                )
                screen.blit(lbl_ep, (board_w + 20, tel_y + 30))

                lbl_eps = font_subtitle.render(
                    f"Epsilon: {agent.epsilon:.4f}", True, COLOR_TEXT_MUTED
                )
                screen.blit(lbl_eps, (board_w + 20, tel_y + 55))

                lbl_states = font_subtitle.render(
                    f"Unique States: {len(agent.q_table.table)}", True, COLOR_GREEN_APPLE
                )
                screen.blit(lbl_states, (board_w + 20, tel_y + 80))
            else:
                # Extract features for display
                feats = StateFeatures.from_game_state(state)

                # Show danger features
                danger_str = []
                if feats.danger_straight:
                    danger_str.append("Straight")
                if feats.danger_left:
                    danger_str.append("Left")
                if feats.danger_right:
                    danger_str.append("Right")
                danger_text = ", ".join(danger_str) if danger_str else "None"

                lbl_danger = font_subtitle.render(
                    f"Danger: {danger_text}", True, COLOR_TEXT_MUTED
                )
                screen.blit(lbl_danger, (board_w + 20, tel_y + 30))

                # Show relative green apple direction
                apple_str = "None"
                if feats.green_apple_front:
                    apple_str = "Front"
                    if feats.green_apple_left:
                        apple_str += "-Left"
                    elif feats.green_apple_right:
                        apple_str += "-Right"
                elif feats.green_apple_back:
                    apple_str = "Back"
                    if feats.green_apple_left:
                        apple_str += "-Left"
                    elif feats.green_apple_right:
                        apple_str += "-Right"
                elif feats.green_apple_left:
                    apple_str = "Left"
                elif feats.green_apple_right:
                    apple_str = "Right"

                lbl_apple = font_subtitle.render(
                    f"Green Apple: {apple_str}", True, COLOR_TEXT_MUTED
                )
                screen.blit(lbl_apple, (board_w + 20, tel_y + 55))

                # Show chosen action and Q-value
                action_id, q_val = agent.q_table.get_best_action_value(feats)
                action_names = {0: "STRAIGHT", 1: "TURN LEFT", 2: "TURN RIGHT"}
                act_name = action_names.get(action_id, "STRAIGHT")

                lbl_act = font_subtitle.render(
                    f"Next Action: {act_name}", True, COLOR_GREEN_APPLE
                )
                screen.blit(lbl_act, (board_w + 20, tel_y + 80))

                lbl_q = font_subtitle.render(
                    f"Action Q-Val: {q_val:.2f}", True, COLOR_GREEN_APPLE
                )
                screen.blit(lbl_q, (board_w + 20, tel_y + 105))

        # ----------------- DRAW WAITING TO START OVERLAY -----------------
        if not game_started and not ai_mode and not state.is_game_over:
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
                "PRESS ANY DIRECTION KEY TO START", True, COLOR_TEXT_PRIMARY
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

    if training and qtable_path:
        agent.q_table.save_to_file(qtable_path)
        print(f"🎉 Progress saved to '{qtable_path}' before exit.")

    pygame.quit()
    sys.exit()

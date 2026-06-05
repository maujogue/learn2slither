import math

import pygame


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
        is_exponential: bool = False,
        ticks: list[int] | None = None,
    ):
        self.x = x
        self.y = y
        self.width = width
        self.min_val = min_val
        self.max_val = max_val
        self.current_val = current_val
        self.label = label
        self.is_exponential = is_exponential
        self.ticks = ticks
        self.is_dragging = False

    def draw(
        self,
        screen,
        font_label,
        font_val,
        text_color,
        track_color,
        handle_color,
    ):
        # Draw label
        lbl_surf = font_label.render(self.label, True, text_color)
        screen.blit(lbl_surf, (self.x, self.y - 22))

        # Draw current value
        val_surf = font_val.render(str(self.current_val), True, handle_color)
        screen.blit(
            val_surf, (self.x + self.width - val_surf.get_width(), self.y - 22)
        )

        # Draw track
        pygame.draw.line(
            screen,
            track_color,
            (self.x, self.y),
            (self.x + self.width, self.y),
            4,
        )

        # Draw ticks if present
        if self.ticks:
            for tick in self.ticks:
                if self.is_exponential:
                    if tick <= 0 or self.min_val <= 0:
                        continue
                    fraction = math.log(tick / self.min_val) / math.log(
                        self.max_val / self.min_val
                    )
                else:
                    fraction = (tick - self.min_val) / (
                        self.max_val - self.min_val
                    )

                tick_x = self.x + int(fraction * self.width)
                # Draw a small vertical tick line
                pygame.draw.line(
                    screen,
                    track_color,
                    (tick_x, self.y - 5),
                    (tick_x, self.y + 5),
                    2,
                )

                # Draw a tiny tick value text below the track
                tick_val_surf = font_val.render(str(tick), True, text_color)
                screen.blit(
                    tick_val_surf,
                    (tick_x - tick_val_surf.get_width() // 2, self.y + 8),
                )

        # Draw handle as a simple full square block (no circles)
        if self.is_exponential:
            val = max(self.min_val, min(self.max_val, self.current_val))
            fraction = math.log(val / self.min_val) / math.log(
                self.max_val / self.min_val
            )
        else:
            fraction = (self.current_val - self.min_val) / (
                self.max_val - self.min_val
            )

        handle_x = self.x + int(fraction * self.width)
        handle_rect = pygame.Rect(handle_x - 6, self.y - 8, 12, 16)
        pygame.draw.rect(screen, handle_color, handle_rect)

    def handle_event(self, event, mouse_pos) -> bool:
        """Returns True if the value changed."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # Check if click is on the track/handle area
                track_rect = pygame.Rect(
                    self.x - 10, self.y - 12, self.width + 20, 24
                )
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
        if self.is_exponential:
            raw_val = self.min_val * (
                (self.max_val / self.min_val) ** fraction
            )
        else:
            raw_val = self.min_val + fraction * (self.max_val - self.min_val)
        self.current_val = int(round(raw_val))

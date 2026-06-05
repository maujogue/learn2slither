import os


def _validation_plot_path(model_path: str) -> str:
    root, _ = os.path.splitext(model_path)
    return f"{root}_validation.svg"


def _loss_plot_path(model_path: str) -> str:
    root, _ = os.path.splitext(model_path)
    return f"{root}_loss.svg"


def _write_loss_plot(points: list[tuple[int, float]], filepath: str) -> None:
    if not points:
        return

    width = 900
    height = 500
    margin_left = 70
    margin_right = 30
    margin_top = 30
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_episode = max(episode for episode, _ in points)
    max_loss = max(loss for _, loss in points)
    y_loss_max = max(1e-9, max_loss)

    def x_pos(episode: int) -> float:
        if max_episode <= 1:
            return margin_left + plot_width
        return margin_left + (episode / max_episode) * plot_width

    def y_pos(loss: float) -> float:
        return margin_top + plot_height - (loss / y_loss_max) * plot_height

    loss_polyline = " ".join(
        f"{x_pos(episode):.2f},{y_pos(loss):.2f}" for episode, loss in points
    )
    circles = "\n".join(
        f'<circle cx="{x_pos(episode):.2f}" cy="{y_pos(loss):.2f}"'
        f' r="3" fill="#dc2626">'
        f"<title>Episode {episode}: mean Huber loss"
        f" {loss:.6f}</title></circle>"
        for episode, loss in points
    )
    loss_mid = y_pos(y_loss_max / 2)
    y_mid = margin_top + plot_height / 2
    r90 = f"rotate(-90 18 {y_mid:.0f})"
    f13 = 'font-size="13" font-family="sans-serif"'
    f12 = 'font-size="12" font-family="sans-serif"'
    svg = "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="20" text-anchor="middle"'
        f' font-size="16" font-family="sans-serif"'
        f' fill="#111827">DQN training loss</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}"'
        f' x2="{margin_left + plot_width}"'
        f' y2="{margin_top + plot_height}" stroke="#374151"/>',
        f'<line x1="{margin_left}" y1="{margin_top}"'
        f' x2="{margin_left}" y2="{margin_top + plot_height}"'
        f' stroke="#374151"/>',
        f'<text x="{margin_left + plot_width / 2:.0f}"'
        f' y="{height - 18}" text-anchor="middle" {f13}'
        f' fill="#374151">Training episodes</text>',
        f'<text x="18" y="{y_mid:.0f}" transform="{r90}"'
        f' text-anchor="middle" {f13}'
        f' fill="#dc2626">Mean Huber loss per episode</text>',
        f'<text x="{margin_left - 8}"'
        f' y="{margin_top + plot_height + 4}"'
        f' text-anchor="end" {f12} fill="#6b7280">0</text>',
        f'<text x="{margin_left - 8}" y="{loss_mid + 4:.2f}"'
        f' text-anchor="end" {f12}'
        f' fill="#6b7280">{y_loss_max / 2:.6f}</text>',
        f'<text x="{margin_left - 8}" y="{margin_top + 4}"'
        f' text-anchor="end" {f12}'
        f' fill="#6b7280">{y_loss_max:.6f}</text>',
        f'<text x="{margin_left}" y="{height - 40}"'
        f' text-anchor="middle" {f12} fill="#6b7280">0</text>',
        f'<text x="{margin_left + plot_width}" y="{height - 40}"'
        f' text-anchor="middle" {f12}'
        f' fill="#6b7280">{max_episode}</text>',
        f'<polyline points="{loss_polyline}" fill="none"'
        f' stroke="#dc2626" stroke-width="3"/>',
        circles,
        "</svg>",
    ])
    with open(filepath, "w") as f:
        f.write(svg)


def _write_validation_plot(
    points: list[tuple[int, float, float]], filepath: str
) -> None:
    if not points:
        return

    width = 900
    height = 500
    margin_left = 70
    margin_right = 70
    margin_top = 30
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_episode = max(episode for episode, _, _ in points)
    max_score = max(score for _, score, _ in points)
    max_seconds = max(seconds for _, _, seconds in points)
    y_score_max = max(1.0, max_score)
    y_seconds_max = max(1.0, max_seconds)

    def x_pos(episode: int) -> float:
        if max_episode <= 1:
            return margin_left + plot_width
        return margin_left + (episode / max_episode) * plot_width

    def score_y_pos(score: float) -> float:
        return margin_top + plot_height - (score / y_score_max) * plot_height

    def time_y_pos(seconds: float) -> float:
        return (
            margin_top + plot_height - (seconds / y_seconds_max) * plot_height
        )

    score_polyline = " ".join(
        f"{x_pos(episode):.2f},{score_y_pos(score):.2f}"
        for episode, score, _ in points
    )
    time_polyline = " ".join(
        f"{x_pos(episode):.2f},{time_y_pos(seconds):.2f}"
        for episode, _, seconds in points
    )
    circles = "\n".join(
        f'<circle cx="{x_pos(episode):.2f}"'
        f' cy="{score_y_pos(score):.2f}" r="4" fill="#2563eb">'
        f"<title>Iteration {episode}: mean score {score:.2f},"
        f" training time {seconds:.2f}s</title></circle>"
        f'<circle cx="{x_pos(episode):.2f}"'
        f' cy="{time_y_pos(seconds):.2f}" r="4" fill="#f97316">'
        f"<title>Iteration {episode}: training time {seconds:.2f}s,"
        f" mean score {score:.2f}</title></circle>"
        for episode, score, seconds in points
    )
    labels = "\n".join(
        f'<text x="{x_pos(episode):.2f}"'
        f' y="{score_y_pos(score) - 10:.2f}"'
        f' text-anchor="middle" font-size="12"'
        f' fill="#111827">{score:.2f}</text>'
        for episode, score, _ in points
    )
    score_mid = score_y_pos(y_score_max / 2)
    time_mid = time_y_pos(y_seconds_max / 2)
    y_mid = margin_top + plot_height / 2
    r90l = f"rotate(-90 18 {y_mid:.0f})"
    r90r = f"rotate(90 {width - 18} {y_mid:.0f})"
    f13 = 'font-size="13" font-family="sans-serif"'
    f12 = 'font-size="12" font-family="sans-serif"'
    svg = "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="20" text-anchor="middle"'
        f' font-size="16" font-family="sans-serif" fill="#111827">'
        f'Validation mean score and training time</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}"'
        f' x2="{margin_left + plot_width}"'
        f' y2="{margin_top + plot_height}" stroke="#374151"/>',
        f'<line x1="{margin_left}" y1="{margin_top}"'
        f' x2="{margin_left}" y2="{margin_top + plot_height}"'
        f' stroke="#374151"/>',
        f'<line x1="{margin_left + plot_width}" y1="{margin_top}"'
        f' x2="{margin_left + plot_width}"'
        f' y2="{margin_top + plot_height}" stroke="#374151"/>',
        f'<text x="{margin_left + plot_width / 2:.0f}"'
        f' y="{height - 18}" text-anchor="middle" {f13}'
        f' fill="#374151">Training iterations</text>',
        f'<text x="18" y="{y_mid:.0f}" transform="{r90l}"'
        f' text-anchor="middle" {f13} fill="#2563eb">'
        f'Mean score over 100 validation runs</text>',
        f'<text x="{width - 18}" y="{y_mid:.0f}"'
        f' transform="{r90r}" text-anchor="middle" {f13}'
        f' fill="#f97316">Training time (seconds)</text>',
        f'<text x="{margin_left - 8}"'
        f' y="{margin_top + plot_height + 4}"'
        f' text-anchor="end" {f12} fill="#6b7280">0</text>',
        f'<text x="{margin_left - 8}" y="{score_mid + 4:.2f}"'
        f' text-anchor="end" {f12}'
        f' fill="#6b7280">{y_score_max / 2:.1f}</text>',
        f'<text x="{margin_left - 8}" y="{margin_top + 4}"'
        f' text-anchor="end" {f12}'
        f' fill="#6b7280">{y_score_max:.1f}</text>',
        f'<text x="{margin_left + plot_width + 8}"'
        f' y="{margin_top + plot_height + 4}"'
        f' text-anchor="start" {f12} fill="#6b7280">0</text>',
        f'<text x="{margin_left + plot_width + 8}"'
        f' y="{time_mid + 4:.2f}" text-anchor="start" {f12}'
        f' fill="#6b7280">{y_seconds_max / 2:.1f}</text>',
        f'<text x="{margin_left + plot_width + 8}"'
        f' y="{margin_top + 4}" text-anchor="start" {f12}'
        f' fill="#6b7280">{y_seconds_max:.1f}</text>',
        f'<text x="{margin_left}" y="{height - 40}"'
        f' text-anchor="middle" {f12} fill="#6b7280">0</text>',
        f'<text x="{margin_left + plot_width}" y="{height - 40}"'
        f' text-anchor="middle" {f12}'
        f' fill="#6b7280">{max_episode}</text>',
        f'<polyline points="{score_polyline}" fill="none"'
        f' stroke="#2563eb" stroke-width="3"/>',
        f'<polyline points="{time_polyline}" fill="none"'
        f' stroke="#f97316" stroke-width="3"/>',
        circles,
        labels,
        "</svg>",
    ])
    with open(filepath, "w") as f:
        f.write(svg)

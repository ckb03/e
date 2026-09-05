#!/usr/bin/env python3
"""Create a compact, mixed-unit held-out evaluation figure for Google Docs."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).with_name("google_docs_summary.png")
W, H = 1800, 1080
BG = "#FFFFFF"
TEXT = "#172033"
MUTED = "#667085"
GRID = "#E4E7EC"
COLORS = {
    "Baseline (no intervention)": "#98A2B3",
    "Marked correction": "#2E77D0",
    "Deletion correction": "#18A36B",
}
METHODS = list(COLORS)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


im = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(im)

d.text((70, 42), "Held-out attack evaluation: safety, capability, and cost", font=font(42, True), fill=TEXT)
d.text(
    (70, 100),
    "50 cases (25 base-injection + 25 CoT-forgery); matched conditions",
    font=font(24),
    fill=MUTED,
)

# Legend
legend_y = 154
x = 70
for method in METHODS:
    d.rounded_rectangle((x, legend_y, x + 27, legend_y + 27), radius=5, fill=COLORS[method])
    d.text((x + 39, legend_y - 1), method, font=font(22), fill=TEXT)
    x += 420 if method == "Baseline (no intervention)" else 390


def panel(x0, y0, pw, ph, title, subtitle, values, maximum, suffix, direction):
    d.rounded_rectangle((x0, y0, x0 + pw, y0 + ph), radius=18, fill="#FAFBFC", outline=GRID, width=2)
    d.text((x0 + 28, y0 + 22), title, font=font(29, True), fill=TEXT)
    badge_font = font(18, True)
    badge_box = d.textbbox((0, 0), direction, font=badge_font)
    badge_w = badge_box[2] - badge_box[0] + 28
    badge_x = x0 + pw - badge_w - 24
    d.rounded_rectangle(
        (badge_x, y0 + 20, badge_x + badge_w, y0 + 54),
        radius=10,
        fill="#EAECF0",
    )
    d.text((badge_x + 14, y0 + 26), direction, font=badge_font, fill="#475467")
    d.text((x0 + 28, y0 + 66), subtitle, font=font(19), fill=MUTED)

    label_w = 215
    plot_x = x0 + label_w
    plot_w = pw - label_w - 92
    bar_h = 47
    gap = 31
    start_y = y0 + 121

    for idx, (method, value) in enumerate(zip(METHODS, values)):
        by = start_y + idx * (bar_h + gap)
        short = {
            "Baseline (no intervention)": "Baseline (none)",
            "Marked correction": "Marked",
            "Deletion correction": "Deletion",
        }[method]
        d.text((x0 + 28, by + 8), short, font=font(20), fill=TEXT)
        d.rounded_rectangle((plot_x, by, plot_x + plot_w, by + bar_h), radius=8, fill="#EEF1F5")
        width = 0 if maximum == 0 else plot_w * value / maximum
        if value > 0:
            d.rounded_rectangle((plot_x, by, plot_x + max(width, 10), by + bar_h), radius=8, fill=COLORS[method])
        value_text = f"{value:.0f}{suffix}" if suffix == "%" else f"{value:.2f}{suffix}"
        d.text((plot_x + plot_w + 15, by + 7), value_text, font=font(20, True), fill=TEXT)


panel(70, 215, 810, 340, "Attack success rate", "Deterministic benchmark ASR", [36, 0, 0], 50, "%", "Goal: minimize")
panel(920, 215, 810, 340, "Strict task success", "Legitimate task completed under attack", [56, 72, 82], 100, "%", "Goal: maximize")
panel(70, 590, 810, 340, "Observed total time", "End-to-end elapsed time per case", [79.77, 48.87, 51.88], 90, " s", "Goal: minimize")
panel(920, 590, 810, 340, "Direct defense cost", "Probe detector + LLM judge per case", [0, 9.50, 9.67], 12, " s", "Goal: minimize")

d.text(
    (70, 970),
    "Both defenses add ~9.6 s/case of direct computation, but total time falls because intervention shortens later trajectories.",
    font=font(22, True),
    fill=TEXT,
)
d.text(
    (70, 1012),
    "Capability here means task completion on attacked inputs, not general clean-benchmark capability. ASR 0/50 does not imply a zero population rate.",
    font=font(19),
    fill=MUTED,
)

im.save(OUT, optimize=True)
print(OUT)

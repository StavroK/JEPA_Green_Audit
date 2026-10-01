"""Interactive RGB-only labeling UI for the Fundidora M3 benchmark.

Controls:
- V: vegetation
- N: non_vegetation
- U: uncertain
- B / Left Arrow: go back
- Right Arrow: skip forward without labeling
- Q / Escape: quit

Labels are autosaved to the CSV after every change. The tool resumes from the
first unlabeled row. Review RGB only; do not consult NDVI or SCL while labeling.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageTk

DEFAULT_CSV = Path("data/labels/fundidora_m3/fundidora_patch_labels_8x8.csv")
RGB_DIR = Path("data/interim/fundidora")
ALLOWED_LABELS = {"vegetation", "non_vegetation", "uncertain"}
GRID = 8
TARGET_DISPLAY_SIZE = 280
CONTEXT_DISPLAY_SIZE = 360
OVERVIEW_DISPLAY_SIZE = 420


def load_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    required = {"year", "row", "col", "patch_path", "label", "reviewer", "notes"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    for row in rows:
        label = row.get("label", "").strip()
        if label and label not in ALLOWED_LABELS:
            raise ValueError(f"Invalid label {label!r} in CSV")

    return rows, fieldnames


def save_rows(csv_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(csv_path)


def first_unlabeled_index(rows: list[dict[str, str]]) -> int:
    for index, row in enumerate(rows):
        if not row.get("label", "").strip():
            return index
    return max(0, len(rows) - 1)


def label_counts(rows: list[dict[str, str]]) -> Counter:
    return Counter(row.get("label", "").strip() or "unlabeled" for row in rows)


class LabelingApp:
    def __init__(self, root: tk.Tk, csv_path: Path, reviewer: str = "") -> None:
        self.root = root
        self.csv_path = csv_path
        self.rows, self.fieldnames = load_rows(csv_path)
        if not self.rows:
            raise ValueError("CSV contains no rows")

        self.reviewer = reviewer.strip()
        self.index = first_unlabeled_index(self.rows)
        self.photo = None

        root.title("JEPA Green Audit — M3 RGB Labeling")
        root.geometry("1280x820")
        root.minsize(1100, 760)

        self.title_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.counts_var = tk.StringVar()
        self.current_label_var = tk.StringVar()

        tk.Label(root, textvariable=self.title_var, font=("Segoe UI", 16, "bold")).pack(pady=(12, 4))
        tk.Label(
            root,
            text="Label the dominant visible cover in the highlighted 8x8 review cell using RGB only. Do not consult NDVI or SCL.",
            font=("Segoe UI", 10),
        ).pack()

        image_frame = tk.Frame(root)
        image_frame.pack(pady=10)

        left = tk.Frame(image_frame)
        left.grid(row=0, column=0, padx=8)
        tk.Label(left, text="Target cell", font=("Segoe UI", 10, "bold")).pack()
        self.target_label = tk.Label(left, bd=2, relief="sunken")
        self.target_label.pack(pady=4)

        middle = tk.Frame(image_frame)
        middle.grid(row=0, column=1, padx=8)
        tk.Label(middle, text="RGB context", font=("Segoe UI", 10, "bold")).pack()
        self.context_label = tk.Label(middle, bd=2, relief="sunken")
        self.context_label.pack(pady=4)

        right = tk.Frame(image_frame)
        right.grid(row=0, column=2, padx=8)
        tk.Label(right, text="AOI location (red box = current review cell)", font=("Segoe UI", 10, "bold")).pack()
        self.overview_label = tk.Label(right, bd=2, relief="sunken")
        self.overview_label.pack(pady=4)

        tk.Label(root, textvariable=self.current_label_var, font=("Segoe UI", 11, "bold")).pack()

        criteria = tk.LabelFrame(root, text="Labeling criteria", padx=12, pady=8)
        criteria.pack(fill="x", padx=36, pady=(6, 8))
        criteria_text = (
            "Vegetation (V): vegetation is the dominant visible cover in the target cell "
            "(roughly more than half).\n"
            "Non-vegetation (N): roads, roofs, buildings, bare ground, water, or other "
            "non-vegetated surfaces dominate the target cell.\n"
            "Uncertain (U): the target cell is genuinely mixed, blurred, obscured, or you "
            "cannot make a confident dominant-cover judgment.\n"
            "Use only the RGB target/context/AOI views. Do not consult NDVI or SCL. "
            "Classify visible cover only; do not infer tree health."
        )
        tk.Label(
            criteria,
            text=criteria_text,
            justify="left",
            anchor="w",
            wraplength=1120,
            font=("Segoe UI", 9),
        ).pack(fill="x")

        tk.Label(root, textvariable=self.status_var, font=("Segoe UI", 10)).pack(pady=(4, 0))
        tk.Label(root, textvariable=self.counts_var, font=("Segoe UI", 9)).pack(pady=(2, 10))

        buttons = tk.Frame(root)
        buttons.pack(pady=4)
        self._button(buttons, "V  Vegetation", lambda: self.set_label("vegetation"), 0)
        self._button(buttons, "N  Non-vegetation", lambda: self.set_label("non_vegetation"), 1)
        self._button(buttons, "U  Uncertain", lambda: self.set_label("uncertain"), 2)

        nav = tk.Frame(root)
        nav.pack(pady=8)
        tk.Button(nav, text="← Back (B)", command=self.back, width=16).grid(row=0, column=0, padx=5)
        tk.Button(nav, text="Skip →", command=self.forward, width=16).grid(row=0, column=1, padx=5)
        tk.Button(nav, text="Save & Quit (Q)", command=self.quit, width=16).grid(row=0, column=2, padx=5)

        help_text = (
            "Keyboard: V vegetation · N non-vegetation · U uncertain · "
            "B/← back · → skip · Q/Esc quit"
        )
        tk.Label(root, text=help_text, font=("Segoe UI", 9)).pack(pady=(8, 0))

        root.bind("<Key-v>", lambda event: self.set_label("vegetation"))
        root.bind("<Key-V>", lambda event: self.set_label("vegetation"))
        root.bind("<Key-n>", lambda event: self.set_label("non_vegetation"))
        root.bind("<Key-N>", lambda event: self.set_label("non_vegetation"))
        root.bind("<Key-u>", lambda event: self.set_label("uncertain"))
        root.bind("<Key-U>", lambda event: self.set_label("uncertain"))
        root.bind("<Key-b>", lambda event: self.back())
        root.bind("<Key-B>", lambda event: self.back())
        root.bind("<Left>", lambda event: self.back())
        root.bind("<Right>", lambda event: self.forward())
        root.bind("<Key-q>", lambda event: self.quit())
        root.bind("<Key-Q>", lambda event: self.quit())
        root.bind("<Escape>", lambda event: self.quit())
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self.refresh()

    def _button(self, parent, text, command, column):
        tk.Button(parent, text=text, command=command, width=20, height=2).grid(
            row=0, column=column, padx=5
        )

    def refresh(self) -> None:
        row = self.rows[self.index]
        patch_path = Path(row["patch_path"])
        if not patch_path.exists():
            messagebox.showerror("Missing patch", f"Patch not found:\n{patch_path}")
            return

        year = row["year"]
        grid_row = int(row["row"])
        grid_col = int(row["col"])
        source_path = RGB_DIR / f"fundidora_{year}_rgb.png"
        if not source_path.exists():
            messagebox.showerror("Missing RGB source", f"Source not found:\n{source_path}")
            return

        with Image.open(source_path) as src:
            src = src.convert("RGB")
            width, height = src.size
            x_edges = np.linspace(0, width, GRID + 1, dtype=int)
            y_edges = np.linspace(0, height, GRID + 1, dtype=int)

            x0, x1 = int(x_edges[grid_col]), int(x_edges[grid_col + 1])
            y0, y1 = int(y_edges[grid_row]), int(y_edges[grid_row + 1])

            # Target cell: smoothed enlargement for visual interpretation.
            target = src.crop((x0, y0, x1, y1)).resize(
                (TARGET_DISPLAY_SIZE, TARGET_DISPLAY_SIZE),
                Image.Resampling.BICUBIC,
            )

            # Context: roughly a 5x5-cell neighborhood around the target.
            radius = 2
            c0 = max(0, grid_col - radius)
            c1 = min(GRID, grid_col + radius + 1)
            r0 = max(0, grid_row - radius)
            r1 = min(GRID, grid_row + radius + 1)
            context = src.crop(
                (
                    int(x_edges[c0]),
                    int(y_edges[r0]),
                    int(x_edges[c1]),
                    int(y_edges[r1]),
                )
            ).resize(
                (CONTEXT_DISPLAY_SIZE, CONTEXT_DISPLAY_SIZE),
                Image.Resampling.BICUBIC,
            )

            # AOI overview with an obvious red marker for the current review cell.
            overview = src.convert("RGBA")
            overlay = Image.new("RGBA", overview.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            line_width = max(2, round(max(width, height) / 90))
            x1_box = max(x0 + 1, x1 - 1)
            y1_box = max(y0 + 1, y1 - 1)

            # Semi-transparent fill plus a thick red outline.
            draw.rectangle(
                (x0, y0, x1_box, y1_box),
                fill=(255, 0, 0, 70),
                outline=(255, 0, 0, 255),
                width=line_width,
            )

            # Center crosshair makes small cells easier to locate.
            cx = (x0 + x1_box) // 2
            cy = (y0 + y1_box) // 2
            marker = max(3, line_width + 1)
            draw.line(
                (cx - marker, cy, cx + marker, cy),
                fill=(255, 0, 0, 255),
                width=max(2, line_width),
            )
            draw.line(
                (cx, cy - marker, cx, cy + marker),
                fill=(255, 0, 0, 255),
                width=max(2, line_width),
            )

            # Add a compact TARGET label near the highlighted cell.
            label_x = min(max(0, x1_box + 4), max(0, width - 58))
            label_y = max(0, y0 - 14)
            draw.rectangle(
                (label_x - 2, label_y - 2, label_x + 54, label_y + 12),
                fill=(255, 0, 0, 230),
            )
            draw.text((label_x, label_y), "TARGET", fill="white")

            overview = Image.alpha_composite(overview, overlay).convert("RGB")
            overview.thumbnail(
                (OVERVIEW_DISPLAY_SIZE, OVERVIEW_DISPLAY_SIZE),
                Image.Resampling.BICUBIC,
            )

        self.target_photo = ImageTk.PhotoImage(target)
        self.context_photo = ImageTk.PhotoImage(context)
        self.overview_photo = ImageTk.PhotoImage(overview)
        self.target_label.configure(image=self.target_photo)
        self.context_label.configure(image=self.context_photo)
        self.overview_label.configure(image=self.overview_photo)
        self.title_var.set(
            f"{row['year']} · row {int(row['row']):02d} · col {int(row['col']):02d}"
        )
        label = row.get("label", "").strip() or "UNLABELED"
        self.current_label_var.set(f"Current label: {label}")

        counts = label_counts(self.rows)
        labeled = len(self.rows) - counts["unlabeled"]
        self.status_var.set(
            f"Patch {self.index + 1} of {len(self.rows)} · "
            f"{labeled}/{len(self.rows)} labeled ({100*labeled/len(self.rows):.1f}%)"
        )
        self.counts_var.set(
            "vegetation: {v} · non-vegetation: {n} · uncertain: {u} · unlabeled: {x}".format(
                v=counts["vegetation"],
                n=counts["non_vegetation"],
                u=counts["uncertain"],
                x=counts["unlabeled"],
            )
        )

    def set_label(self, label: str) -> None:
        if label not in ALLOWED_LABELS:
            raise ValueError(label)

        row = self.rows[self.index]
        row["label"] = label
        if self.reviewer and not row.get("reviewer", "").strip():
            row["reviewer"] = self.reviewer
        save_rows(self.csv_path, self.rows, self.fieldnames)

        if self.index < len(self.rows) - 1:
            self.index += 1
            self.refresh()
        else:
            self.refresh()
            if all(r.get("label", "").strip() for r in self.rows):
                messagebox.showinfo("Complete", "All patches have been labeled.")

    def back(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.refresh()

    def forward(self) -> None:
        if self.index < len(self.rows) - 1:
            self.index += 1
            self.refresh()

    def quit(self) -> None:
        save_rows(self.csv_path, self.rows, self.fieldnames)
        self.root.destroy()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--reviewer",
        default="",
        help="Optional reviewer name/initials written to newly labeled rows.",
    )
    args = parser.parse_args()

    root = tk.Tk()
    LabelingApp(root, args.csv, reviewer=args.reviewer)
    root.mainloop()


if __name__ == "__main__":
    main()

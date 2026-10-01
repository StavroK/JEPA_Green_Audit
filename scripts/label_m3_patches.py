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

from PIL import Image, ImageTk

DEFAULT_CSV = Path("data/labels/fundidora_m3/fundidora_patch_labels.csv")
ALLOWED_LABELS = {"vegetation", "non_vegetation", "uncertain"}
DISPLAY_SIZE = 448


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
        root.geometry("760x720")
        root.minsize(680, 650)

        self.title_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.counts_var = tk.StringVar()
        self.current_label_var = tk.StringVar()

        tk.Label(root, textvariable=self.title_var, font=("Segoe UI", 16, "bold")).pack(pady=(12, 4))
        tk.Label(
            root,
            text="RGB-only review. Do not consult NDVI or SCL.",
            font=("Segoe UI", 10),
        ).pack()

        self.image_label = tk.Label(root, bd=2, relief="sunken")
        self.image_label.pack(pady=12)

        tk.Label(root, textvariable=self.current_label_var, font=("Segoe UI", 11, "bold")).pack()
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

        with Image.open(patch_path) as image:
            image = image.convert("RGB")
            image.thumbnail((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.NEAREST)
            self.photo = ImageTk.PhotoImage(image)

        self.image_label.configure(image=self.photo)
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

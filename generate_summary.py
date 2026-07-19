"""Generate a single PDF summary for all COLA/STM challenge scripts.

Run from the repository root:
    python generate_summary.py

Outputs:
    outputs/summary_report.pdf
    outputs/figures/*.png

This script intentionally runs each challenge part in a separate Python subprocess.
That keeps report generation robust: if one challenge raises an exception, calls
sys.exit(), or hangs, the parent process records the problem and continues to the
next challenge part.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = OUTPUT_DIR / "summary_report.pdf"
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("SUMMARY_TIMEOUT_SECONDS", "120"))


@dataclass(frozen=True)
class ChallengePart:
    title: str
    module_name: str
    preamble: str


@dataclass
class ChallengeResult:
    title: str
    module_name: str
    preamble: str
    console_output: str
    figures: list[Path]
    status: str


CHALLENGE_PARTS = [
    ChallengePart(
        title="Challenge 1 Part A: RAAN Drift Deployment Trade",
        module_name="challenge_1_part_a",
        preamble=(
            "This case compares two altitude strategies for exploiting differential "
            "nodal precession. The report preserves the original analytical RAAN-rate "
            "and Hohmann-transfer calculations, then summarizes the resulting separation "
            "and delta-v accounting."
        ),
    ),
    ChallengePart(
        title="Challenge 1 Part B: In-Plane Slotting by Differential Mean Motion",
        module_name="challenge_1_part_b",
        preamble=(
            "This case places spacecraft into in-plane slots by temporarily changing "
            "mean motion through phasing orbits. The output lists target slots, final "
            "slot errors, and the round-trip phasing delta-v for each satellite."
        ),
    ),
    ChallengePart(
        title="Challenge 2 Part 1: Low-Thrust Collision Avoidance",
        module_name="challenge_2_part_1",
        preamble=(
            "This case assesses a nominal conjunction and applies a continuous low-thrust "
            "RTN-frame avoidance maneuver. The output reports nominal risk, selected "
            "lead time, post-maneuver miss distance, and post-maneuver collision probability."
        ),
    ),
    ChallengePart(
        title="Challenge 2 Part 2: Differential-Drag Collision Avoidance",
        module_name="challenge_2_part_2",
        preamble=(
            "This case assesses a nominal conjunction and uses a high-drag attitude window "
            "to alter the primary spacecraft trajectory. The output reports the selected "
            "lead time, along-track displacement, equivalent timing shift, and residual risk."
        ),
    ),
    ChallengePart(
        title="Challenge 3 Part 1: Operational State Machine Demonstration",
        module_name="challenge_3_part_1",
        preamble=(
            "This case demonstrates the spacecraft operational state machine for nominal, "
            "tumbling, recovery, recovered, and mission-loss states. The report summarizes "
            "the resulting attitude, drag-area, and operational-state histories."
        ),
    ),
    ChallengePart(
        title="Challenge 3 Part 2: Coupled Attitude, Drag, and Orbit Response",
        module_name="challenge_3_part_2",
        preamble=(
            "This case compares nominal and anomalous spacecraft operations with attitude-driven "
            "drag-area changes. The output shows how operational state affects altitude decay, "
            "orbital evolution, and separation from the nominal trajectory."
        ),
    ),
    ChallengePart(
        title="Challenge 3 Part 3: Recovery Timing Sweep",
        module_name="challenge_3_part_3",
        preamble=(
            "This case sweeps recovery start time to identify the latest safe recovery opportunity. "
            "The output reports successful and failed recovery cases based on the altitude safety "
            "constraint and final separation from the nominal trajectory."
        ),
    ),
]


CHILD_RUNNER = r'''
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

module_name = sys.argv[1]
figure_dir = Path(sys.argv[2])
figure_dir.mkdir(parents=True, exist_ok=True)

# Make scripts non-interactive.
plt.show = lambda *args, **kwargs: None
plt.close("all")

try:
    module = importlib.import_module(module_name)
    module.main()
except BaseException:
    print(f"ERROR while running {module_name}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    exit_code = 1
else:
    exit_code = 0

# Save whatever plots exist, even if the script failed after making figures.
for index, fig_number in enumerate(plt.get_fignums(), start=1):
    fig = plt.figure(fig_number)
    fig_path = figure_dir / f"{module_name}_figure_{index}.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")

plt.close("all")
sys.exit(exit_code)
'''


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def remove_old_figures() -> None:
    for figure_path in FIGURE_DIR.glob("*.png"):
        figure_path.unlink()


def run_challenge_part(part: ChallengePart) -> ChallengeResult:
    """Run one challenge module in an isolated subprocess and collect its output."""
    print(f"Running {part.title} ...", flush=True)

    before = set(FIGURE_DIR.glob(f"{part.module_name}_figure_*.png"))

    try:
        completed = subprocess.run(
            [sys.executable, "-c", CHILD_RUNNER, part.module_name, str(FIGURE_DIR)],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        status = f"TIMEOUT after {DEFAULT_TIMEOUT_SECONDS} seconds"
        console_output = combine_output(stdout, stderr, status)
        figures = sorted(set(FIGURE_DIR.glob(f"{part.module_name}_figure_*.png")) - before)
        print(f"  {status}", flush=True)
        return ChallengeResult(part.title, part.module_name, part.preamble, console_output, figures, status)

    status = "OK" if completed.returncode == 0 else f"FAILED, return code {completed.returncode}"
    console_output = combine_output(completed.stdout, completed.stderr, status)
    figures = sorted(FIGURE_DIR.glob(f"{part.module_name}_figure_*.png"))

    print(f"  {status}; saved {len(figures)} figure(s)", flush=True)
    return ChallengeResult(part.title, part.module_name, part.preamble, console_output, figures, status)


def combine_output(stdout: str | bytes | None, stderr: str | bytes | None, status: str) -> str:
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")

    chunks = [f"Run status: {status}"]
    if stdout and stdout.strip():
        chunks.append("STDOUT:\n" + stdout.strip())
    if stderr and stderr.strip():
        chunks.append("STDERR:\n" + stderr.strip())
    return "\n\n".join(chunks)


def add_text_page(pdf: PdfPages, title: str, body: str, *, font_size: int = 10) -> None:
    """Add one or more text pages to the PDF."""
    wrapped_lines: list[str] = []
    for paragraph in body.splitlines():
        if not paragraph.strip():
            wrapped_lines.append("")
        else:
            wrapped_lines.extend(textwrap.wrap(paragraph, width=96, replace_whitespace=False))

    max_lines_per_page = max(18, int(760 / (font_size * 1.6)))
    if not wrapped_lines:
        wrapped_lines = ["No console output was produced."]

    for page_index, start in enumerate(range(0, len(wrapped_lines), max_lines_per_page), start=1):
        page_lines = wrapped_lines[start : start + max_lines_per_page]
        fig = plt.figure(figsize=(8.5, 11.0))
        fig.patch.set_facecolor("white")
        page_title = title if page_index == 1 else f"{title} continued"
        fig.text(0.08, 0.95, page_title, fontsize=15, fontweight="bold", va="top")
        fig.text(
            0.08,
            0.90,
            "\n".join(page_lines),
            fontsize=font_size,
            family="monospace" if "Output" in title else "sans-serif",
            va="top",
            linespacing=1.25,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    image = plt.imread(image_path)
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.96, title, fontsize=14, fontweight="bold", va="top")
    ax = fig.add_axes((0.05, 0.05, 0.90, 0.84))
    ax.imshow(image)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_report(results: list[ChallengeResult]) -> None:
    with PdfPages(REPORT_PATH) as pdf:
        add_text_page(
            pdf,
            "COLA / Space Traffic Management Challenge Summary",
            (
                "This report was generated automatically from the repository challenge scripts. "
                "Each challenge part was executed sequentially in its own Python subprocess. "
                "Each section contains a short preamble, captured console output, and the plots "
                "created by that script.\n\n"
                f"Challenge parts attempted: {len(results)}\n"
                f"Timeout per challenge part: {DEFAULT_TIMEOUT_SECONDS} seconds\n"
                f"Figure output directory: {FIGURE_DIR}\n"
            ),
            font_size=11,
        )

        for result in results:
            add_text_page(
                pdf,
                result.title,
                (
                    f"{result.preamble}\n\n"
                    f"Module: {result.module_name}\n"
                    f"Run status: {result.status}\n"
                    f"Generated figures: {len(result.figures)}"
                ),
                font_size=11,
            )
            add_text_page(pdf, f"{result.title} — Output", result.console_output, font_size=8)

            if result.figures:
                for index, figure_path in enumerate(result.figures, start=1):
                    add_image_page(pdf, f"{result.title} — Plot {index}", figure_path)
            else:
                add_text_page(
                    pdf,
                    f"{result.title} — Plots",
                    "No plots were generated or saved for this challenge part.",
                    font_size=10,
                )


def main() -> None:
    ensure_output_dirs()
    remove_old_figures()

    results: list[ChallengeResult] = []
    for part in CHALLENGE_PARTS:
        results.append(run_challenge_part(part))
        build_report(results)
        print(f"  Updated PDF: {REPORT_PATH}", flush=True)

    build_report(results)

    print(f"Summary PDF written to: {REPORT_PATH}", flush=True)
    print(f"Saved figures to: {FIGURE_DIR}", flush=True)

    non_ok = [result for result in results if result.status != "OK"]
    if non_ok:
        print("Some challenge parts did not complete normally:", flush=True)
        for result in non_ok:
            print(f"  - {result.title}: {result.status}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter


PIPELINE_STEPS: list[str] = ["ingest", "eda", "blocking", "features", "model", "stat_validation"]


@dataclass
class StepResult:
    """Execution result for one pipeline step."""

    step: str
    script: str
    started_at: datetime
    ended_at: datetime
    duration_sec: float
    passed: bool
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run linkage pipeline scripts in order.")
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip the EDA step.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model and downstream stat_validation steps.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated subset of steps to run (e.g. ingest,blocking,features).",
    )
    return parser.parse_args()


def resolve_steps(args: argparse.Namespace) -> list[str]:
    if args.steps:
        requested = [s.strip() for s in args.steps.split(",") if s.strip()]
        unknown = sorted(set(requested) - set(PIPELINE_STEPS))
        if unknown:
            raise ValueError(f"Unknown steps in --steps: {', '.join(unknown)}")
        steps = [s for s in PIPELINE_STEPS if s in requested]
    else:
        steps = PIPELINE_STEPS.copy()

    if args.skip_eda:
        steps = [s for s in steps if s != "eda"]

    if args.skip_model:
        steps = [s for s in steps if s not in {"model", "stat_validation"}]

    return steps


def run_step(step: str, project_root: Path) -> StepResult:
    """Run one pipeline script as a subprocess."""
    script = str(Path("scripts") / f"{step}.py")
    started_at = datetime.now()
    started_perf = perf_counter()

    print(f"\n[{step}] START {started_at.isoformat(timespec='seconds')} -> {script}")
    try:
        if not (project_root / script).exists():
            raise FileNotFoundError(f"Missing script: {script}")
        subprocess.run(
            [sys.executable, script],
            cwd=project_root,
            check=True,
        )
        passed = True
        error = None
    except subprocess.CalledProcessError as exc:
        passed = False
        error = f"Step '{step}' failed with exit code {exc.returncode}."
        print(f"[{step}] ERROR {error}", file=sys.stderr)
    except Exception as exc:
        passed = False
        error = f"Step '{step}' failed: {exc}"
        print(f"[{step}] ERROR {error}", file=sys.stderr)

    ended_at = datetime.now()
    duration_sec = perf_counter() - started_perf
    print(
        f"[{step}] END   {ended_at.isoformat(timespec='seconds')} "
        f"(duration={duration_sec:.2f}s, status={'PASS' if passed else 'FAIL'})"
    )

    return StepResult(
        step=step,
        script=script,
        started_at=started_at,
        ended_at=ended_at,
        duration_sec=duration_sec,
        passed=passed,
        error=error,
    )


def print_summary(results: list[StepResult]) -> None:
    """Print final pipeline summary table."""
    print("\nPipeline Summary")
    print("-" * 104)
    print(f"{'Step':<18}{'Status':<8}{'Start':<22}{'End':<22}{'Duration(s)':<12}{'Error'}")
    print("-" * 104)
    for r in results:
        print(
            f"{r.step:<18}"
            f"{('PASS' if r.passed else 'FAIL'):<8}"
            f"{r.started_at.isoformat(timespec='seconds'):<22}"
            f"{r.ended_at.isoformat(timespec='seconds'):<22}"
            f"{r.duration_sec:<12.2f}"
            f"{(r.error or '')}"
        )
    print("-" * 104)


def main() -> int:
    args = parse_args()
    try:
        steps = resolve_steps(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not steps:
        print("No steps selected. Nothing to run.")
        return 0

    project_root = Path(__file__).resolve().parent
    results: list[StepResult] = []

    for step in steps:
        result = run_step(step, project_root)
        results.append(result)
        if not result.passed:
            print_summary(results)
            return 1

    print_summary(results)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

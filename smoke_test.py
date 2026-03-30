#! python3.12
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(slots=True)
class StepResult:
    name: str
    command: list[str]
    returncode: int
    log_path: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the corridor framework smoke-test sequence.")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol.")
    parser.add_argument("--bars-csv", default="my_bars.csv", help="Intraday bars CSV used for local smoke runs.")
    parser.add_argument("--center-method", default="vwap", help="Center estimator passed to the runners.")
    parser.add_argument("--output-root", default="", help="Optional directory for smoke-test outputs.")
    parser.add_argument("--with-ib", action="store_true", help="Also run an IB delayed live-prep smoke test.")
    parser.add_argument("--ib-client-id", type=int, default=61, help="IB client id for the optional IB smoke step.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed step and summarize all results.")
    return parser.parse_args(argv)


def default_output_root() -> Path:
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    return Path("corridor_outputs") / "smoke_test" / stamp


def run_step(name: str, command: list[str], workdir: Path, log_path: Path) -> StepResult:
    print(f"[smoke] running {name}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=str(workdir),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = []
    if completed.stdout:
        combined.append(completed.stdout.rstrip())
    if completed.stderr:
        combined.append(completed.stderr.rstrip())
    log_path.write_text("\n\n".join(combined) + ("\n" if combined else ""), encoding="utf-8")
    return StepResult(name=name, command=command, returncode=completed.returncode, log_path=str(log_path))


def write_summary(output_root: Path, steps: list[StepResult]) -> Path:
    summary_path = output_root / "summary.json"
    payload = {
        "ok": all(step.ok for step in steps),
        "steps": [
            {
                **asdict(step),
                "ok": step.ok,
            }
            for step in steps
        ],
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    workdir = Path(__file__).resolve().parent
    bars_csv = (workdir / args.bars_csv).resolve()
    if not bars_csv.exists():
        print(f"Bars CSV not found: {bars_csv}", file=sys.stderr)
        return 1

    output_root = Path(args.output_root).expanduser() if args.output_root else default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"

    steps_to_run: list[tuple[str, list[str]]] = [
        (
            "unit_tests",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        ),
        (
            "backtest_underlying",
            [
                sys.executable,
                "run_backtest.py",
                "--symbol",
                args.symbol,
                "--bars-csv",
                str(bars_csv),
                "--payoff-mode",
                "underlying_only",
                "--center-method",
                args.center_method,
                "--output-dir",
                str(output_root / "underlying"),
            ],
        ),
        (
            "backtest_simplified",
            [
                sys.executable,
                "run_backtest.py",
                "--symbol",
                args.symbol,
                "--bars-csv",
                str(bars_csv),
                "--payoff-mode",
                "simplified",
                "--center-method",
                args.center_method,
                "--output-dir",
                str(output_root / "simplified"),
            ],
        ),
        (
            "live_prep_csv",
            [
                sys.executable,
                "run_live_prep.py",
                "--symbol",
                args.symbol,
                "--bars-csv",
                str(bars_csv),
                "--output",
                str(output_root / "live_prep" / "snapshot.json"),
            ],
        ),
    ]

    if args.with_ib:
        steps_to_run.append(
            (
                "live_prep_ib_delayed",
                [
                    sys.executable,
                    "run_live_prep.py",
                    "--symbol",
                    args.symbol,
                    "--mode",
                    "delayed",
                    "--client-id",
                    str(args.ib_client_id),
                    "--output",
                    str(output_root / "live_prep_ib" / "snapshot.json"),
                ],
            )
        )

    results: list[StepResult] = []
    for name, command in steps_to_run:
        result = run_step(name, command, workdir, logs_dir / f"{name}.log")
        results.append(result)
        if result.ok:
            print(f"[smoke] ok {name}")
            continue
        print(f"[smoke] failed {name} -> {result.log_path}", file=sys.stderr)
        if not args.keep_going:
            break

    summary_path = write_summary(output_root, results)
    print(f"[smoke] summary: {summary_path}")

    if all(step.ok for step in results) and len(results) == len(steps_to_run):
        print(f"[smoke] all {len(results)} steps passed")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

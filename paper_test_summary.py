#! python3.12
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

from corridor.execution.paper import build_paper_test_summary, format_paper_test_summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the latest daily paper test summary.")
    parser.add_argument("--symbol", default="SPX", help="Symbol used in the paper runner output directory.")
    parser.add_argument("--output-dir", default="", help="Override the paper runner output directory.")
    parser.add_argument("--prefix", default="paper", help="Paper runner file prefix.")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON instead of text.")
    parser.add_argument("--write", action="store_true", help="Write fresh summary json/csv/txt files.")
    return parser.parse_args(argv)


def _default_output_dir(symbol: str) -> Path:
    return Path("corridor_outputs") / "paper_runner" / symbol.upper()


def _write_csv(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.symbol)
    prefix = str(args.prefix)

    state_path = output_dir / f"{prefix}_state.json"
    daily_report_path = output_dir / f"{prefix}_daily_report.json"
    summary_json_path = output_dir / f"{prefix}_test_summary.json"
    summary_csv_path = output_dir / f"{prefix}_test_summary.csv"
    summary_txt_path = output_dir / f"{prefix}_test_summary.txt"

    if state_path.exists() and daily_report_path.exists():
        state_payload = json.loads(state_path.read_text(encoding="utf-8"))
        daily_report_payload = json.loads(daily_report_path.read_text(encoding="utf-8"))
        summary_payload = build_paper_test_summary(state_payload, daily_report_payload)
        summary_text = format_paper_test_summary(summary_payload)
        if args.write:
            summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
            _write_csv(summary_csv_path, summary_payload)
            summary_txt_path.write_text(summary_text, encoding="utf-8")
    elif summary_json_path.exists():
        summary_payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
        summary_text = format_paper_test_summary(summary_payload)
    else:
        print(
            f"Could not find daily report/state in {output_dir}. "
            "Run the paper runner first or pass --output-dir."
        )
        return 1

    if args.json:
        print(json.dumps(summary_payload, indent=2))
    else:
        print(summary_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

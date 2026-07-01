import argparse
import json
from pathlib import Path

from helpers.records import load_eval_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull JSONL evaluation records.")
    parser.add_argument("--path", default="output/logs/framework/eval_records.jsonl", help="Evaluation JSONL path.")
    parser.add_argument("--run-id", default=None, help="Optional run id filter.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum records to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_eval_records(Path(args.path), run_id=args.run_id)
    if args.limit is not None:
        records = records[: args.limit]
    print(json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

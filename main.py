import asyncio
import os
import random
from pathlib import Path

from dotenv import load_dotenv

from eval.medqa_smoke import run_medqa_smoke_eval
from helpers.config import startup_check


def main() -> None:
    load_dotenv()
    startup_check()

    random.seed(int(os.environ.get("RANDOM_SEED", "42")))
    sample_size = int(os.environ.get("EVAL_SAMPLE_SIZE", "1"))
    input_dir = Path(os.environ["INPUT_BASE_DIR"])
    output_dir = Path(os.environ["OUTPUT_BASE_DIR"])

    asyncio.run(
        run_medqa_smoke_eval(
            input_dir=input_dir,
            output_dir=output_dir,
            sample_size=sample_size,
        )
    )


if __name__ == "__main__":
    main()

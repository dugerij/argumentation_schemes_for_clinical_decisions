import asyncio
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from graphrag.config.load_config import load_config

from eval.medqa_smoke import run_medqa_smoke_eval
from helpers.config import startup_check


CONFIG_PATH = Path("settings.yaml")


def main() -> None:
    load_dotenv()
    startup_check()

    random.seed(int(os.environ.get("RANDOM_SEED", "42")))
    sample_size = int(os.environ.get("EVAL_SAMPLE_SIZE", "1"))
    index_method = os.environ.get("GRAPHRAG_INDEX_METHOD", "standard")
    output_dir = Path(os.environ["OUTPUT_BASE_DIR"])
    config = load_config(CONFIG_PATH)

    asyncio.run(run_medqa_smoke_eval(config, output_dir, sample_size, index_method))


if __name__ == "__main__":
    main()

# imports
import asyncio
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from graphrag.config.load_config import load_config
from graphrag import api

from make_index import build_index
from utils import startup_check, load_dataset
    

load_dotenv()
startup_check() # Ensure required variables are set
graphrag_config = load_config(Path('./'))


asyncio.run(build_index(graph_config=graphrag_config))

entities = pd.read_parquet("./output/entities.parquet")
communities = pd.read_parquet("./output/communities.parquet")
community_reports = pd.read_parquet(
        "./output/community_reports.parquet"
)

response, context = asyncio.run(
    api.local_search(
    config=graphrag_config,
    entities=entities,
    communities=communities,
    community_reports=community_reports,
    community_level=2,
    dynamic_community_selection=False,
    response_type="Multiple Paragraphs",
    query="Who is Scrooge and what are his main relationships?",
    )
)

print("Response:", response)
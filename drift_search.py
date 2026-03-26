import os
from dotenv import load_dotenv

import pandas as pd
from graphrag.config.enums import ModelType
from graphrag.config.models.drift_search_config import DRIFTSearchConfig
from graphrag.config.defaults import DriftSearchDefaults
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_report_embeddings,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.structured_search.drift_search.drift_context import (
    DRIFTSearchContextBuilder,
)
from graphrag.query.structured_search.drift_search.search import DRIFTSearch
from graphrag.tokenizer.get_tokenizer import get_tokenizer
from graphrag_vectors.lancedb import LanceDBVectorStore

load_dotenv()
output_dir = os.environ.get('DEFAULT_OUTPUT_BASE_DIR')

RAG_INPUT_DIR = f"./{DEFAULT_INPUT_BASE_DIR}/data/medqa/data_clean/textbooks/en"
LANCEDB_URI = f"./{output_dir}/rag_files/lancedb"

COMMUNITY_REPORT_TABLE = "community_reports"
COMMUNITY_TABLE = "communities"
ENTITY_TABLE = "entities"
RELATIONSHIP_TABLE = "relationships"
COVARIATE_TABLE = "covariates"
TEXT_UNIT_TABLE = "text_units"
COMMUNITY_LEVEL = 2


# read nodes table to get community and degree data
entity_df = pd.read_parquet(f"{output_dir}/{ENTITY_TABLE}.parquet")
community_df = pd.read_parquet(f"{output_dir}/{COMMUNITY_TABLE}.parquet")

print(f"Entity df columns: {entity_df.columns}")

entities = read_indexer_entities(entity_df, community_df, COMMUNITY_LEVEL)

# load description embeddings to an in-memory lancedb vectorstore
# to connect to a remote db, specify url and port values.
description_embedding_store = LanceDBVectorStore(
    db_uri=LANCEDB_URI,
    index_name="entity_description",
)
description_embedding_store.connect()

full_content_embedding_store = LanceDBVectorStore(
    db_uri=LANCEDB_URI,
    index_name="community_full_content",
)
full_content_embedding_store.connect()

print(f"Entity count: {len(entity_df)}")
entity_df.head()

relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
relationships = read_indexer_relationships(relationship_df)

print(f"Relationship count: {len(relationship_df)}")
relationship_df.head()

text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
text_units = read_indexer_text_units(text_unit_df)

print(f"Text unit records: {len(text_unit_df)}")
text_unit_df.head()

report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
reports = read_indexer_reports(report_df, community_df, COMMUNITY_LEVEL)
read_indexer_report_embeddings(reports, full_content_embedding_store)
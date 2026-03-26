from pathlib import Path
from pprint import pprint

import graphrag.api as api
import pandas as pd
from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.typing.pipeline_run_result import PipelineRunResult

async def build_index(graph_config: GraphRagConfig):

    index_result: list[PipelineRunResult] = await api.build_index(config=graph_config)

    # index_result is a list of workflows that make up the indexing pipeline that was run
    for workflow_result in index_result:
        status = f"error\n{workflow_result.error}" if workflow_result.error else "success"
        print(f"Workflow Name: {workflow_result.workflow}\tStatus: {status}")

    return index_result
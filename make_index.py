from pathlib import Path
from pprint import pprint

from graphrag.config.load_config import load_config
from graphrag.index.typing.pipeline_run_result import PipelineRunResult
from graphrag.cli.index import index_cli
from graphrag.config.enums import IndexingMethod


async def build_index(
    root_dir: Path = None,
    method: IndexingMethod = IndexingMethod.Standard,
    verbose: bool = True,
    cache: bool = True,
    dry_run: bool = False,
    skip_validation: bool = False,
) -> list[PipelineRunResult]:
    """
    Run the GraphRAG index pipeline using the CLI interface.
    Adapted from the original CLI command to be called from Python code, with error handling and result reporting.
    https://github.com/microsoft/graphrag/discussions/513
    
    Args:
        root_dir: Root directory for input/output data (defaults to current directory)
        method: Indexing method (Full, Update, or Diff)
        verbose: Enable verbose logging
        cache: Enable LLM cache (set to False to disable)
        dry_run: Run without executing steps
        skip_validation: Skip validation checks

    Returns:
        List containing a PipelineRunResult with success/error status
    """
    try:
        if root_dir is None:
            root_dir = Path("./")
        else:
            root_dir = Path(root_dir)
            
        index_cli(
            root_dir=root_dir,
            method=method,
            verbose=verbose,
            cache=cache,
            dry_run=dry_run,
            skip_validation=skip_validation,
        )
        index_result = [PipelineRunResult(workflow="index", error=None)]
    except Exception as e:
        error_msg = str(e)
        index_result = [PipelineRunResult(workflow="index", error=error_msg)]

    # Print statuses
    for workflow_result in index_result:
        status = f"error\n{workflow_result.error}" if workflow_result.error else "success"
        print(f"Workflow Name: {workflow_result.workflow}\tStatus: {status}")

    return index_result

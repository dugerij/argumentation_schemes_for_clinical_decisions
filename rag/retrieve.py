from graphrag import api


async def drift_search_context(
    config,
    query: str,
    entities,
    communities,
    community_reports,
    community_level: int = 2,
    response_type: str = "Single Sentence",
):
    return await api.drift_search(
        config=config,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        community_level=community_level,
        dynamic_community_selection=True,
        response_type=response_type,
        query=query,
    )

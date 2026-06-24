import logging
from app.agent.state import ResearchState
from app.agent.tools.search import web_search

logger = logging.getLogger(__name__)


def research_node(state: ResearchState) -> dict:
    """
    Research Node:
    Queries the search tool and returns the collected search results to update the state.
    """
    query = state.get("query")
    logger.info(f"Research node started searching for query: {query}")

    if not query:
        logger.warning("Empty query in state. Skipping search.")
        return {"search_results": []}

    # Execute search
    results = web_search(query=query, max_results=5)
    logger.info(f"Research node gathered {len(results)} search results.")

    return {"search_results": results}

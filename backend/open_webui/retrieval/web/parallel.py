from __future__ import annotations

import requests

from open_webui.retrieval.web.main import SearchResult, get_filtered_results


def search_parallel(
    api_key: str,
    query: str,
    count: int | None,
    filter_list: list[str] | None = None,
) -> list[SearchResult]:
    """Search using Parallel's Search API and return the results as SearchResult objects.

    Args:
        api_key (str): A Parallel API key
        query (str): The query to search for
        count (int | None): The maximum number of results to return
        filter_list (list[str] | None): The domains to allow or block

    Returns:
        A list of SearchResult objects.
    """
    payload: dict[str, object] = {'search_queries': [query]}
    if count:
        payload['advanced_settings'] = {'max_results': count}

    response = requests.post(
        'https://api.parallel.ai/v1/search',
        headers={
            'Content-Type': 'application/json',
            'x-api-key': api_key,
        },
        json=payload,
        timeout=30,
    )
    response.raise_for_status()

    results = response.json().get('results', [])
    if filter_list:
        results = get_filtered_results(results, filter_list)
    if count:
        results = results[:count]

    return [
        SearchResult(
            link=result.get('url', ''),
            title=result.get('title'),
            snippet='\n\n'.join(result.get('excerpts') or []),
        )
        for result in results
    ]

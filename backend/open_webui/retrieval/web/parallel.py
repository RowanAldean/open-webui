from __future__ import annotations

import requests

from open_webui.retrieval.web.main import SearchResult, get_filtered_results


def search_parallel(
    api_key: str,
    query: str,
    count: int,
    filter_list: list[str] | None = None,
) -> list[SearchResult]:
    """Search using Parallel's Search API and return SearchResult objects."""
    response = requests.post(
        'https://api.parallel.ai/v1/search',
        headers={
            'Content-Type': 'application/json',
            'x-api-key': api_key,
        },
        json={
            'search_queries': [query],
            'advanced_settings': {'max_results': count},
        },
    )
    response.raise_for_status()

    results = response.json().get('results', [])
    if filter_list:
        results = get_filtered_results(results, filter_list)

    return [
        SearchResult(
            link=result['url'],
            title=result.get('title'),
            snippet='\n\n'.join(result.get('excerpts', [])),
        )
        for result in results
    ]

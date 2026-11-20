import logging
import math
from typing import Iterator, Sequence

import requests
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

log = logging.getLogger(__name__)


class ParallelLoader(BaseLoader):
    """Extract web page content from URLs using Parallel's Extract API."""

    def __init__(
        self,
        web_paths: str | Sequence[str],
        api_key: str,
        timeout: str | None = None,
        verify_ssl: bool = True,
        trust_env: bool = False,
        continue_on_failure: bool = True,
        **kwargs,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError('Parallel web loader requires a PARALLEL_API_KEY')

        self.web_paths = [web_paths] if isinstance(web_paths, str) else list(web_paths)
        self.api_key = api_key
        try:
            request_timeout = float(timeout)
        except (TypeError, ValueError):
            request_timeout = 60
        self.timeout = request_timeout if math.isfinite(request_timeout) and request_timeout > 0 else 60
        self.verify_ssl = verify_ssl
        self.trust_env = trust_env
        self.continue_on_failure = continue_on_failure

    def lazy_load(self) -> Iterator[Document]:
        loaded = 0
        with requests.Session() as session:
            session.trust_env = self.trust_env
            session.verify = self.verify_ssl
            session.headers.update(
                {
                    'Content-Type': 'application/json',
                    'x-api-key': self.api_key,
                }
            )

            for offset in range(0, len(self.web_paths), 20):
                urls = self.web_paths[offset : offset + 20]
                try:
                    response = session.post(
                        'https://api.parallel.ai/v1/extract',
                        json={
                            'urls': urls,
                            'advanced_settings': {'full_content': True},
                        },
                        timeout=self.timeout,
                    )
                    if response.status_code in (401, 402, 403):
                        raise PermissionError(
                            f'Parallel web loader authentication or billing failed (HTTP {response.status_code})'
                        )
                    response.raise_for_status()
                    data = response.json()
                    if not isinstance(data, dict) or not isinstance(data.get('results'), list):
                        raise ValueError('Invalid Parallel Extract response')
                except PermissionError:
                    raise
                except (requests.RequestException, ValueError) as e:
                    log.warning('Parallel web loader batch failed (%s)', type(e).__name__)
                    if not self.continue_on_failure:
                        raise ValueError('Parallel web loader request failed') from None
                    continue

                errors = data.get('errors') or []
                if errors:
                    log.warning('Parallel web loader could not load %s URL(s)', len(errors))
                    if not self.continue_on_failure:
                        raise ValueError(f'Parallel web loader could not load {len(errors)} URL(s)')

                for result in data['results']:
                    if not isinstance(result, dict):
                        continue

                    source = result.get('url')
                    full_content = result.get('full_content')
                    excerpts = result.get('excerpts') or []
                    content = (
                        full_content
                        if isinstance(full_content, str) and full_content.strip()
                        else '\n\n'.join(excerpt for excerpt in excerpts if isinstance(excerpt, str))
                    )
                    if not isinstance(source, str) or not content.strip():
                        continue

                    metadata = {'source': source}
                    if isinstance(result.get('title'), str):
                        metadata['title'] = result['title']
                    if isinstance(result.get('publish_date'), str):
                        metadata['publish_date'] = result['publish_date']

                    loaded += 1
                    yield Document(page_content=content, metadata=metadata)

        if not loaded:
            raise ValueError('Parallel web loader could not load any page content')

import logging
import math
import time
from typing import AsyncIterator, Iterator, Sequence

import requests
from fastapi.concurrency import run_in_threadpool
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
        requests_per_second: float | None = None,
        continue_on_failure: bool = True,
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
        self.requests_per_second = requests_per_second
        self.last_request_time: float | None = None
        self.continue_on_failure = continue_on_failure

    def _sync_wait_for_rate_limit(self) -> None:
        if not self.requests_per_second or self.requests_per_second <= 0:
            return

        now = time.monotonic()
        if self.last_request_time is not None:
            wait = (1 / self.requests_per_second) - (now - self.last_request_time)
            if wait > 0:
                time.sleep(wait)
                now += wait
        self.last_request_time = now

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
                self._sync_wait_for_rate_limit()
                try:
                    response = session.post(
                        'https://api.parallel.ai/v1/extract',
                        json={
                            'urls': urls,
                            'advanced_settings': {'full_content': True},
                        },
                        timeout=self.timeout,
                        allow_redirects=False,
                    )
                    if response.status_code in (401, 402, 403):
                        raise PermissionError(
                            f'Parallel web loader authentication or billing failed (HTTP {response.status_code})'
                        )
                    response.raise_for_status()
                    data = response.json()
                    if not isinstance(data, dict) or not isinstance(data.get('results'), list):
                        raise ValueError('Invalid Parallel Extract response')
                    if data.get('errors') is not None and not isinstance(data['errors'], list):
                        raise ValueError('Invalid Parallel Extract errors')
                except PermissionError:
                    raise
                except (requests.RequestException, ValueError) as e:
                    log.warning('Parallel web loader batch failed (%s)', type(e).__name__)
                    if not self.continue_on_failure:
                        raise ValueError('Parallel web loader request failed') from None
                    continue

                failed_urls = {
                    error.get('url')
                    for error in (data.get('errors') or [])
                    if isinstance(error, dict) and isinstance(error.get('url'), str)
                }

                documents = {}
                for result in data['results']:
                    if not isinstance(result, dict):
                        continue

                    source = result.get('url')
                    if source not in urls or source in failed_urls:
                        continue

                    full_content = result.get('full_content')
                    excerpts = result.get('excerpts')
                    if not isinstance(excerpts, list):
                        excerpts = []
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

                    documents[source] = Document(page_content=content, metadata=metadata)

                missing = len(set(urls) - documents.keys())
                if missing:
                    log.warning('Parallel web loader could not load %s URL(s)', missing)
                    if not self.continue_on_failure:
                        raise ValueError(f'Parallel web loader could not load {missing} URL(s)')
                for url in urls:
                    if url in documents:
                        loaded += 1
                        yield documents[url]

        if not loaded:
            raise ValueError('Parallel web loader could not load any page content')

    async def alazy_load(self) -> AsyncIterator[Document]:
        docs = await run_in_threadpool(lambda: list(self.lazy_load()))
        for doc in docs:
            yield doc

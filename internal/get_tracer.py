import os
from phoenix.otel import register
from unittest.mock import MagicMock
from internal.config import settings
import requests

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = settings.PHOENIX_COLLECTOR_ENDPOINT
tracer_provider = register(
    project_name="india_solar_scraper",
    auto_instrument=True,
)


class EmptyDecorator:
    def chain(self, func):
        print(f"Calling the empty decorator with function {func.__name__}.")
        return func

    def start_as_current_span(self, name, **kwargs):
        return MagicMock()


def _is_endpoint_available(endpoint: str, timeout: float = 1.0) -> bool:
    try:
        resp = requests.head(endpoint, timeout=timeout)
        return resp.status_code < 400
    except Exception:
        return False


if _is_endpoint_available(settings.PHOENIX_COLLECTOR_ENDPOINT):
    tracer = tracer_provider.get_tracer(__name__)
else:
    tracer = EmptyDecorator()

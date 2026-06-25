"""OpenTelemetry tracing hooks (optional)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from config.settings import OTEL_ENABLED_ENV

_tracer = None
_configured = False


def setup_telemetry(service_name: str = "rag-retrieval-lab") -> None:
    global _tracer, _configured
    if _configured or os.getenv(OTEL_ENABLED_ENV, "").lower() not in ("1", "true", "yes"):
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name)
        _configured = True
    except Exception:
        _tracer = None


@contextmanager
def trace_span(name: str, **attributes: str) -> Iterator[None]:
    setup_telemetry()
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as span:
        for k, v in attributes.items():
            span.set_attribute(k, v)
        yield

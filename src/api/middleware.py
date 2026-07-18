from __future__ import annotations

import logging
import re
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_MAX_LENGTH = 128
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
logger = logging.getLogger("requests_classifier.api")


def normalize_request_id(raw: str | None) -> str:
    if raw is None:
        return uuid.uuid4().hex
    value = raw.strip()
    if (
        not value
        or len(value) > REQUEST_ID_MAX_LENGTH
        or not REQUEST_ID_PATTERN.fullmatch(value)
    ):
        return uuid.uuid4().hex
    return value


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = normalize_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        started = time.perf_counter()
        logger.info(
            "request_started",
            extra={
                "requestId": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )
        try:
            response = await call_next(request)
        except Exception as error:
            latency_ms = (time.perf_counter() - started) * 1000
            logger.error(
                "request_failed",
                extra={
                    "requestId": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "latencyMs": latency_ms,
                    "errorType": type(error).__name__,
                },
            )
            raise
        latency_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request_completed",
            extra={
                "requestId": request_id,
                "method": request.method,
                "path": request.url.path,
                "statusCode": response.status_code,
                "latencyMs": latency_ms,
            },
        )
        return response

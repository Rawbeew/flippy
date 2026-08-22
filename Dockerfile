# flippy — stdlib-only router container. No pip installs needed at runtime.
FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PORT=8080

# Copy the source tree (src/loomweaver + src/flippy_providers + src/server.py).
COPY src/ ./src/

# Run as non-root.
RUN useradd --system --uid 10001 flippy && chown -R flippy /app
USER flippy

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD ["python", "-c", "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"]

CMD ["python", "src/server.py"]

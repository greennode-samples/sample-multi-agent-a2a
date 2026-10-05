FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY src/backend/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Run as an unprivileged user. The application writes nothing to disk: state lives in AgentBase Memory.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app
COPY src/backend ./backend
COPY src/frontend ./frontend

USER 10001
WORKDIR /app/backend
EXPOSE 8080

# /health is the SDK liveness probe (public even when AGENT_API_KEY is set).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"]

CMD ["python", "main.py"]

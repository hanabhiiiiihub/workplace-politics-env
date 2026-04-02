FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    httpx \
    groq \
    python-dotenv

RUN pip install --no-cache-dir \
    git+https://github.com/meta-pytorch/OpenEnv.git#subdirectory=src/openenv

# Debug: show exactly what got installed and where
RUN python -c "import openenv; print(openenv.__file__)" || echo "openenv not found"
RUN find / -name "env_server" -type d 2>/dev/null || echo "env_server not found"
RUN pip show openenv-core || pip list | grep -i open

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
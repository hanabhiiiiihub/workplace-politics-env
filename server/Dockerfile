FROM python:3.11-slim

WORKDIR /app

# Copy everything
COPY . .

# Install OpenEnv from source
RUN pip install --no-cache-dir \
    fastapi uvicorn pydantic httpx \
    websockets groq python-dotenv

# Install openenv-core
RUN pip install --no-cache-dir openenv-core 2>/dev/null || \
    pip install --no-cache-dir -e OpenEnv/src/openenv 2>/dev/null || true

ENV PYTHONPATH=/app:/app/OpenEnv/src/openenv

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

Also create `requirements.txt`:
```
fastapi
uvicorn
pydantic
httpx
groq
python-dotenv
openenv-core
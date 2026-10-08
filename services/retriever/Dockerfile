FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV FASTEMBED_CACHE_PATH=/srv/models DOCS_DIR=/srv/data/docs INDEX_DIR=/srv/data/index
COPY retriever ./retriever
COPY data/docs ./data/docs
# Download the embedding model and embed the whole corpus at build time,
# so pods start in seconds and never need internet access.
RUN python -m retriever.build_index
RUN useradd -r -u 10001 app && chown -R app /srv
USER 10001
EXPOSE 8001
CMD ["uvicorn", "retriever.main:app", "--host", "0.0.0.0", "--port", "8001"]

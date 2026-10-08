#!/usr/bin/env bash
# Download the Markdown docs for vLLM and NVIDIA Triton Inference Server (the RAG corpus).
# Uses a shallow, sparse git checkout so only the docs folders are fetched.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=services/retriever/data/docs
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$DEST"

fetch() {  # repo  docs_subdir  name
  git clone --quiet --depth 1 --filter=blob:none --sparse "https://github.com/$1.git" "$TMP/$3"
  git -C "$TMP/$3" sparse-checkout set "$2"
  rm -rf "${DEST:?}/$3"
  mkdir -p "$DEST/$3"
  (cd "$TMP/$3/$2" && find . -name '*.md' -type f -exec cp --parents {} "$OLDPWD/$DEST/$3/" \;)
  echo "$3: $(find "$DEST/$3" -name '*.md' | wc -l) markdown files"
}

fetch triton-inference-server/server docs triton
fetch vllm-project/vllm docs vllm

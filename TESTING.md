# Test report

What was run, where, and with what result, so anyone can reproduce it.
Environment for the live runs: Linux x86_64, **2 vCPUs, 8 GB RAM, no GPU**, Docker 29.8, vLLM 0.29.0 CPU.

## 1. Automated tests (no model needed)

```bash
pip install -r requirements-dev.txt
pytest -q            # 88 passed
python -m pyflakes services bench tests   # clean
```

| Area | What is covered |
|---|---|
| Chunking and indexing | heading-aware chunks, overlap, save/load round trip, rebuild when docs change (fingerprint) |
| Retrieval | tokenizer keeps `config.pbtxt` and `gpu-memory-utilization`, BM25 exact-term ranking, dense/BM25/hybrid modes |
| Agent loop | search → answer, search → evidence check → revise, refuses to answer before searching, step limit forces a grounded answer |
| Malformed model output | `[1]`, `true`, `null`, `5`, `"text"`, `not json`, `{broken`, empty reply, `{"query": 5}`, `{"answer": ["x"]}`, `{"action": 7}`, a list wrapping an object: no crash, safe fallback |
| Evidence check | only a real JSON boolean counts. `not json`, `[true]`, `true`, `{"supported": "yes"}`, `{"supported": 1}` and `{}` all give `verified: null` (unverified), never success, and trigger no extra search |
| Answer cleanup | does exactly two things: removes `[…]` around a whole sentence-like answer (`[Triton batches requests [S2].]` → `Triton batches requests [S2].`), and removes matching quotes around the whole answer. It never touches angle brackets, parentheses or braces, so HTML (`<img src="diagram.png">`, `<a href=…>`), asides and objects pass through unchanged. Also kept as-is: valid JSON (`{"enabled": true}`, `["hello", "world"]`), citations (`[S1] says …`), single tokens (`[TODO]`) and Markdown links. |
| HTTP API | all malformed outputs above return **200** with a non-empty answer in both modes. A model server that's down returns **502**. Invalid requests return **422**. |

## 2. Live runs with the real model

| Check | Result |
|---|---|
| vLLM 0.29.0 CPU serving Qwen2.5-0.5B-Instruct (`bfloat16`) | starts in about 40–65 s; 0 errors across every benchmark and query |
| `bench/bench_llm.py`, concurrency 1/4/8 | 10.5 / 10.7 / 15.9 output tok/s (2 vCPUs); TTFT p50 0.8 s at concurrency 1 |
| `bench/eval_retrieval.py`, 20 questions, 4,722 chunks | dense 60% / 90%, BM25 75% / 90%, **hybrid 80% / 95%** (hit@1 / hit@4) |
| `bench/bench_gateway.py`, 5 questions | rag p50 17.2 s, agent p50 27.9 s (3.4 LLM calls); 98% of time in the LLM |
| UI in headless Chromium, desktop 900 px and mobile 390 px | renders, citations link to sources, error state shown when vLLM is down, no console errors, no horizontal scroll |

## 3. Docker

```bash
docker build -t docs-rag/gateway:dev services/gateway        # 248 MB (multi-stage: Node build + Python)
docker build -t docs-rag/retriever:dev services/retriever    # 814 MB (model + prebuilt index baked in)
docker compose up
```

- Both images built. The full `docker-compose.yml` stack ran with 3 containers healthy, and the gateway reached `vllm:8000` and `retriever:8001` by service name.
- The retriever container loaded its prebuilt index in under 1 s and served searches in about 20 ms.
- 6 real questions, 3 in each mode, went through the containerized stack: **6/6 HTTP 200**, with evidence-check results `True`/`None`/forced reported honestly.
- The first answers after vLLM starts are slow (about 2 minutes) while it compiles CPU kernels; after warm-up a short reply takes about 1.5 s.

Two sandbox-only differences from a normal laptop run:
1. The sandbox's network proxy certificate was added to temporary copies of the Dockerfiles.
2. The model was loaded from a local cache in `bfloat16` to fit 8 GB RAM. The default is `float32`; see Troubleshooting in the README.

## 4. Kubernetes

```bash
kustomize build k8s/base          | kubeconform -strict -kubernetes-version 1.30.0   # 10/10 valid
kustomize build k8s/overlays/gpu  | kubeconform -strict -kubernetes-version 1.30.0   # 10/10 valid
```

**Not yet run on a live cluster.** The manifests are schema-valid, and the images and configuration are the ones verified in Docker, but `kubectl apply` on minikube (README, step 3b) has not been executed. Do that on your machine before claiming a Kubernetes deployment.

## 5. Not yet tested

- vLLM on an NVIDIA GPU: run `notebooks/free_gpu_benchmark.ipynb` on a free Kaggle T4.
- A live Kubernetes cluster (minikube), the GPU overlay on a GPU node, and the Windows/WSL2 setup steps.

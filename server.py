#!/usr/bin/env python3
import json
import math
import os
import threading
import time
import warnings
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("PYTHONWARNINGS", "ignore")

try:
    from urllib3.exceptions import NotOpenSSLWarning

    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
except Exception as exc:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TextIteratorStreamer = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""


HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
DIST_CACHE_MAX_ENTRIES = int(os.environ.get("DIST_CACHE_MAX_ENTRIES", "128"))


MODEL_CACHE = {}
CACHE_LOCK = threading.Lock()


def _normalize_model_id(model_id):
    return (model_id or "gpt2-large").strip() or "gpt2-large"


def _load_runtime(model_id):
    model_id = _normalize_model_id(model_id)
    with CACHE_LOCK:
        if model_id in MODEL_CACHE:
            return MODEL_CACHE[model_id]

    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError(f"Missing local ML deps: {IMPORT_ERROR}")

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)
    model.eval()
    runtime = {
        "model_id": model_id,
        "device": device,
        "tokenizer": tokenizer,
        "model": model,
        "loaded_in_s": round(time.time() - started, 2),
        "dist_cache": {},
        "dist_cache_order": [],
        "dist_cache_lock": threading.Lock(),
    }
    with CACHE_LOCK:
        MODEL_CACHE[model_id] = runtime
    return runtime


class DecodingLabAPIHandler(SimpleHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return None

    def _start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _send_sse(self, payload):
        body = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()

    def _generate(self, runtime, prompt, max_tokens, temperature, top_p, top_k, beam_size=1):
        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        device = runtime["device"]

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        max_tokens = max(1, min(int(max_tokens), 200))
        temperature = float(temperature)
        top_p = float(top_p)
        top_k = int(top_k) if top_k is not None else 0

        beam_size = max(1, int(beam_size or 1))
        do_sample = (temperature > 0.0001) and beam_size == 1
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if beam_size > 1:
            gen_kwargs.update(
                {
                    "do_sample": False,
                    "num_beams": beam_size,
                    "early_stopping": True,
                }
            )
        elif do_sample:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": max(temperature, 0.05),
                    "top_p": min(max(top_p, 0.01), 1.0),
                }
            )
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
        else:
            gen_kwargs.update({"do_sample": False})

        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        elapsed_s = max(time.perf_counter() - started, 1e-6)
        new_ids = output[0][input_ids.shape[1] :]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        completion_tokens = int(new_ids.shape[0])
        prompt_tokens = int(input_ids.shape[1])
        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_ms": int(round(elapsed_s * 1000)),
            "tokens_per_second": float(completion_tokens / elapsed_s),
        }

    def _stream_generate(self, runtime, prompt, max_tokens, temperature, top_p, top_k, beam_size=1):
        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        device = runtime["device"]

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        max_tokens = max(1, min(int(max_tokens), 200))
        temperature = float(temperature)
        top_p = float(top_p)
        top_k = int(top_k) if top_k is not None else 0
        beam_size = max(1, int(beam_size or 1))
        do_sample = (temperature > 0.0001) and beam_size == 1

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "streamer": streamer,
        }
        if beam_size > 1:
            gen_kwargs.update(
                {
                    "do_sample": False,
                    "num_beams": beam_size,
                    "early_stopping": True,
                }
            )
        elif do_sample:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": max(temperature, 0.05),
                    "top_p": min(max(top_p, 0.01), 1.0),
                }
            )
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
        else:
            gen_kwargs["do_sample"] = False

        produced = []
        worker_error = {}
        started = time.perf_counter()

        def _worker():
            try:
                with torch.inference_mode():
                    model.generate(**gen_kwargs)
            except Exception as exc:
                worker_error["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        for chunk in streamer:
            if not chunk:
                continue
            produced.append(chunk)
            yield {"type": "delta", "text": chunk}

        thread.join()
        if worker_error.get("error"):
            raise RuntimeError(worker_error["error"])

        elapsed_s = max(time.perf_counter() - started, 1e-6)
        completion_text = "".join(produced)
        completion_tokens = len(tokenizer(completion_text, add_special_tokens=False)["input_ids"])
        prompt_tokens = int(input_ids.shape[1])
        yield {
            "type": "done",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "performance": {
                "elapsed_ms": int(round(elapsed_s * 1000)),
                "tokens_per_second": float(completion_tokens / elapsed_s),
            },
        }

    def _next_token_distribution(self, runtime, prompt, top_n, top_k=None, top_p=None):
        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        device = runtime["device"]
        prompt_key = prompt

        top_n = max(5, min(int(top_n), 320))
        with runtime["dist_cache_lock"]:
            cached = runtime["dist_cache"].get(prompt_key)
            if cached is not None:
                sorted_probs = cached["sorted_probs"]
                sorted_indices = cached["sorted_indices"]
            else:
                sorted_probs = None
                sorted_indices = None

        if sorted_probs is None or sorted_indices is None:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with torch.inference_mode():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
                sorted_probs_dev, sorted_indices_dev = torch.sort(probs, descending=True)
                sorted_probs = sorted_probs_dev.detach().to("cpu")
                sorted_indices = sorted_indices_dev.detach().to("cpu")

            with runtime["dist_cache_lock"]:
                if prompt_key in runtime["dist_cache"]:
                    try:
                        runtime["dist_cache_order"].remove(prompt_key)
                    except ValueError:
                        pass
                runtime["dist_cache"][prompt_key] = {
                    "sorted_probs": sorted_probs,
                    "sorted_indices": sorted_indices,
                }
                runtime["dist_cache_order"].append(prompt_key)
                while len(runtime["dist_cache_order"]) > DIST_CACHE_MAX_ENTRIES:
                    old_key = runtime["dist_cache_order"].pop(0)
                    runtime["dist_cache"].pop(old_key, None)

        vocab_size = int(sorted_probs.shape[0])
        top_n = min(top_n, vocab_size)
        top_k = max(1, min(int(top_k or 20), vocab_size))
        top_p = float(top_p if top_p is not None else 0.9)
        top_p = min(max(top_p, 0.01), 1.0)

        cumulative = torch.cumsum(sorted_probs, dim=-1)
        top_p_rank = int(torch.searchsorted(cumulative, torch.tensor(top_p), right=False).item() + 1)
        top_p_rank = max(1, min(top_p_rank, vocab_size))

        tokens = []
        for rank in range(top_n):
            idx = int(sorted_indices[rank].item())
            p = float(sorted_probs[rank].item())
            token_text = tokenizer.decode([idx], clean_up_tokenization_spaces=False)
            safe_p = max(p, 1e-12)
            tokens.append(
                {
                    "token": token_text,
                    "prob": safe_p,
                    "logprob": float(math.log(safe_p)),
                    "rank": rank + 1,
                    "in_top_k": rank < top_k,
                    "in_top_p": rank < top_p_rank,
                }
            )
        return {
            "tokens": tokens,
            "top_k_rank": top_k,
            "top_p_rank": top_p_rank,
            "vocab_size": vocab_size,
        }

    def do_OPTIONS(self):
        if self.path.startswith("/api/"):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            return
        self.send_error(405, "Method not allowed")

    def do_POST(self):
        body = self._read_json_body()
        if body is None:
            self._send_json(400, {"error": "Invalid JSON body"})
            return
        model_for_log = body.get("model", "")
        mt = body.get("max_tokens")
        tp = body.get("top_p")
        tm = body.get("temperature")
        tk = body.get("top_k")
        print(
            f"[POST] {self.path} model={model_for_log} max_tokens={mt} temp={tm} top_p={tp} top_k={tk}",
            flush=True,
        )

        if self.path == "/api/local/health":
            model_id = _normalize_model_id(body.get("model"))
            preload = bool(body.get("preload", False))
            try:
                payload = {"ok": True, "backend": "transformers", "model": model_id}
                if preload:
                    runtime = _load_runtime(model_id)
                    payload["loaded"] = True
                    payload["device"] = runtime["device"]
                    payload["loaded_in_s"] = runtime["loaded_in_s"]
                else:
                    with CACHE_LOCK:
                        loaded = model_id in MODEL_CACHE
                    payload["loaded"] = loaded
                self._send_json(200, payload)
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        if self.path == "/api/local/completions":
            model_id = _normalize_model_id(body.get("model"))
            prompt = body.get("prompt")
            if not prompt:
                self._send_json(400, {"error": "prompt is required"})
                return

            try:
                runtime = _load_runtime(model_id)
                gen = self._generate(
                    runtime=runtime,
                    prompt=prompt,
                    max_tokens=body.get("max_tokens", 120),
                    temperature=body.get("temperature", 0.9),
                    top_p=body.get("top_p", 1.0),
                    top_k=body.get("top_k"),
                    beam_size=body.get("beam_size", 1),
                )
                payload = {
                    "id": f"local-{int(time.time() * 1000)}",
                    "object": "text_completion",
                    "model": model_id,
                    "choices": [{"index": 0, "text": gen["text"], "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": gen["prompt_tokens"],
                        "completion_tokens": gen["completion_tokens"],
                        "total_tokens": gen["prompt_tokens"] + gen["completion_tokens"],
                    },
                    "performance": {
                        "elapsed_ms": gen["elapsed_ms"],
                        "tokens_per_second": gen["tokens_per_second"],
                    },
                }
                self._send_json(200, payload)
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        if self.path == "/api/local/completions_stream":
            model_id = _normalize_model_id(body.get("model"))
            prompt = body.get("prompt")
            if not prompt:
                self._send_json(400, {"error": "prompt is required"})
                return
            if TextIteratorStreamer is None:
                self._send_json(500, {"error": "Streaming not available in current transformers install"})
                return
            self._start_sse()
            self._send_sse({"type": "status", "message": "loading_runtime"})
            try:
                runtime = _load_runtime(model_id)
                self._send_sse(
                    {
                        "type": "status",
                        "message": "runtime_ready",
                        "device": runtime["device"],
                    }
                )
                for event in self._stream_generate(
                    runtime=runtime,
                    prompt=prompt,
                    max_tokens=body.get("max_tokens", 120),
                    temperature=body.get("temperature", 0.9),
                    top_p=body.get("top_p", 1.0),
                    top_k=body.get("top_k"),
                    beam_size=body.get("beam_size", 1),
                ):
                    self._send_sse(event)
                self.close_connection = True
            except Exception as exc:
                try:
                    self._send_sse({"type": "error", "error": str(exc)})
                except Exception:
                    pass
            return

        if self.path == "/api/local/next-token-distribution":
            model_id = _normalize_model_id(body.get("model"))
            prompt = body.get("prompt")
            top_n = body.get("top_n", 40)
            top_k = body.get("top_k", 20)
            top_p = body.get("top_p", 0.9)
            if not prompt:
                self._send_json(400, {"error": "prompt is required"})
                return
            try:
                runtime = _load_runtime(model_id)
                dist = self._next_token_distribution(
                    runtime=runtime,
                    prompt=prompt,
                    top_n=top_n,
                    top_k=top_k,
                    top_p=top_p,
                )
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "backend": "transformers",
                        "mode": "exact_logits",
                        "model": model_id,
                        "device": runtime["device"],
                        "tokens": dist["tokens"],
                        "top_k_rank": dist["top_k_rank"],
                        "top_p_rank": dist["top_p_rank"],
                        "vocab_size": dist["vocab_size"],
                    },
                )
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        self.send_error(404, "Unknown API route")

    def log_message(self, format, *args):
        pass


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = ThreadingHTTPServer((HOST, PORT), DecodingLabAPIHandler)
    print(f"Serving Decoding Lab on http://{HOST}:{PORT}", flush=True)
    print("Using local Transformers backend via /api/local/*", flush=True)
    if torch is None:
        print(f"Torch unavailable: {IMPORT_ERROR}", flush=True)
    else:
        mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(
            f"Device support: cuda={torch.cuda.is_available()} mps_built={mps_built} mps_available={mps_avail}",
            flush=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)


if __name__ == "__main__":
    main()

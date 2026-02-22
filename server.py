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
TOP_K_MAX = 320
ALLOWED_DECODE_STRATEGIES = {"greedy", "beam", "topk", "topp", "sample"}
ALLOWED_SAMPLING_STRATEGIES = {"greedy", "topk", "topp", "sample"}


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
        "vocab_ids_cache": {},
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

    def _parse_decode_request(
        self,
        body,
        default_max_tokens=120,
        allowed_strategies=None,
    ):
        allowed = allowed_strategies or ALLOWED_DECODE_STRATEGIES
        strategy = body.get("strategy")
        strategy = strategy.strip().lower() if isinstance(strategy, str) else ""
        if strategy not in allowed:
            supported = ", ".join(sorted(allowed))
            raise ValueError(f"strategy is required and must be one of: {supported}")
        return {
            "max_tokens": body.get("max_tokens", default_max_tokens),
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "top_k": body.get("top_k"),
            "beam_size": body.get("beam_size", 1),
            "strategy": strategy,
        }

    def _normalize_decoding_params(self, strategy, temperature, top_p, top_k, beam_size):
        strategy = (strategy or "").strip().lower()
        if strategy not in ALLOWED_DECODE_STRATEGIES:
            raise ValueError("Unsupported strategy")
        normalized = {"strategy": strategy, "beam_size": max(1, int(beam_size or 1))}

        if strategy == "greedy":
            normalized["temperature"] = 0.0
            normalized["top_p"] = 1.0
            normalized["top_k"] = 0
            normalized["beam_size"] = 1
        elif strategy == "beam":
            normalized["temperature"] = 0.0
            normalized["top_p"] = 1.0
            normalized["top_k"] = 0
            normalized["beam_size"] = max(1, normalized["beam_size"])
        elif strategy == "topk":
            if temperature is None:
                raise ValueError("temperature is required for top-k strategy")
            if top_k is None:
                raise ValueError("top_k is required for top-k strategy")
            normalized["temperature"] = float(temperature)
            if normalized["temperature"] <= 0.0:
                raise ValueError("temperature must be > 0 for top-k strategy")
            normalized["top_k"] = int(top_k)
            if normalized["top_k"] < 1:
                raise ValueError("top_k must be >= 1 for top-k strategy")
            if normalized["top_k"] > TOP_K_MAX:
                raise ValueError(f"top_k must be <= {TOP_K_MAX} for top-k strategy")
            normalized["top_p"] = 1.0
            normalized["beam_size"] = 1
        elif strategy == "topp":
            if temperature is None:
                raise ValueError("temperature is required for top-p strategy")
            if top_p is None:
                raise ValueError("top_p is required for top-p strategy")
            normalized["temperature"] = float(temperature)
            if normalized["temperature"] <= 0.0:
                raise ValueError("temperature must be > 0 for top-p strategy")
            normalized["top_p"] = float(top_p)
            if not (0.1 <= normalized["top_p"] <= 1.0):
                raise ValueError("top_p must be in [0.1, 1] for top-p strategy")
            normalized["top_k"] = 0
            normalized["beam_size"] = 1
        else:  # sample
            if temperature is None:
                raise ValueError("temperature is required for sample strategy")
            normalized["temperature"] = float(temperature)
            normalized["top_p"] = 1.0
            normalized["top_k"] = 0
            normalized["beam_size"] = 1

        return normalized

    def _build_hf_generate_kwargs(self, tokenizer, max_tokens, decoding, streamer=None):
        kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if streamer is not None:
            kwargs["streamer"] = streamer

        if decoding["strategy"] == "beam":
            kwargs.update(
                {
                    "do_sample": False,
                    "num_beams": decoding["beam_size"],
                    "early_stopping": True,
                }
            )
            return kwargs

        if decoding["strategy"] in {"topk", "topp", "sample"}:
            # In the limit temp->0, pure sampling should match greedy behavior.
            if decoding["strategy"] == "sample" and decoding["temperature"] <= 0.0001:
                kwargs["do_sample"] = False
                return kwargs
            kwargs.update(
                {
                    "do_sample": True,
                    "temperature": decoding["temperature"],
                    # Explicitly set both to avoid GenerationConfig defaults (e.g., top_k=50).
                    "top_k": decoding["top_k"],
                    "top_p": decoding["top_p"],
                }
            )
            return kwargs

        kwargs["do_sample"] = False
        return kwargs

    def _generate(self, runtime, prompt, max_tokens, temperature, top_p, top_k, beam_size=1, strategy=None):
        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        device = runtime["device"]

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        max_tokens = max(1, min(int(max_tokens), 200))
        decoding = self._normalize_decoding_params(
            strategy=strategy,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            beam_size=beam_size,
        )
        gen_kwargs = self._build_hf_generate_kwargs(tokenizer, max_tokens, decoding)

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

    def _stream_generate(self, runtime, prompt, max_tokens, temperature, top_p, top_k, beam_size=1, strategy=None):
        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        device = runtime["device"]

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        max_tokens = max(1, min(int(max_tokens), 200))
        decoding = self._normalize_decoding_params(
            strategy=strategy,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            beam_size=beam_size,
        )

        if decoding["strategy"] == "beam":
            # HF streamer + beam search may not emit incremental deltas reliably.
            gen_kwargs = self._build_hf_generate_kwargs(tokenizer, max_tokens, decoding)
            gen_kwargs["input_ids"] = input_ids
            gen_kwargs["attention_mask"] = attention_mask

            started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**gen_kwargs)
            elapsed_s = max(time.perf_counter() - started, 1e-6)
            new_ids = output[0][input_ids.shape[1] :]
            completion_text = tokenizer.decode(new_ids, skip_special_tokens=True)
            completion_tokens = int(new_ids.shape[0])
            prompt_tokens = int(input_ids.shape[1])

            if completion_text:
                yield {"type": "delta", "text": completion_text}
            yield {
                "type": "done",
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "performance": {
                    "elapsed_ms": int(round(elapsed_s * 1000)),
                    "tokens_per_second": float(completion_tokens / elapsed_s) if completion_tokens > 0 else 0.0,
                },
            }
            return

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        gen_kwargs = self._build_hf_generate_kwargs(tokenizer, max_tokens, decoding, streamer=streamer)
        gen_kwargs["input_ids"] = input_ids
        gen_kwargs["attention_mask"] = attention_mask

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
        top_p = min(max(top_p, 0.1), 1.0)

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

    def _token_probability_trace(self, runtime, prompt, completion_text, max_points=200):
        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        device = runtime["device"]

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
        if not completion_ids:
            return []

        if not prompt_ids:
            prompt_ids = [int(tokenizer.eos_token_id or 0)]

        completion_ids = completion_ids[: max(1, min(int(max_points), 400))]
        seq_ids = prompt_ids + completion_ids
        input_ids = torch.tensor([seq_ids], dtype=torch.long, device=device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits[0]

        start = len(prompt_ids) - 1
        target = torch.tensor(completion_ids, dtype=torch.long, device=device)
        logits_slice = logits[start : start + len(completion_ids), :]
        probs = torch.softmax(logits_slice, dim=-1)
        token_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1).detach().to("cpu")

        trace = []
        for i, tok_id in enumerate(completion_ids):
            p = float(token_probs[i].item())
            trace.append(
                {
                    "timestep": i + 1,
                    "token": tokenizer.decode([int(tok_id)], clean_up_tokenization_spaces=False),
                    "prob": max(p, 1e-12),
                    "logprob": float(math.log(max(p, 1e-12))),
                }
            )
        return trace

    def _sample_from_logits(self, logits, strategy, temperature, top_p, top_k):
        strategy = (strategy or "").strip().lower()
        if strategy not in ALLOWED_SAMPLING_STRATEGIES:
            raise ValueError("Unsupported strategy")
        if strategy == "greedy":
            return int(torch.argmax(logits).item())
        if strategy == "sample" and float(temperature) <= 0.0001:
            return int(torch.argmax(logits).item())

        temp = max(float(temperature), 0.05)
        probs = torch.softmax(logits / temp, dim=-1)
        vocab_size = int(probs.shape[0])

        if strategy == "sample":
            return int(torch.multinomial(probs, 1).item())

        if strategy == "topk":
            if top_k is None:
                raise ValueError("top_k is required for top-k strategy")
            k = max(1, min(int(top_k), vocab_size))
            topk_probs, topk_idx = torch.topk(probs, k)
            denom = torch.sum(topk_probs)
            if float(denom.item()) <= 0:
                return int(topk_idx[0].item())
            topk_probs = topk_probs / denom
            sample_idx = int(torch.multinomial(topk_probs, 1).item())
            return int(topk_idx[sample_idx].item())

        if top_p is None:
            raise ValueError("top_p is required for top-p strategy")
        p = min(max(float(top_p), 0.1), 1.0)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative > p
        # Keep the first token above threshold so nucleus mass is >= p.
        remove_mask[1:] = remove_mask[:-1].clone()
        remove_mask[0] = False
        filtered = sorted_probs.masked_fill(remove_mask, 0.0)
        denom = torch.sum(filtered)
        if float(denom.item()) <= 0:
            return int(sorted_idx[0].item())
        filtered = filtered / denom
        sample_idx = int(torch.multinomial(filtered, 1).item())
        return int(sorted_idx[sample_idx].item())

    def _kgw_seed(self, prev_token_id, seeding_key):
        prev_tok = int(prev_token_id) & 0xFFFFFFFF
        key = int(seeding_key) & 0xFFFFFFFF
        return ((prev_tok * 2654435761) ^ (key * 2246822519)) & 0xFFFFFFFF

    def _kgw_green_mask(self, runtime, vocab_size, seed, gamma, device):
        key = f"{device}:{vocab_size}"
        ids = runtime["vocab_ids_cache"].get(key)
        if ids is None or int(ids.shape[0]) != int(vocab_size):
            ids = torch.arange(vocab_size, dtype=torch.int64, device=device)
            runtime["vocab_ids_cache"][key] = ids

        seed_t = torch.full_like(ids, int(seed), dtype=torch.int64)
        x = torch.bitwise_xor(ids, seed_t)
        x = torch.bitwise_and(x * 0x9E3779B1, 0xFFFFFFFF)
        x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
        x = torch.bitwise_and(x * 0x85EBCA77, 0xFFFFFFFF)
        x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 13))
        threshold = int(min(max(float(gamma), 0.01), 0.99) * 4294967295.0)
        return x <= threshold

    def _generate_kgw_watermarked(
        self,
        runtime,
        prompt,
        max_tokens,
        strategy,
        temperature,
        top_p,
        top_k,
        gamma,
        delta,
        seeding_key,
    ):
        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        device = runtime["device"]

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        else:
            attention_mask = torch.ones_like(input_ids, device=device)

        max_tokens = max(1, min(int(max_tokens), 200))
        gamma = min(max(float(gamma), 0.01), 0.99)
        delta = float(delta)

        generated = []
        token_meta = []
        started = time.perf_counter()
        eos_id = tokenizer.eos_token_id

        with torch.inference_mode():
            for _ in range(max_tokens):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1, :]
                prev_token = int(input_ids[0, -1].item())
                seed = self._kgw_seed(prev_token, seeding_key)
                green_mask = self._kgw_green_mask(runtime, int(logits.shape[0]), seed, gamma, device)
                adjusted = logits + green_mask.to(logits.dtype) * delta

                next_token = self._sample_from_logits(
                    logits=adjusted,
                    strategy=strategy,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                generated.append(next_token)
                is_green = bool(green_mask[next_token].item())
                tok_text = tokenizer.decode([next_token], clean_up_tokenization_spaces=False)
                token_meta.append({"id": int(next_token), "text": tok_text, "green": is_green})

                next_tok = torch.tensor([[next_token]], dtype=input_ids.dtype, device=device)
                input_ids = torch.cat([input_ids, next_tok], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1)
                if eos_id is not None and next_token == int(eos_id):
                    break

        elapsed_s = max(time.perf_counter() - started, 1e-6)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        completion_tokens = len(generated)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        return {
            "text": text,
            "tokens": token_meta,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_ms": int(round(elapsed_s * 1000)),
            "tokens_per_second": float(completion_tokens / elapsed_s) if completion_tokens > 0 else 0.0,
        }

    def _stream_kgw_and_plain(
        self,
        runtime,
        prompt,
        max_tokens,
        strategy,
        temperature,
        top_p,
        top_k,
        gamma,
        delta,
        seeding_key,
    ):
        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        device = runtime["device"]

        max_tokens = max(1, min(int(max_tokens), 200))
        gamma = min(max(float(gamma), 0.01), 0.99)
        delta = float(delta)
        eos_id = tokenizer.eos_token_id
        inputs = tokenizer(prompt, return_tensors="pt")
        base_ids = inputs["input_ids"].to(device)
        base_mask = inputs.get("attention_mask")
        if base_mask is not None:
            base_mask = base_mask.to(device)
        else:
            base_mask = torch.ones_like(base_ids, device=device)

        def run_one_stream(apply_watermark):
            input_ids = base_ids
            attention_mask = base_mask
            prev_tok = int(base_ids[0, -1].item())
            past_key_values = None
            gen_count = 0
            gen_green = 0

            with torch.inference_mode():
                for _ in range(max_tokens):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    logits = out.logits[0, -1, :]
                    past_key_values = out.past_key_values

                    seed = self._kgw_seed(prev_tok, seeding_key)
                    green_mask = self._kgw_green_mask(runtime, int(logits.shape[0]), seed, gamma, device)
                    decode_logits = logits + green_mask.to(logits.dtype) * delta if apply_watermark else logits
                    next_tok = self._sample_from_logits(
                        logits=decode_logits,
                        strategy=strategy,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    is_green = bool(green_mask[next_tok].item())
                    gen_green += int(is_green)
                    gen_count += 1
                    tok_text = tokenizer.decode([next_tok], clean_up_tokenization_spaces=False)
                    yield next_tok, tok_text, is_green, gen_count, gen_green

                    prev_tok = int(next_tok)
                    input_ids = torch.tensor([[next_tok]], dtype=base_ids.dtype, device=device)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
                        dim=1,
                    )
                    if eos_id is not None and next_tok == int(eos_id):
                        break

        started = time.perf_counter()
        wm_count = 0
        wm_green = 0
        plain_count = 0
        plain_green = 0

        yield {"type": "status", "message": "streaming_watermarked"}
        for _, tok_text, is_green, gen_count, gen_green in run_one_stream(True):
            wm_count = gen_count
            wm_green = gen_green
            yield {"type": "delta_wm", "text": tok_text, "green": is_green}

        yield {"type": "status", "message": "streaming_plain"}
        for _, tok_text, is_green, gen_count, gen_green in run_one_stream(False):
            plain_count = gen_count
            plain_green = gen_green
            yield {"type": "delta_plain", "text": tok_text, "green": is_green}

        elapsed_s = max(time.perf_counter() - started, 1e-6)
        yield {
            "type": "done",
            "wm": {
                "tokens": wm_count,
                "green_tokens": wm_green,
                "tokens_per_second": float(wm_count / elapsed_s) if wm_count > 0 else 0.0,
            },
            "plain": {
                "tokens": plain_count,
                "green_tokens": plain_green,
                "tokens_per_second": float(plain_count / elapsed_s) if plain_count > 0 else 0.0,
            },
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
        st = body.get("strategy")
        print(
            f"[POST] {self.path} model={model_for_log} strategy={st} max_tokens={mt} temp={tm} top_p={tp} top_k={tk}",
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
                params = self._parse_decode_request(body)
                gen = self._generate(
                    runtime=runtime,
                    prompt=prompt,
                    **params,
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
                params = self._parse_decode_request(body)
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
                    **params,
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

        if self.path == "/api/local/token-probability-trace":
            model_id = _normalize_model_id(body.get("model"))
            prompt = body.get("prompt")
            completion = body.get("completion")
            if prompt is None or completion is None:
                self._send_json(400, {"error": "prompt and completion are required"})
                return
            try:
                runtime = _load_runtime(model_id)
                trace = self._token_probability_trace(
                    runtime=runtime,
                    prompt=prompt,
                    completion_text=completion,
                    max_points=body.get("max_points", 200),
                )
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "model": model_id,
                        "device": runtime["device"],
                        "trace": trace,
                    },
                )
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        if self.path == "/api/local/watermark/completions_stream":
            model_id = _normalize_model_id(body.get("model"))
            prompt = body.get("prompt")
            if not prompt:
                self._send_json(400, {"error": "prompt is required"})
                return
            self._start_sse()
            self._send_sse({"type": "status", "message": "loading_runtime"})
            try:
                params = self._parse_decode_request(
                    body,
                    default_max_tokens=80,
                    allowed_strategies=ALLOWED_SAMPLING_STRATEGIES,
                )
                decoded = self._normalize_decoding_params(
                    strategy=params["strategy"],
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    top_k=params["top_k"],
                    beam_size=1,
                )
                runtime = _load_runtime(model_id)
                self._send_sse({"type": "status", "message": "runtime_ready", "device": runtime["device"]})
                for event in self._stream_kgw_and_plain(
                    runtime=runtime,
                    prompt=prompt,
                    max_tokens=params["max_tokens"],
                    strategy=decoded["strategy"],
                    temperature=decoded["temperature"],
                    top_p=decoded["top_p"],
                    top_k=decoded["top_k"],
                    gamma=body.get("gamma", 0.25),
                    delta=body.get("delta", 2.0),
                    seeding_key=body.get("seeding_key", 15485863),
                ):
                    self._send_sse(event)
                self.close_connection = True
            except Exception as exc:
                try:
                    self._send_sse({"type": "error", "error": str(exc)})
                except Exception:
                    pass
            return

        if self.path == "/api/local/watermark/completions":
            model_id = _normalize_model_id(body.get("model"))
            prompt = body.get("prompt")
            if not prompt:
                self._send_json(400, {"error": "prompt is required"})
                return
            try:
                params = self._parse_decode_request(
                    body,
                    default_max_tokens=80,
                    allowed_strategies=ALLOWED_SAMPLING_STRATEGIES,
                )
                decoded = self._normalize_decoding_params(
                    strategy=params["strategy"],
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    top_k=params["top_k"],
                    beam_size=1,
                )
                runtime = _load_runtime(model_id)
                gen = self._generate_kgw_watermarked(
                    runtime=runtime,
                    prompt=prompt,
                    max_tokens=params["max_tokens"],
                    strategy=decoded["strategy"],
                    temperature=decoded["temperature"],
                    top_p=decoded["top_p"],
                    top_k=decoded["top_k"],
                    gamma=body.get("gamma", 0.25),
                    delta=body.get("delta", 2.0),
                    seeding_key=body.get("seeding_key", 15485863),
                )
                self._send_json(
                    200,
                    {
                        "id": f"wm-{int(time.time() * 1000)}",
                        "object": "watermarked_completion",
                        "model": model_id,
                        "choices": [{"index": 0, "text": gen["text"], "finish_reason": "stop"}],
                        "tokens": gen["tokens"],
                        "usage": {
                            "prompt_tokens": gen["prompt_tokens"],
                            "completion_tokens": gen["completion_tokens"],
                            "total_tokens": gen["prompt_tokens"] + gen["completion_tokens"],
                        },
                        "performance": {
                            "elapsed_ms": gen["elapsed_ms"],
                            "tokens_per_second": gen["tokens_per_second"],
                        },
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

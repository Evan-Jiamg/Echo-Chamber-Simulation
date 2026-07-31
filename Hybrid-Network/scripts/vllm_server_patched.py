#!/usr/bin/env python3
"""
vLLM server wrapper that patches guided decoding to never fall back to xgrammar.
xgrammar.so is not compiled in this environment — redirect all xgrammar paths to
outlines or lm-format-enforcer instead.

Usage (same as -m vllm.entrypoints.openai.api_server):
  python scripts/vllm_server_patched.py --model ... --guided-decoding-backend lm-format-enforcer ...
"""

# ── Patch must happen before vLLM imports guided_decoding internals ──────────
import vllm.model_executor.guided_decoding as _gd

_orig_fallback = _gd.maybe_backend_fallback


def _patched_fallback(guided_params):
    result = _orig_fallback(guided_params)
    if getattr(result, "backend", None) == "xgrammar":
        result.backend = "outlines"
    return result


_gd.maybe_backend_fallback = _patched_fallback

# Also patch the dispatch in both async and sync functions so xgrammar branch
# is never reached even if something sets backend="xgrammar" externally.
_orig_async = _gd.get_guided_decoding_logits_processor
_orig_sync  = _gd.get_local_guided_decoding_logits_processor


async def _patched_async(guided_params, tokenizer, model_config):
    if getattr(guided_params, "backend", None) == "xgrammar":
        guided_params.backend = "outlines"
    return await _orig_async(guided_params, tokenizer, model_config)


def _patched_sync(guided_params, tokenizer, model_config):
    if getattr(guided_params, "backend", None) == "xgrammar":
        guided_params.backend = "outlines"
    return _orig_sync(guided_params, tokenizer, model_config)


_gd.get_guided_decoding_logits_processor       = _patched_async
_gd.get_local_guided_decoding_logits_processor = _patched_sync

# ── Now hand off to the real vLLM entrypoint ────────────────────────────────
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.api_server import validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser
import uvloop

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args   = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))

# Local Model Setup

Chronicle is now local-first for browser inference:

- the browser runtime loads models from `/models`
- `env.allowRemoteModels` is disabled in the frontend
- end users are not asked for Hugging Face authentication when the bundle exists locally

## Preferred browser bundle

For adaptive Gemma 3n slicing, place the ONNX bundle here:

```text
models/
  onnx-community/
    gemma-3n-E2B-it-ONNX/
```

Chronicle will also detect these fallback locations:

```text
models/
  gemma-3n-E2B-it-ONNX/
  gemma-3-it/
```

## Download the public Gemma 3n bundle

```bash
gemma-env/bin/python download_browser_model.py
```

The downloader targets `onnx-community/gemma-3n-E2B-it-ONNX` and does not require a Hugging Face token.

## Backend fallback

The native Python runtime is still separate from the browser bundle:

- Apple Silicon fallback expects `models/mlx-community/gemma-3-4b-it-4bit`
- portable Transformers fallback expects `models/google/gemma-2-2b-it`

If those fallback directories are missing, Chronicle can still run through the browser-local path as long as a browser bundle exists under `/models`.

## Optional overrides

- `NEWSLETTER_AGENT_MODEL_ROOT`
- `NEWSLETTER_AGENT_BROWSER_MODEL_ID`
- `NEWSLETTER_AGENT_MODEL`
- `NEWSLETTER_AGENT_MODEL_TRANSFORMERS`
- `NEWSLETTER_AGENT_MODEL_LOW`
- `NEWSLETTER_AGENT_MODEL_MEDIUM`
- `NEWSLETTER_AGENT_MODEL_HIGH`
- `NEWSLETTER_AGENT_MODEL_SLICE_12_5`
- `NEWSLETTER_AGENT_MODEL_SLICE_25`
- `NEWSLETTER_AGENT_MODEL_SLICE_50`
- `NEWSLETTER_AGENT_MODEL_SLICE_75`
- `NEWSLETTER_AGENT_MODEL_SLICE_100`

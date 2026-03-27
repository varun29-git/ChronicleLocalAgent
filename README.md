# Chronicle

Chronicle is a local-first newsletter studio. The backend handles research and saving outputs, while the browser loads a model bundle from this repo’s `/models` directory and runs generation on the user’s device.

## Quick start

```bash
gemma-env/bin/python chronicle_server.py --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`.

## Local model behavior

- Chronicle never needs a Hugging Face token for the browser path when the model bundle is already present under `/models`.
- The current browser runtime auto-detects a local bundle in this order:
  - `models/onnx-community/gemma-3n-E2B-it-ONNX`
  - `models/gemma-3n-E2B-it-ONNX`
  - `models/gemma-3-it`
- If the detected bundle is Gemma 3n, Chronicle enables device-adaptive slice selection in the browser.
- If the detected bundle is not Gemma 3n, Chronicle still runs locally but without MatFormer slice selection.

## Download Gemma 3n without auth prompts

To pull the public Gemma 3n ONNX browser bundle into the expected local path:

```bash
gemma-env/bin/python download_browser_model.py
```

That downloader uses `huggingface_hub` against the public `onnx-community/gemma-3n-E2B-it-ONNX` repo and does not require a Hugging Face API key.

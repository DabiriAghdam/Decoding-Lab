# Decoding Dashboard Explorer (CPSC 532B)

Interactive local dashboard for exploring decoding strategies from *The Curious Case of Neural Text Degeneration* (Holtzman et al., 2019), plus side-by-side comparisons and preference rounds.

## What this project includes
- **Distribution Explorer**: next-token probability chart with top-k and top-p overlays.
- **Side-by-Side Decode**: compare two decoding strategies on the same prompt.
- **Human Preference Arena**: blind pairwise rounds to see which decoding style is preferred.

## Requirements
- Python 3.9+
- A local environment with:
  - `torch`
  - `transformers`
  - other dependencies in `requirements.txt`

## Setup
If using your existing conda env named `presentation`:

```bash
conda activate presentation
pip install -r requirements.txt
```

If `torch`/`transformers` are missing, install them in the same env before running.

## Run
From the project root:

```bash
python3 server.py
```

Then open:

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

Optional host/port override:

```bash
HOST=127.0.0.1 PORT=8017 python3 server.py
```

## Notes
- The frontend auto-preloads the default model (`gpt2-large`) on page load.
- Backend endpoints are served under `/api/local/*`.
- Generation and distribution are computed locally with Hugging Face Transformers.

## Troubleshooting
- **Address already in use**: run on a different port, e.g. `PORT=8017 python3 server.py`.
- **Slow first request**: first load includes model initialization; later requests are faster.
- **Model load errors**: verify model name exists in the dropdown and dependencies are installed in the active env.

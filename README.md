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

## System Requirements
- **OS**: macOS, Linux, or Windows 10/11
- **Python**: 3.9 or newer
- **CPU**: modern 64-bit CPU (Apple Silicon, Intel, or AMD)
- **RAM**:
  - Minimum: 8 GB (small models, slower experience)
  - Recommended: 16 GB+ (for smoother use with `gpt2-large`)
- **Storage**:
  - At least 6 GB free for dependencies + model cache
  - Recommended 10 GB+ if trying multiple models
- **GPU (optional)**:
  - macOS: Apple Silicon GPU via MPS (automatic if available)
  - Linux/Windows: NVIDIA GPU with CUDA-capable PyTorch build
- **Network**:
  - Internet is needed the first time a model is downloaded from Hugging Face
  - After download, models are loaded from local cache

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

## Windows and Linux
Yes, the project runs on both.

- **Linux**:
  ```bash
  python3 server.py
  ```
- **Windows (PowerShell)**:
  ```powershell
  py server.py
  ```
- **Windows (host/port override)**:
  ```powershell
  $env:HOST="127.0.0.1"; $env:PORT="8017"; py server.py
  ```

Notes:
- `mps` acceleration is Mac-only. On Linux/Windows, this will use CUDA (if available) or CPU.
- The first load is always slower because the model is loaded into memory.

## Notes
- The frontend auto-preloads the default model (`gpt2-large`) on page load.
- Backend endpoints are served under `/api/local/*`.
- Generation and distribution are computed locally with Hugging Face Transformers.

## Free Hosting Options
You can host it for free, but with an important caveat: free tiers are usually **CPU-only**, so large local models will be slow.

- **Hugging Face Spaces (Free CPU Basic)**:
  - Good for demos and sharing.
  - Best path for this project if you keep models small.
- **Render (Free web service tier)**:
  - Can host Python services, but free instances can sleep when idle.
- **Cloudflare Tunnel (free) + your local machine**:
  - Quick way to share your running local app publicly during presentation.
  - Your own machine still does all model inference.

## Troubleshooting
- **Address already in use**: run on a different port, e.g. `PORT=8017 python3 server.py`.
- **Slow first request**: first load includes model initialization; later requests are faster.
- **Model load errors**: verify model name exists in the dropdown and dependencies are installed in the active env.

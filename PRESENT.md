# Decoding Lab Runbook

## 1) Use local conda env
- Ensure env exists: `presentation`
- Ensure packages are installed there: `torch`, `transformers`

## 2) Start the presentation app (local transformers backend + static)
```bash
cd /Users/ahda/Desktop/Presentation
conda run -n presentation python3 server.py
```

If port `8000` is busy:
```bash
PORT=8010 conda run -n presentation python3 server.py
```

Open:
- `http://127.0.0.1:8000` (or `:8010`)

## 3) In-app settings
- Model: `gpt2-large`
- Click `Test Local Backend` (loads model + shows device)
- Set `Max New Tokens` for how long each sample should be

## 4) Live talk flow (fast)
1. `Live Decoding DJ`
   - Pick preset: `Paper-style analysis`
   - Generate with `Greedy` and point out repetition.
   - Switch to `Top-p` (`p=0.92`) and generate again.
2. `Human Judge Arena`
   - Click `Generate Round Options`
   - Ask audience to vote left/right.
   - Click `Reveal Strategies`.
3. `Probability Tail Theater`
   - Click `Fetch Real Distribution`.
   - Move `Top-k` and `Top-p` overlay sliders.
   - Explain fixed-count truncation vs dynamic-mass truncation on exact logits from the real local model.

## 5) If output looks weird (common with gpt2-large)
- Use the `Paper-style analysis` preset.
- Lower `temperature` to `0.8`.
- Keep `top-p` between `0.88` and `0.94`.

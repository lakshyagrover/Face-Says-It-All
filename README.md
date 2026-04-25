# Face Says it All — Emotion-Based AI DJ

An AI-powered web app that detects your facial emotion via webcam and recommends a matching Spotify playlist. The emotion model runs as a Flask backend.

## Architecture

```
Browser (index.html)
   └──POST /predict──► Flask (predict.py)
                           └── TensorFlow model (auto-downloaded from HuggingFace)
```

Everything is served from **one Render web service** — Flask serves `index.html` at `/` and the ML API at `/predict`.

---

## Deploy on Render (one-click)

1. **Push this repo to GitHub** (the `.h5` model file is intentionally in `.gitignore` — it is downloaded automatically at startup).
2. Go to [render.com](https://render.com) → **New → Web Service**.
3. Connect your GitHub repo.
4. Render will auto-detect `render.yaml`. Hit **Deploy**.
5. First deploy takes ~3–5 min (TF install + model download ~90 MB). Subsequent deploys are faster.
6. Open the Render URL — the frontend loads and calls `/predict` on the same origin.

### Manual settings (if not using render.yaml)

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn predict:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1` |
| Plan | Free (or paid for faster startup) |

---

## Run locally

```bash
pip install -r requirements.txt
python predict.py          # starts on http://localhost:5001
# Open index.html in your browser (file:// or a local server)
```

---

## Notes

- The model (`final_model.h5`) is downloaded automatically from HuggingFace on first boot. It is **not** committed to git.
- `opencv-python-headless` is used instead of `opencv-python` to avoid display-library errors on headless servers.
- Render free tier sleeps after 15 min of inactivity — first request after sleep may take ~30 s.

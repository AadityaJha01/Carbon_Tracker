## Project explanation and review prep

This document explains what the project is, what I changed to make the web demo fully functional, how to run it, and answers to likely reviewer questions so you're prepared for a viva or demo.

---

## 1) One-sentence summary

This is a Carbon-Aware ML Training Dashboard: a Flask web UI to launch and monitor model training jobs, measure energy/CO₂, maintain an efficiency leaderboard, and provide model recommendations.

## 2) What happened (issues found) and what I fixed

Problems observed before this work:
- The web UI sometimes appeared to ``vanish`` or not show training progress when a training job was started. Heavy ML imports and real training caused the Flask process to be unstable in the dev server.
- A JavaScript bug in the recommendations tab used `event.target` without an event object and caused runtime errors in the browser.
- Leaderboard had either no data or a single row with zeros; UI looked empty for demonstrations.

What I implemented to make the project demo-ready:
- Demo trainer: added `src/core/demo_trainer.py` — a fast, simulated trainer that mimics epochs and metrics and writes a leaderboard entry. This enables reliable demos without heavy GPU compute.
- Demo wiring: added a "Demo Mode" checkbox to the training form and changed the server to use `DemoTrainer` when `demo=true`.
- Lazy imports: the real `Trainer` (PyTorch heavy) is now lazy-imported only when a non-demo job starts. This keeps the Flask web server responsive at startup.
- Fixed frontend bugs: corrected recommendation tab click handlers and the tab switching function in `web/static/script.js`.
- Added small debugging logfile per-job so progress updates are visible if you need to inspect background thread activity: `results/job_<jobid>.log`.
- Helper to populate leaderboard: `scripts/populate_leaderboard.py` — create sample runs for UI display.

Files added/modified (high level):
- Added: `src/core/demo_trainer.py`
- Added: `scripts/populate_leaderboard.py`
- Modified: `web/app.py`, `web/templates/index.html`, `web/static/script.js`

These changes are intentionally low-risk: the demo trainer is used only when requested and the production training path (real Trainer) is untouched except for lazy import.

## 3) How the system is organized (components)

- Web UI (Flask): `web/app.py`, `web/run.py`, templates in `web/templates/`, static JS/CSS in `web/static/`.
- Trainer (real): `src/core/trainer.py` — uses PyTorch, dataset loaders, logger, carbon tracker, and writes leaderboard entries.
- Demo trainer: `src/core/demo_trainer.py` — fast, deterministic-ish simulated training (safe for demos).
- Leaderboard: `leaderboard.py` — manages `results/leaderboard.csv`, computes efficiency metrics, and produces plots.
- Recommender: `recommender.py` — simple rules-based recommender that selects runs from the leaderboard based on accuracy, CO₂ budget, or time budget.

## 4) How to run the demo locally (Windows PowerShell)

1) Install dependencies (if not already):

```powershell
cd "c:\Users\aadit\Desktop\Collage Material\Major Project\Project"
pip install -r requirements.txt
```

2) (Optional) Add sample leaderboard rows for a better-looking demo:

```powershell
python scripts\populate_leaderboard.py
```

3) Start the web app (keep this terminal open to see logs):

```powershell
cd web
python run.py
```

4) Open `http://localhost:5000` in your browser.

5) In the Train tab: ensure the "Demo Mode (fast simulated training)" checkbox is selected (it is checked by default). Click "Start Training". The Dashboard tab will show the job progress. The run completes quickly and the leaderboard updates.

6) Debug logs (if needed) are written to `results/job_<jobid>.log` and the leaderboard is at `results/leaderboard.csv`.

## 5) How to run a real training job

Important: running real training uses PyTorch and datasets and may require GPU and time.

1) Ensure PyTorch and torchvision matching your CUDA are installed (see `requirements.txt`), or adjust to CPU-only versions if you don't have a GPU.
2) Uncheck the "Demo Mode" checkbox in the Train tab so the server will lazy-import and use the real `Trainer`.
3) Start training from the UI. Monitor logs and resource use. Training outputs a `best_model.pt` and appends a row to `results/leaderboard.csv` when finished.

Notes on real training:
- The trainer downloads CIFAR-10 if not already in `data/`.
- Training can be slow — for presentations, prefer Demo Mode.

## 6) Key review questions (with suggested answers)

Q: What is the purpose of this project?
- A: To help ML practitioners understand the trade-offs between model accuracy and environmental cost (energy and CO₂) by providing an interface to run experiments, measure energy, and compare models with efficiency metrics.

Q: How do you measure energy and CO₂?
- A: The project integrates a CarbonTracker (from `tracker.py` / `codecarbon`) to collect energy consumed and estimate CO₂ emissions; final numbers are saved in `results` and recorded in the leaderboard CSV.

Q: How reproducible are the results?
- A: The `Trainer` sets a seed (configurable) and logs hyperparameters. Full reproducibility requires controlling hardware, software versions, and non-deterministic GPU ops — we log seeds and hyperparameters to make experiments as reproducible as possible.

Q: How is the leaderboard computed?
- A: Each run records `accuracy`, `energy_kwh`, `co2_g`, and `training_time_sec`. Efficiency is computed as `accuracy_per_kwh = accuracy / energy_kwh` (guarding against division by zero). The leaderboard sorts runs by this metric by default.

Q: How does the recommender work?
- A: The `recommender.py` queries the leaderboard CSV and selects runs by matching criteria: nearest accuracy within tolerance, highest accuracy under a CO₂ budget, or highest accuracy within a time budget. It returns the recommended configuration and expected metrics based on past runs.

Q: What security/privacy concerns exist?
- A: The app runs locally; no external data exfiltration is performed. If you enable MLflow/W&B integrations, ensure credentials and network settings are properly configured. Logged results contain only experiment metadata and anonymized metrics.

Q: What are the main limitations right now?
- A: The real training path relies on local hardware and can be slow/fragile in the Flask debug server. For production-level orchestration, we'd use a worker system (e.g., Celery/RQ) and containerization. Energy estimates depend on CodeCarbon accuracy and machine-specific reporting.

Q: How does the demo mode differ from real training?
- A: Demo mode (`demo=true`) runs `src/core/demo_trainer.py`, which simulates training epochs, generates synthetic but realistic metrics, and writes a leaderboard row. It is fast and safe for demos; it does not load PyTorch.

Q: How would you scale this to many users or long-running jobs?
- A: Offload training to separate worker processes or a job queue, place the web app behind a production server (Gunicorn, uWSGI), store runs in a database rather than a CSV, and add monitoring and quotas.

Q: How do you validate correctness of the leaderboard/recommender?
- A: Unit tests should cover leaderboard metrics calculation and recommender logic. Currently you can run `scripts/populate_leaderboard.py` for deterministic sample entries; adding pytest tests is recommended.

## 7) Troubleshooting (common issues & fixes)

- Server keeps restarting / crash on job start: ensure demo mode is selected for demos; running real training requires proper PyTorch/CUDA installation. Use lazy imports (implemented) to reduce restart surface.
- No leaderboard entries: run `python scripts\populate_leaderboard.py` or run at least one demo job from the UI. Check `results/leaderboard.csv`.
- Recommendation tab not switching: ensure updated `web/static/script.js` and `web/templates/index.html` are in place (fixed in this version).
- Job status shows "running" but epoch stays 0: ensure the job-specific log `results/job_<jobid>.log` contains epoch lines. If empty, the progress callback may have thrown an exception — check server console for errors.

## 8) Demo script / presentation steps (what to show in review)

1. Open terminal, run `python run.py`. Show the server starting logs briefly.
2. Open browser to `http://localhost:5000`.
3. In Train tab, confirm "Demo Mode" is checked and start a training job. Show Dashboard tab updating with progress (epoch bars)
4. When the job finishes, open Leaderboard tab and show the new entry.
5. Run a recommendation: go to Recommendations tab and ask for a model by CO₂ budget / accuracy. Show the returned recommended configuration.
6. Optionally, show `results/leaderboard.csv` and `results/job_<jobid>.log` to prove data is recorded.

## 9) Next steps to make it a strong final-year project

- Add unit tests for leaderboard and recommender (pytest).
- Add a demo automation script that: starts server, posts demo job(s), verifies leaderboard updates, and saves screenshots.
- Add a simple Dockerfile to containerize the web app and a requirements lockfile for reproducible environments.
- Improve documentation: a `README.md` demo section and `start_web.ps1` powershell script to start the server and open a browser.
- Add screenshots and a short video/gif demonstrating the UI for your submission.

## 10) Where to find quick artifacts

- Server: `web/app.py`, run with `web/run.py`.
- Demo trainer: `src/core/demo_trainer.py`.
- Leaderboard CSV: `results/leaderboard.csv`.
- Job logs: `results/job_<jobid>.log`.
- Helper to seed runs: `scripts/populate_leaderboard.py`.

---

If you'd like, I can now:
- Add automated tests and a demo script (recommended), or
- Create a `start_web.ps1` and tidy README instructions for submission.

Choose one and I'll implement it next (I'll update the todo list and run tests afterwards).

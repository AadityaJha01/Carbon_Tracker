# Carbon-Aware ML Training Web Dashboard

A beautiful web interface for training machine learning models while tracking carbon emissions.

## Features

- 🎯 **Interactive Dashboard**: View training statistics and model comparisons
- 🚀 **Start Training Jobs**: Configure and launch training jobs through the web interface
- 📊 **Real-time Progress**: Monitor training progress with live updates
- 🏆 **Leaderboard**: Compare model efficiency (accuracy per kWh)
- 💡 **Smart Recommendations**: Get model recommendations based on accuracy, CO₂ budget, or time constraints
- 📈 **Visualizations**: Charts and graphs for better insights

## Running the Web Dashboard

### Option 1: Using the run script
```bash
cd web
python run.py
```

### Option 2: Using Flask directly
```bash
cd web
python app.py
```

### Option 3: Using Flask CLI
```bash
cd web
export FLASK_APP=app.py
flask run
```

Then open your browser to: `http://localhost:5000`

## API Endpoints

The web application provides REST API endpoints:

- `GET /api/jobs` - Get all training jobs
- `GET /api/jobs/<job_id>` - Get specific job status
- `POST /api/jobs` - Create and start a new training job
- `GET /api/leaderboard` - Get leaderboard data
- `POST /api/recommend` - Get model recommendations
- `GET /api/stats` - Get overall statistics

## Usage

1. **Dashboard Tab**: View overall statistics and active training jobs
2. **Train Model Tab**: Configure and start new training jobs
3. **Leaderboard Tab**: View model efficiency rankings
4. **Recommendations Tab**: Get optimal model configurations based on your constraints

## Requirements

Make sure you have installed all dependencies:
```bash
pip install -r requirements.txt
```

The web dashboard requires:
- Flask >= 3.0.0
- All core ML dependencies (torch, torchvision, etc.)
- CodeCarbon for carbon tracking


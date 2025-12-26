# Quick Start: Web Dashboard

## Start the Web Interface

### Windows
```bash
start_web.bat
```

### Linux/Mac
```bash
chmod +x start_web.sh
./start_web.sh
```

### Manual Start
```bash
cd web
python run.py
```

Then open your browser to: **http://localhost:5000**

## Features

### 1. Dashboard Tab
- View overall statistics (total runs, energy, CO₂)
- See active training jobs with real-time progress
- Model comparison charts

### 2. Train Model Tab
- Configure training parameters through a form
- Start training jobs with one click
- Monitor progress in real-time

### 3. Leaderboard Tab
- View all training runs ranked by efficiency
- Compare models by accuracy, energy, and CO₂

### 4. Recommendations Tab
- Get model recommendations based on:
  - Target accuracy
  - CO₂ budget
  - Time budget

## Example: Starting a Training Job

1. Go to the "Train Model" tab
2. Select model: ResNet18
3. Set epochs: 50
4. Enable FP16 for faster training
5. Click "Start Training"
6. Switch to Dashboard to see progress
7. Check Leaderboard when complete

## API Usage

The web interface also exposes a REST API:

```bash
# Get all jobs
curl http://localhost:5000/api/jobs

# Start a training job
curl -X POST http://localhost:5000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model": "resnet18",
    "dataset": "cifar10",
    "epochs": 50,
    "batch_size": 64,
    "fp16": true
  }'

# Get leaderboard
curl http://localhost:5000/api/leaderboard

# Get recommendations
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "type": "accuracy",
    "target_accuracy": 80.0,
    "tolerance": 2.0
  }'
```

## Troubleshooting

**Port 5000 already in use?**
- Change the port in `web/app.py` or `web/run.py`
- Look for: `app.run(port=5000)` and change to another port

**Import errors?**
- Make sure you're in the project root when running
- Install dependencies: `pip install -r requirements.txt`

**Training jobs not showing?**
- Check that the results directory exists: `mkdir results`
- Verify CodeCarbon is installed: `pip install codecarbon`


"""
Flask web application for Carbon-Aware ML Training Dashboard
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import threading
import uuid
from datetime import datetime
from typing import Dict, Optional
import sys

# Add parent directory to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(project_root))

from src.core.config_loader import load_config
from leaderboard import Leaderboard
from recommender import ModelRecommender
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'carbon-aware-ml-training-secret-key'

# Training jobs storage
training_jobs: Dict[str, Dict] = {}
training_threads: Dict[str, threading.Thread] = {}


def run_training_job(job_id: str, config: Dict):
    """Run training in a separate thread"""
    try:
        training_jobs[job_id]['status'] = 'running'
        training_jobs[job_id]['start_time'] = datetime.now().isoformat()
        
        def progress_callback(epoch, total_epochs, metrics):
            training_jobs[job_id]['current_epoch'] = epoch + 1
            training_jobs[job_id]['total_epochs'] = total_epochs
            training_jobs[job_id]['metrics'] = metrics
            training_jobs[job_id]['last_update'] = datetime.now().isoformat()
            # Debug: append progress to a job-specific logfile so we can inspect background updates
            try:
                log_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, f'job_{job_id}.log')
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now().isoformat()} - epoch={epoch+1}/{total_epochs} metrics={metrics}\n")
            except Exception:
                pass
        # Support a lightweight demo trainer (fast, simulated) to allow presentations
        # without running full PyTorch training. Use config['demo'] = True to enable.
        if config.get('demo', False):
            try:
                from src.core.demo_trainer import DemoTrainer as Trainer
            except Exception as e:
                training_jobs[job_id]['status'] = 'failed'
                training_jobs[job_id]['error'] = f'Failed to import DemoTrainer: {e}'
                training_jobs[job_id]['end_time'] = datetime.now().isoformat()
                return
        else:
            # Lazy import Trainer to keep the web server lightweight at startup
            try:
                from src.core.trainer import Trainer
            except Exception as e:
                training_jobs[job_id]['status'] = 'failed'
                training_jobs[job_id]['error'] = f'Failed to import Trainer: {e}'
                training_jobs[job_id]['end_time'] = datetime.now().isoformat()
                return

        trainer = Trainer(config, progress_callback=progress_callback)
        results = trainer.train()
        
        training_jobs[job_id]['status'] = 'completed'
        training_jobs[job_id]['results'] = results
        training_jobs[job_id]['end_time'] = datetime.now().isoformat()
        
    except Exception as e:
        training_jobs[job_id]['status'] = 'failed'
        training_jobs[job_id]['error'] = str(e)
        training_jobs[job_id]['end_time'] = datetime.now().isoformat()


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    """Get all training jobs"""
    return jsonify(training_jobs)


@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    """Get specific training job"""
    if job_id not in training_jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(training_jobs[job_id])


@app.route('/api/jobs', methods=['POST'])
def create_job():
    """Create and start a new training job"""
    data = request.json
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Build config from request
    config = {
        'model': data.get('model', 'resnet18'),
        'dataset': data.get('dataset', 'cifar10'),
        'epochs': int(data.get('epochs', 50)),
        'batch_size': int(data.get('batch_size', 64)),
        'learning_rate': float(data.get('learning_rate', 0.01)),
        'optimizer': data.get('optimizer', 'adam'),
        'momentum': float(data.get('momentum', 0.9)),
        'weight_decay': float(data.get('weight_decay', 0.0001)),
        'fp16': data.get('fp16', False),
        'early_stop': data.get('early_stop', False),
        'early_stop_patience': int(data.get('early_stop_patience', 10)),
        'target_acc': float(data.get('target_acc')) if data.get('target_acc') else None,
        'device': data.get('device', 'cuda'),
        'num_workers': int(data.get('num_workers', 4)),
        'region': data.get('region'),
    'demo': data.get('demo', False),
        'seed': int(data.get('seed', 42)),
        'log_dir': './results',
        'use_mlflow': data.get('use_mlflow', False),
        'use_wandb': data.get('use_wandb', False)
    }
    
    # Initialize job
    training_jobs[job_id] = {
        'id': job_id,
        'config': config,
        'status': 'queued',
        'created_at': datetime.now().isoformat(),
        'current_epoch': 0,
        'total_epochs': config['epochs'],
        'metrics': {}
    }
    
    # Start training in background thread
    thread = threading.Thread(target=run_training_job, args=(job_id, config))
    thread.daemon = True
    thread.start()
    training_threads[job_id] = thread
    
    return jsonify({'job_id': job_id, 'status': 'started'}), 201


@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    """Get leaderboard data"""
    try:
        leaderboard_path = './results/leaderboard.csv'
        if not os.path.exists(leaderboard_path):
            return jsonify({'runs': [], 'message': 'No runs yet'})
        
        leaderboard = Leaderboard(csv_path=leaderboard_path)
        df = leaderboard.get_leaderboard(sort_by='accuracy_per_kwh', top_n=50)
        
        if df.empty:
            return jsonify({'runs': [], 'message': 'No runs yet'})
        
        # Convert to dict
        runs = df.to_dict('records')
        return jsonify({'runs': runs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Get model recommendation"""
    data = request.json
    recommendation_type = data.get('type', 'accuracy')
    
    try:
        leaderboard_path = './results/leaderboard.csv'
        if not os.path.exists(leaderboard_path):
            return jsonify({'error': 'No training data available for recommendations'}), 404
        
        df = pd.read_csv(leaderboard_path)
        if df.empty:
            return jsonify({'error': 'No training data available for recommendations'}), 404
        
        recommender = ModelRecommender(df)
        
        if recommendation_type == 'accuracy':
            target_acc = float(data.get('target_accuracy', 80.0))
            tolerance = float(data.get('tolerance', 2.0))
            recommendation = recommender.recommend_by_accuracy(target_acc, tolerance)
        elif recommendation_type == 'co2':
            max_co2 = float(data.get('max_co2_g', 100.0))
            recommendation = recommender.recommend_by_co2_budget(max_co2)
        elif recommendation_type == 'time':
            max_time = float(data.get('max_time_hours', 1.0))
            recommendation = recommender.recommend_by_time_budget(max_time)
        else:
            return jsonify({'error': 'Invalid recommendation type'}), 400
        
        if recommendation is None:
            return jsonify({'error': 'No suitable recommendation found'}), 404
        
        return jsonify(recommendation)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get overall statistics"""
    try:
        leaderboard_path = './results/leaderboard.csv'
        if not os.path.exists(leaderboard_path):
            return jsonify({
                'total_runs': 0,
                'total_energy_kwh': 0,
                'total_co2_g': 0,
                'total_time_hours': 0,
                'models': {}
            })
        
        df = pd.read_csv(leaderboard_path)
        if df.empty:
            return jsonify({
                'total_runs': 0,
                'total_energy_kwh': 0,
                'total_co2_g': 0,
                'total_time_hours': 0,
                'models': {}
            })
        
        stats = {
            'total_runs': len(df),
            'total_energy_kwh': float(df['energy_kwh'].sum()),
            'total_co2_g': float(df['co2_g'].sum()),
            'total_time_hours': float(df['training_time_sec'].sum() / 3600),
            'models': {}
        }
        
        # Per-model stats
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            stats['models'][model] = {
                'runs': len(model_df),
                'avg_accuracy': float(model_df['accuracy'].mean()),
                'avg_energy_kwh': float(model_df['energy_kwh'].mean()),
                'avg_co2_g': float(model_df['co2_g'].mean()),
                'best_accuracy': float(model_df['accuracy'].max()),
                'best_efficiency': float(model_df['accuracy_per_kwh'].max())
            }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/plots/<plot_name>')
def get_plot(plot_name):
    """Serve plot images"""
    plot_dir = './results'
    return send_from_directory(plot_dir, plot_name)


@app.route('/api/job_log/<job_id>', methods=['GET'])
def get_job_log(job_id):
    """Return job log contents for a given job id"""
    try:
        log_path = os.path.join('./results', f'job_{job_id}.log')
        if not os.path.exists(log_path):
            return jsonify({'error': 'Log not found'}), 404

        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return jsonify({'job_id': job_id, 'log': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/leaderboard', methods=['GET'])
def download_leaderboard():
    """Download the raw leaderboard CSV"""
    try:
        leaderboard_path = os.path.abspath('./results')
        filename = 'leaderboard.csv'
        if not os.path.exists(os.path.join(leaderboard_path, filename)):
            return jsonify({'error': 'Leaderboard not available'}), 404
        return send_from_directory(leaderboard_path, filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Create results directory if it doesn't exist
    os.makedirs('./results', exist_ok=True)
    
    print("Starting Carbon-Aware ML Training Dashboard...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)


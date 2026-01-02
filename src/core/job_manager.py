"""
Asynchronous job manager for training jobs.

Provides a JobManager class that creates and manages training jobs in background
threads. Each job has a unique id, status, progress, and a job-specific log
written to `results/job_<jobid>.log`.

This keeps orchestration separated from the Flask app and follows the
project architecture requirements.
"""
import os
import threading
import uuid
import time
from datetime import datetime
from typing import Callable, Dict, Optional


class JobManager:
    def __init__(self, results_dir: str = './results'):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.jobs: Dict[str, Dict] = {}
        self.threads: Dict[str, threading.Thread] = {}

    def _write_log(self, job_id: str, text: str):
        path = os.path.join(self.results_dir, f'job_{job_id}.log')
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(text + '\n')
        except Exception:
            pass

    def _run_job(self, job_id: str, config: Dict, trainer_class):
        """Internal runner executed inside a thread."""
        try:
            job = self.jobs[job_id]
            job['status'] = 'running'
            job['start_time'] = datetime.now().isoformat()

            def progress_callback(epoch, total_epochs, metrics):
                job['current_epoch'] = epoch + 1
                job['total_epochs'] = total_epochs
                job['metrics'] = metrics
                job['last_update'] = datetime.now().isoformat()
                # append to job log
                self._write_log(job_id, f"{datetime.now().isoformat()} - epoch={epoch+1}/{total_epochs} metrics={metrics}")

            trainer = trainer_class(config, progress_callback=progress_callback)
            results = trainer.train()

            job['status'] = 'completed'
            job['results'] = results
            job['end_time'] = datetime.now().isoformat()
            self._write_log(job_id, f"{datetime.now().isoformat()} - completed: {results}")

        except Exception as e:
            self.jobs[job_id]['status'] = 'failed'
            self.jobs[job_id]['error'] = str(e)
            self.jobs[job_id]['end_time'] = datetime.now().isoformat()
            self._write_log(job_id, f"{datetime.now().isoformat()} - failed: {e}")

    def create_job(self, config: Dict, trainer_class) -> str:
        """Create and start a new job using the provided trainer class.

        Args:
            config: configuration dict forwarded to the trainer
            trainer_class: callable/class implementing Trainer(config, progress_callback)

        Returns:
            job_id
        """
        job_id = str(uuid.uuid4())
        job = {
            'id': job_id,
            'config': config,
            'status': 'queued',
            'created_at': datetime.now().isoformat(),
            'current_epoch': 0,
            'total_epochs': int(config.get('epochs', 0)),
            'metrics': {}
        }
        self.jobs[job_id] = job

        thread = threading.Thread(target=self._run_job, args=(job_id, config, trainer_class))
        thread.daemon = True
        thread.start()
        self.threads[job_id] = thread

        return job_id

    def get_job(self, job_id: str) -> Optional[Dict]:
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> Dict[str, Dict]:
        return self.jobs

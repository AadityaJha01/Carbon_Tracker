"""
Utility script to move core files to src/ directory
This helps maintain backward compatibility
"""

import os
import shutil

# Files to move to src/core
files_to_move = {
    'tracker.py': 'src/core/tracker.py',
    'logger.py': 'src/core/logger.py',
    'optimizers.py': 'src/core/optimizers.py',
    'leaderboard.py': 'src/core/leaderboard.py',
    'recommender.py': 'src/core/recommender.py'
}

def move_files():
    """Move files to src/core while keeping originals for compatibility"""
    for src, dst in files_to_move.items():
        if os.path.exists(src) and not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"Copied {src} to {dst}")

if __name__ == '__main__':
    move_files()


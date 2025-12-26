"""
Run the Flask web application
"""

from app import app

if __name__ == '__main__':
    import os
    os.makedirs('./results', exist_ok=True)
    
    print("="*60)
    print("Carbon-Aware ML Training Dashboard")
    print("="*60)
    print("Starting server...")
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop the server")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)


#!/usr/bin/env python3
"""
Production runner for AI Resume Parser application
"""

import os
from app import create_app

# Get configuration from environment
config_name = os.environ.get('FLASK_ENV', 'development')

# Create application
app = create_app(config_name)

if __name__ == '__main__':
    # Production settings
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = config_name == 'development'
    
    print(f"Starting AI Resume Parser in {config_name} mode...")
    print(f"Server running on http://{host}:{port}")
    
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )

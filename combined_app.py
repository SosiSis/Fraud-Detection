#!/usr/bin/env python3
"""
Combined Fraud Detection Application
Serves both API and Dashboard from the same Flask instance
"""

import os
from flask import Flask, request, jsonify
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

# Import both applications
import serve_model
import dashboard_app

def create_combined_app():
    """Create a combined Flask app that serves both API and Dashboard"""
    
    # Get the main API app
    api_app = serve_model.app
    
    # Get the dashboard server
    dashboard_server = dashboard_app.server
    
    # Create a dispatcher that routes requests
    application = DispatcherMiddleware(api_app, {
        '/dashboard': dashboard_server,
    })
    
    return application

if __name__ == '__main__':
    # Create combined application
    app = create_combined_app()
    
    # Run on the specified port
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Combined Fraud Detection App on port {port}")
    print(f"📊 API available at: http://0.0.0.0:{port}/")
    print(f"📈 Dashboard available at: http://0.0.0.0:{port}/dashboard/")
    
    run_simple('0.0.0.0', port, app, use_reloader=False, use_debugger=False)
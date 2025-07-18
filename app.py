# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:53:45 2025

@author: freez
"""

from flask import Flask
from flask_cors import CORS
import logging

# Import our modules
from models import MovieRecommendationModel
from routes import register_routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create model instance
model = MovieRecommendationModel()

# Register all routes
register_routes(app, model)

if __name__ == '__main__':
    # Try to load model on startup
    try:
        logger.info("Starting Flask application...")
        model.load_and_train_model()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.warning(f"Could not load model on startup: {str(e)}")
        logger.info("Model can be trained via /api/train endpoint")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
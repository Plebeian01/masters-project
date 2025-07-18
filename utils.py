# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:54:37 2025

@author: freez
"""

import numpy as np
import os
import csv
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def make_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    return obj

def save_study_response(user_id, movie_title, explanation_type, helpfulness_rating):
    """Save user study response to CSV"""
    try:
        filename = 'study_responses.csv'
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'user_id', 'movie_title', 'explanation_type', 'helpfulness_rating']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'movie_title': movie_title,
                'explanation_type': explanation_type,
                'helpfulness_rating': helpfulness_rating
            })
        
        logger.info(f"Saved study response: {user_id}, {movie_title}, {explanation_type}, {helpfulness_rating}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving study response: {str(e)}")
        return False
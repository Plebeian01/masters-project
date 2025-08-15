# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:54:37 2025

@author: freez
"""

# -*- coding: utf-8 -*-
"""
Utility functions for the Movie Recommendation Study
"""

import numpy as np
import os
import csv
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def make_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
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

def save_study_response(user_id, movie_title, explanation_type, helpfulness_rating,
                        recommendation_rating=None, is_baseline=False,
                        predicted_rating=None, genre_similarity=None,
                        input_movies=None):
    """
    Save a user’s response to the study CSV.

    Parameters
    ----------
    user_id : str
        Unique participant identifier.
    movie_title : str
        Title of the recommended movie.
    explanation_type : str
        Type of explanation shown (none, basic, shap, lime, llm).
    helpfulness_rating : int
        User rating of explanation helpfulness (1–5).
    recommendation_rating : int, optional
        User rating of how good the recommendation seemed (1–5).
    is_baseline : bool, optional
        Whether this was the intentionally poor decoy recommendation.
    predicted_rating : float, optional
        Predicted rating shown to the user (for later analysis).
    genre_similarity : float, optional
        Genre similarity between the user’s profile and the movie.
    input_movies : str, optional
        Movies rated by the user before generating recommendations (semicolon-separated).
    """

    try:
        filename = 'study_responses.csv'
        file_exists = os.path.isfile(filename)

        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'timestamp',
                'user_id',
                'movie_title',
                'explanation_type',
                'helpfulness_rating',
                'recommendation_rating',
                'is_baseline',
                'predicted_rating',
                'genre_similarity',
                'input_movies'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'movie_title': movie_title,
                'explanation_type': explanation_type,
                'helpfulness_rating': helpfulness_rating,
                'recommendation_rating': recommendation_rating,
                'is_baseline': is_baseline,
                'predicted_rating': predicted_rating,
                'genre_similarity': genre_similarity,
                'input_movies': input_movies
            })

        logger.info(f"Saved study response for '{movie_title}' "
                    f"(helpfulness={helpfulness_rating}, recommendation={recommendation_rating}, baseline={is_baseline})")
        return True

    except Exception as e:
        logger.error(f"Error saving study response: {str(e)}")
        return False

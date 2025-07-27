# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:54:16 2025

@author: freez
"""

# -*- coding: utf-8 -*-
"""
Flask Routes for the Movie Recommendation Study
"""

from flask import request, jsonify, send_from_directory
import uuid
import logging
import os

logger = logging.getLogger(__name__)

def register_routes(app, model):
    """Register all Flask routes"""

    # Import utils functions here to avoid circular imports
    from utils import make_json_serializable, save_study_response

    @app.route('/')
    def serve_frontend():
        """Serve the main HTML page"""
        return send_from_directory('.', 'index.html')

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'model_trained': model.is_trained
        })

    @app.route('/api/movies/search', methods=['GET'])
    def search_movies():
        """Search for movies"""
        search_term = request.args.get('q', '')
        try:
            movies = model.get_available_movies(search_term)
            return jsonify({"success": True, "movies": movies})
        except Exception as e:
            logger.error(f"Error searching movies: {str(e)}")
            return jsonify({"success": False, "error": "Error searching movies"}), 500

    @app.route('/api/recommendations', methods=['POST'])
    def get_recommendations():
        """Get movie recommendations with explanations"""
        try:
            data = request.json
            if not data or 'ratings' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Invalid request. Expected format: {"ratings": [{"title": "Movie Title", "rating": 5.0}]}'
                }), 400

            ratings = data['ratings']
            n_recommendations = data.get('n', 5)
            include_explanations = data.get('include_explanations', True)

            user_ratings = [(r['title'], r['rating']) for r in ratings]
            if not user_ratings:
                return jsonify({"success": False, "error": "No valid ratings provided"}), 400

            recommendations = model.get_recommendations_with_explanations(
                user_ratings, n_recommendations, include_explanations
            )
            recommendations = make_json_serializable(recommendations)

            return jsonify({
                "success": True,
                "recommendations": recommendations,
                "input_ratings": ratings
            })

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return jsonify({"success": False, "error": "Error generating recommendations"}), 500

    @app.route('/api/study/recommendations', methods=['POST'])
    def get_study_recommendations():
        """Get movie recommendations for user study with randomized explanations"""
        try:
            data = request.json
            if not data or 'ratings' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Invalid request. Expected format: {"ratings": [{"title": "Movie Title", "rating": 5.0}]}'
                }), 400

            ratings = data['ratings']
            user_id = data.get('user_id') or str(uuid.uuid4())  # Generate if not provided
            user_ratings = [(r['title'], r['rating']) for r in ratings]

            if not user_ratings:
                return jsonify({"success": False, "error": "No valid ratings provided"}), 400

            recommendations = model.get_recommendations_for_study(user_ratings, 5)
            recommendations = make_json_serializable(recommendations)

            return jsonify({
                "success": True,
                "recommendations": recommendations,
                "user_id": user_id,
                "input_ratings": ratings
            })

        except Exception as e:
            logger.error(f"Error generating study recommendations: {str(e)}")
            return jsonify({"success": False, "error": "Error generating study recommendations"}), 500

    @app.route('/api/study/response', methods=['POST'])
    def submit_study_response():
        """Submit user study response (now logs both recommendation & explanation ratings)"""
        try:
            data = request.json
            if not data:
                return jsonify({"success": False, "error": "No data provided"}), 400

            required_fields = [
                'user_id',
                'movie_title',
                'explanation_type',
                'helpfulness_rating',
                'recommendation_rating'
            ]
            missing = [field for field in required_fields if field not in data]
            if missing:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field(s): {', '.join(missing)}"
                }), 400

            # Validate ratings
            for field in ['helpfulness_rating', 'recommendation_rating']:
                val = data[field]
                if not isinstance(val, (int, float)) or val < 1 or val > 5:
                    return jsonify({
                        "success": False,
                        "error": f"{field.replace('_',' ').capitalize()} must be between 1 and 5"
                    }), 400

            success = save_study_response(
                user_id=data['user_id'],
                movie_title=data['movie_title'],
                explanation_type=data['explanation_type'],
                helpfulness_rating=data['helpfulness_rating'],
                recommendation_rating=data['recommendation_rating'],
                is_baseline=data.get('is_baseline', False),
                predicted_rating=data.get('predicted_rating'),
                genre_similarity=data.get('genre_similarity'),
                input_movies=data.get('input_movies')
            )

            if success:
                return jsonify({"success": True, "message": "Response saved successfully"})
            else:
                return jsonify({"success": False, "error": "Failed to save response"}), 500

        except Exception as e:
            logger.error(f"Error saving study response: {str(e)}")
            return jsonify({"success": False, "error": "Unexpected error saving response"}), 500

    @app.route('/api/train', methods=['POST'])
    def train_model():
        """Train or retrain the model (for development/testing)"""
        try:
            data = request.json or {}
            data_dir = data.get('data_dir', 'data/100kDataset')

            model.load_and_train_model(data_dir)
            return jsonify({"success": True, "message": "Model trained successfully"})

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return jsonify({"success": False, "error": "Error training model"}), 500

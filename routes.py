# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:54:16 2025

@author: freez
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
            return jsonify({
                'success': True,
                'movies': movies
            })
        except Exception as e:
            logger.error(f"Error searching movies: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

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
            
            # Convert to expected format
            user_ratings = [(r['title'], r['rating']) for r in ratings]
            
            if not user_ratings:
                return jsonify({
                    'success': False,
                    'error': 'No valid ratings provided'
                }), 400
            
            recommendations = model.get_recommendations_with_explanations(
                user_ratings, n_recommendations, include_explanations
            )
            
            # Ensure all values are JSON serializable
            recommendations = make_json_serializable(recommendations)
            
            return jsonify({
                'success': True,
                'recommendations': recommendations,
                'input_ratings': ratings
            })
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

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
            
            # Convert to expected format
            user_ratings = [(r['title'], r['rating']) for r in ratings]
            
            if not user_ratings:
                return jsonify({
                    'success': False,
                    'error': 'No valid ratings provided'
                }), 400
            
            recommendations = model.get_recommendations_for_study(user_ratings, 5)
            
            # Ensure all values are JSON serializable
            recommendations = make_json_serializable(recommendations)
            
            return jsonify({
                'success': True,
                'recommendations': recommendations,
                'user_id': user_id,
                'input_ratings': ratings
            })
            
        except Exception as e:
            logger.error(f"Error generating study recommendations: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/study/response', methods=['POST'])
    def submit_study_response():
        """Submit user study response"""
        try:
            data = request.json
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No data provided'
                }), 400
            
            required_fields = ['user_id', 'movie_title', 'explanation_type', 'helpfulness_rating']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'success': False,
                        'error': f'Missing required field: {field}'
                    }), 400
            
            # Validate helpfulness rating
            rating = data['helpfulness_rating']
            if not isinstance(rating, (int, float)) or rating < 1 or rating > 5:
                return jsonify({
                    'success': False,
                    'error': 'Helpfulness rating must be between 1 and 5'
                }), 400
            
            success = save_study_response(
                data['user_id'],
                data['movie_title'],
                data['explanation_type'],
                rating
            )
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Response saved successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to save response'
                }), 500
            
        except Exception as e:
            logger.error(f"Error saving study response: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/train', methods=['POST'])
    def train_model():
        """Train the model (for development purposes)"""
        try:
            data = request.json or {}
            data_dir = data.get('data_dir', 'data/100kDataset')
            
            model.load_and_train_model(data_dir)
            
            return jsonify({
                'success': True,
                'message': 'Model trained successfully'
            })
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
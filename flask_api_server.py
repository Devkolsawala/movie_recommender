# flask_api_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import sys
import os
from enhanced_test import ImprovedMultiGenreRecommender
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global recommenders cache
recommenders_cache = {}

def convert_to_serializable(obj):
    """Convert numpy/pandas data types to native Python types for JSON serialization"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def get_or_create_recommender(media_type, lang_id=None):
    """Get or create a recommender instance with caching"""
    cache_key = f"{media_type}_{lang_id}"
    
    if cache_key not in recommenders_cache:
        try:
            logger.info(f"Creating new recommender for {media_type}, lang_id: {lang_id}")
            recommender = ImprovedMultiGenreRecommender(
                media_type=media_type,
                lang_id=lang_id,
                top_k=15,  # Get more recommendations
                view_threshold=5000,  # Lower threshold for more results
                enforce_language_matching=True
            )
            recommenders_cache[cache_key] = recommender
            logger.info(f"Successfully created recommender for {cache_key}")
        except Exception as e:
            logger.error(f"Error creating recommender for {cache_key}: {str(e)}")
            raise
    
    return recommenders_cache[cache_key]

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "AI Movie Recommendation System is running"})

@app.route('/api/dataset/stats', methods=['GET'])
def get_dataset_stats():
    """Get dataset statistics"""
    try:
        # Load the main dataset
        df = pd.read_csv('final_dataset.csv')
        
        stats = {
            'total_items': int(len(df)),
            'total_movies': int(len(df[df['ismovie'] == 1])),
            'total_series': int(len(df[df['ismovie'] == 0])),
            'total_short_dramas': int(len(df[df['ismovie'] == 2])),
            'total_languages': int(len(df['lang_id'].unique())) if 'lang_id' in df.columns else 0,
            'unique_genres': len(set(genre.strip() for genres in df['genres'].dropna() 
                                   for genre in str(genres).split(',') if genre.strip()))
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    
    except Exception as e:
        logger.error(f"Error getting dataset stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/movies/search', methods=['GET'])
def search_movies():
    """Search for movies and series"""
    try:
        query = request.args.get('q', '').lower()
        media_type = request.args.get('media_type', '')
        lang_id = request.args.get('lang_id', '')
        limit = int(request.args.get('limit', 20))
        
        # Load dataset
        df = pd.read_csv('final_dataset.csv')
        
        # Apply filters
        filtered_df = df.copy()
        
        if media_type:
            media_type_mapping = {'movie': 1, 'series': 0, 'short_drama': 2}
            if media_type in media_type_mapping:
                filtered_df = filtered_df[filtered_df['ismovie'] == media_type_mapping[media_type]]
        
        if lang_id:
            filtered_df = filtered_df[filtered_df['lang_id'] == int(lang_id)]
        
        # Search by title and genres
        if query:
            mask = (filtered_df['title'].str.lower().str.contains(query, na=False) |
                   filtered_df['genres'].str.lower().str.contains(query, na=False))
            filtered_df = filtered_df[mask]
        
        # Limit results and convert to list
        results = []
        for _, row in filtered_df.head(limit).iterrows():
            result = {
                'id': convert_to_serializable(row['id']),
                'title': str(row['title']) if pd.notna(row['title']) else '',
                'genres': str(row['genres']) if pd.notna(row['genres']) else '',
                'lang_id': convert_to_serializable(row['lang_id']) if pd.notna(row['lang_id']) else 1,
                'language': get_language_name(int(row['lang_id']) if pd.notna(row['lang_id']) else 1),
                'ismovie': convert_to_serializable(row['ismovie']),
                'media_type': get_media_type_name(int(row['ismovie'])),
                'imdb_rating': convert_to_serializable(row['imdb_rating']) if pd.notna(row['imdb_rating']) else 0.0,
                'views': convert_to_serializable(row['views']) if pd.notna(row['views']) else 0
            }
            results.append(result)
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        logger.error(f"Error searching movies: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/languages', methods=['GET'])
def get_languages():
    """Get available languages"""
    try:
        # Load language mapping
        try:
            lang_df = pd.read_csv('language_tbl.csv', header=None)
            lang_df.columns = ['lang_id', 'language_name', 'priority']
            languages = [
                {'id': convert_to_serializable(row['lang_id']), 'name': str(row['language_name'])}
                for _, row in lang_df.iterrows()
            ]
        except FileNotFoundError:
            # Fallback to default languages
            languages = [
                {'id': 1, 'name': 'English'},
                {'id': 2, 'name': 'हिन्दी'},
                {'id': 3, 'name': 'ગુજરાતી'},
                {'id': 4, 'name': 'Español'},
                {'id': 5, 'name': 'தமிழ்'},
                {'id': 7, 'name': 'മലയാളം'},
                {'id': 24, 'name': 'Korean'},
                {'id': 25, 'name': 'తెలుగు'}
            ]
        
        return jsonify({
            'success': True,
            'languages': languages
        })
    
    except Exception as e:
        logger.error(f"Error getting languages: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get AI-powered recommendations"""
    try:
        data = request.get_json()
        
        if not data or 'clicked_ids' not in data:
            return jsonify({
                'success': False,
                'error': 'clicked_ids is required'
            }), 400
        
        clicked_ids = data['clicked_ids']
        media_type = data.get('media_type', 'movie')
        lang_id = data.get('lang_id', None)
        top_k = data.get('top_k', 10)
        
        if not clicked_ids:
            return jsonify({
                'success': False,
                'error': 'At least one clicked_id is required'
            }), 400
        
        logger.info(f"Getting recommendations for IDs: {clicked_ids}, media_type: {media_type}, lang_id: {lang_id}")
        
        # Get or create recommender
        recommender = get_or_create_recommender(media_type, lang_id)
        
        # Validate clicked IDs
        valid_clicked_ids = recommender.validate_clicked_ids(clicked_ids)
        
        if not valid_clicked_ids:
            return jsonify({
                'success': False,
                'error': 'No valid clicked IDs found in the current dataset',
                'suggested_ids': get_sample_ids(media_type, lang_id)
            }), 400
        
        # Get recommendations
        recommendations = recommender.get_recommendations(
            clicked_ids=valid_clicked_ids,
            apply_genre_split=True
        )
        
        # Format response with proper data type conversion
        formatted_recommendations = []
        for rec in recommendations[:top_k]:
            formatted_rec = {
                'id': convert_to_serializable(rec['id']),
                'title': str(rec['title']) if 'title' in rec else '',
                'final_score': convert_to_serializable(rec['final_score']),
                'semantic_score': convert_to_serializable(rec['semantic_score']),
                'genre_score': convert_to_serializable(rec['genre_score']),
                'rating_score': convert_to_serializable(rec['rating_score']),
                'popularity_score': convert_to_serializable(rec['popularity_score']),
                'content_boost': convert_to_serializable(rec['content_boost']),
                'views': convert_to_serializable(rec['views']),
                'imdb_rating': convert_to_serializable(rec['imdb_rating']),
                'genres': str(rec['genres']) if 'genres' in rec else '',
                'lang_id': convert_to_serializable(rec['lang_id']),
                'language_name': str(rec['language_name']) if 'language_name' in rec else '',
                'ismovie': convert_to_serializable(rec['ismovie']),
                'media_type_name': str(rec['media_type_name']) if 'media_type_name' in rec else ''
            }
            formatted_recommendations.append(formatted_rec)
        
        response_data = {
            'success': True,
            'recommendations': formatted_recommendations,
            'valid_clicked_ids': [convert_to_serializable(id) for id in valid_clicked_ids],
            'total_found': len(recommendations),
            'returned_count': len(formatted_recommendations)
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/validate_ids', methods=['POST'])
def validate_ids():
    """Validate clicked IDs and return sample IDs if needed"""
    try:
        data = request.get_json()
        clicked_ids = data.get('clicked_ids', [])
        media_type = data.get('media_type', 'movie')
        lang_id = data.get('lang_id', None)
        
        # Get or create recommender
        recommender = get_or_create_recommender(media_type, lang_id)
        
        # Validate IDs
        valid_ids = recommender.validate_clicked_ids(clicked_ids)
        invalid_ids = [id for id in clicked_ids if id not in valid_ids]
        
        # Get sample IDs if needed
        sample_ids = get_sample_ids(media_type, lang_id) if not valid_ids else []
        
        return jsonify({
            'success': True,
            'valid_ids': [convert_to_serializable(id) for id in valid_ids],
            'invalid_ids': [convert_to_serializable(id) for id in invalid_ids],
            'sample_ids': sample_ids
        })
    
    except Exception as e:
        logger.error(f"Error validating IDs: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def get_language_name(lang_id):
    """Get language name from ID"""
    language_map = {
        1: "English", 2: "हिन्दी", 3: "ગુજરાતી", 4: "Español", 5: "தமிழ்",
        6: "Other", 7: "മലയാളം", 10: "Français", 12: "ਪੰਜਾਬੀ", 13: "বাংলা",
        14: "मराठी", 15: "ಕನ್ನಡ", 23: "Chinese", 24: "Korean", 25: "తెలుగు",
        49: "عربي", 55: "Deutsch", 67: "Português"
    }
    return language_map.get(lang_id, f"Language {lang_id}")

def get_media_type_name(ismovie):
    """Get media type name from ismovie value"""
    mapping = {0: 'Series/TV Show', 1: 'Movie', 2: 'Short Drama'}
    return mapping.get(ismovie, 'Unknown')

def get_sample_ids(media_type, lang_id, limit=10):
    """Get sample IDs for the given media type and language"""
    try:
        df = pd.read_csv('final_dataset.csv')
        
        # Apply filters
        media_type_mapping = {'movie': 1, 'series': 0, 'short_drama': 2}
        if media_type in media_type_mapping:
            df = df[df['ismovie'] == media_type_mapping[media_type]]
        
        if lang_id is not None:
            df = df[df['lang_id'] == lang_id]
        
        # Get sample IDs with details
        samples = []
        for _, row in df.head(limit).iterrows():
            sample = {
                'id': convert_to_serializable(row['id']),
                'title': str(row['title']) if pd.notna(row['title']) else '',
                'genres': str(row['genres']) if pd.notna(row['genres']) else '',
                'language': get_language_name(int(row['lang_id']) if pd.notna(row['lang_id']) else 1),
                'media_type': get_media_type_name(int(row['ismovie']))
            }
            samples.append(sample)
        
        return samples
    
    except Exception as e:
        logger.error(f"Error getting sample IDs: {str(e)}")
        return []

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Check if required files exist
    required_files = ['final_dataset.csv', 'enhanced_test.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure all required files are in the same directory as this script.")
        sys.exit(1)
    
    print("🚀 Starting AI Movie Recommendation API Server...")
    print("📊 Loading dataset and initializing AI models...")
    
    try:
        # Pre-load a default recommender to check everything works
        test_recommender = get_or_create_recommender('movie', None)
        print("✅ AI models loaded successfully!")
        print("🌐 API server is ready!")
        print("\n📡 Available endpoints:")
        print("  GET  /api/health - Health check")
        print("  GET  /api/dataset/stats - Dataset statistics")
        print("  GET  /api/movies/search - Search movies/series")
        print("  GET  /api/languages - Get available languages")
        print("  POST /api/recommendations - Get AI recommendations")
        print("  POST /api/validate_ids - Validate clicked IDs")
        print("\n🔗 Frontend integration ready!")
        
    except Exception as e:
        print(f"❌ Error initializing server: {str(e)}")
        sys.exit(1)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to False for production
        threaded=True
    )
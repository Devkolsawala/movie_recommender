# config.py
"""
Configuration file for the Enhanced Movie Recommendation System
Adjust these parameters to fine-tune your recommendations
"""

class RecommendationConfig:
    # Model Configuration
    SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'  # Can be changed to other models
    
    # Recommendation Weights (should sum to 1.0)
    WEIGHTS = {
        'semantic_similarity': 0.5,   # Text embedding similarity (most important)
        'genre_similarity': 0.25,     # Genre overlap
        'rating_similarity': 0.15,    # IMDB rating similarity  
        'popularity_boost': 0.1       # View-based popularity
    }
    
    # Alternative weight configurations for different use cases:
    
    # Content-focused (prioritizes content similarity over popularity)
    CONTENT_FOCUSED_WEIGHTS = {
        'semantic_similarity': 0.6,
        'genre_similarity': 0.3,
        'rating_similarity': 0.1,
        'popularity_boost': 0.0
    }
    
    # Popularity-focused (includes trending items)
    POPULARITY_FOCUSED_WEIGHTS = {
        'semantic_similarity': 0.4,
        'genre_similarity': 0.2,
        'rating_similarity': 0.1,
        'popularity_boost': 0.3
    }
    
    # Quality-focused (prioritizes highly rated content)
    QUALITY_FOCUSED_WEIGHTS = {
        'semantic_similarity': 0.4,
        'genre_similarity': 0.2,
        'rating_similarity': 0.3,
        'popularity_boost': 0.1
    }
    
    # Filtering Parameters
    DEFAULT_VIEW_THRESHOLD = 10000      # Minimum views to be considered
    LOW_VIEW_THRESHOLD = 1000           # For discovering hidden gems
    HIGH_VIEW_THRESHOLD = 100000        # For popular content only
    
    # Diversity Parameters
    DIVERSITY_THRESHOLD = 0.8           # Similarity threshold for diversity filtering
    STRICT_DIVERSITY_THRESHOLD = 0.7    # More strict diversity
    LOOSE_DIVERSITY_THRESHOLD = 0.9     # Less strict diversity
    
    # Rating Thresholds
    HIGH_QUALITY_RATING_THRESHOLD = 7.5  # Movies above this are considered high quality
    DECENT_RATING_THRESHOLD = 6.0        # Movies above this are decent
    
    # Recommendation Parameters
    DEFAULT_TOP_K = 10
    MAX_RECOMMENDATIONS = 50
    
    # Genre Boost Parameters
    GENRE_EXACT_MATCH_BOOST = 0.1       # Boost for exact genre matches
    GENRE_PARTIAL_MATCH_BOOST = 0.05    # Boost for partial genre matches
    
    # Text Enhancement Parameters
    TITLE_WEIGHT_MULTIPLIER = 3          # How many times to repeat title in enhanced text
    GENRE_WEIGHT_MULTIPLIER = 2          # How many times to repeat genres
    
    # User Preference Profiles
    USER_PROFILES = {
        'casual_viewer': {
            'weights': POPULARITY_FOCUSED_WEIGHTS,
            'view_threshold': DEFAULT_VIEW_THRESHOLD,
            'diversity_threshold': LOOSE_DIVERSITY_THRESHOLD
        },
        'cinephile': {
            'weights': QUALITY_FOCUSED_WEIGHTS,
            'view_threshold': LOW_VIEW_THRESHOLD,
            'diversity_threshold': STRICT_DIVERSITY_THRESHOLD
        },
        'content_explorer': {
            'weights': CONTENT_FOCUSED_WEIGHTS,
            'view_threshold': LOW_VIEW_THRESHOLD,
            'diversity_threshold': DIVERSITY_THRESHOLD
        },
        'mainstream': {
            'weights': WEIGHTS,
            'view_threshold': HIGH_VIEW_THRESHOLD,
            'diversity_threshold': DIVERSITY_THRESHOLD
        }
    }
    
    @classmethod
    def get_profile_config(cls, profile_name='default'):
        """Get configuration for a specific user profile"""
        if profile_name in cls.USER_PROFILES:
            return cls.USER_PROFILES[profile_name]
        else:
            return {
                'weights': cls.WEIGHTS,
                'view_threshold': cls.DEFAULT_VIEW_THRESHOLD,
                'diversity_threshold': cls.DIVERSITY_THRESHOLD
            }
    
    @classmethod
    def validate_weights(cls, weights):
        """Validate that weights sum to approximately 1.0"""
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return True

# Example usage configurations for different scenarios
RECOMMENDATION_SCENARIOS = {
    'new_user_cold_start': {
        'description': 'For new users with no history - focus on popular, well-rated content',
        'weights': RecommendationConfig.POPULARITY_FOCUSED_WEIGHTS,
        'view_threshold': RecommendationConfig.HIGH_VIEW_THRESHOLD,
        'diversity_threshold': RecommendationConfig.LOOSE_DIVERSITY_THRESHOLD
    },
    
    'niche_content_discovery': {
        'description': 'For discovering hidden gems and niche content',
        'weights': RecommendationConfig.CONTENT_FOCUSED_WEIGHTS,
        'view_threshold': RecommendationConfig.LOW_VIEW_THRESHOLD,
        'diversity_threshold': RecommendationConfig.STRICT_DIVERSITY_THRESHOLD
    },
    
    'trending_recommendations': {
        'description': 'For showing what\'s currently popular',
        'weights': {
            'semantic_similarity': 0.3,
            'genre_similarity': 0.2,
            'rating_similarity': 0.1,
            'popularity_boost': 0.4
        },
        'view_threshold': RecommendationConfig.DEFAULT_VIEW_THRESHOLD,
        'diversity_threshold': RecommendationConfig.DIVERSITY_THRESHOLD
    }
}
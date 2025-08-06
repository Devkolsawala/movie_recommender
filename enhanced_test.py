# enhanced_test_improved_updated.py
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter, defaultdict
import random

class ImprovedMultiGenreRecommender:
    def __init__(self, media_type='movie', lang_id=None, top_k=10, view_threshold=10000, 
                 enforce_language_matching=True):
        self.media_type = media_type
        self.lang_id = lang_id
        self.top_k = top_k
        self.view_threshold = view_threshold
        self.enforce_language_matching = enforce_language_matching
        
        # Media type mapping (based on ismovie column)
        self.media_type_mapping = {
            'series': 0,      # TV Shows/Series
            'movie': 1,       # Movies  
            'short_drama': 2  # Short Dramas
        }
        
        # Media type display names
        self.media_type_names = {
            0: 'Series/TV Show',
            1: 'Movie',
            2: 'Short Drama'
        }
        
        # Load language mapping
        self.language_map = self._load_language_mapping()
        
        # Optimized weights for better semantic matching
        self.weights = {
            'semantic_similarity': 0.6,   # Increased for better content matching
            'genre_similarity': 0.2,      # Maintained for genre relevance
            'rating_similarity': 0.1,     # Reduced - less important than content
            'popularity_boost': 0.1       # Maintained
        }
        
        # Load model and data
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"🎬 Media Type: {self.media_type_names.get(self.media_type_mapping.get(media_type, 1), media_type)}")
        if lang_id:
            lang_name = self.language_map.get(lang_id, f'Language {lang_id}')
            print(f"🌍 Language Filter: {lang_name} (ID: {lang_id})")
        
        # Load from the main dataset
        self._load_and_filter_data()
        
        # Preprocess data
        self._preprocess_data()
    
    def _load_language_mapping(self):
        """Load language mapping from CSV file"""
        try:
            lang_df = pd.read_csv('language_tbl.csv', header=None)
            lang_df.columns = ['lang_id', 'language_name', 'priority']
            lang_mapping = dict(zip(lang_df['lang_id'], lang_df['language_name']))
            return lang_mapping
        except FileNotFoundError:
            print("⚠️ language_tbl.csv not found. Using default language mapping.")
            return {
                1: "English", 2: "हिन्दी", 3: "ગુજરાતી", 4: "Español", 5: "தமிழ்",
                6: "Other", 7: "മലയാളം", 10: "Français", 12: "ਪੰਜਾਬੀ", 13: "বাংলা",
                14: "मराठी", 15: "ಕನ್ನಡ", 23: "Chinese", 24: "Korean", 25: "తెలుగు",
                49: "عربي", 55: "Deutsch", 67: "Português"
            }
    
    def _load_and_filter_data(self):
        """Load and filter data based on media type and language"""
        try:
            # Load the main dataset
            full_df = pd.read_csv('final_dataset.csv')
            print(f"📊 Loaded dataset with {len(full_df):,} total records")
        except FileNotFoundError:
            print("❌ final_dataset.csv not found. Please run prepare_dataset.py first.")
            raise
        
        # Get media type value
        ismovie_value = self.media_type_mapping.get(self.media_type)
        if ismovie_value is None:
            raise ValueError(f"Invalid media type: {self.media_type}. Must be one of {list(self.media_type_mapping.keys())}")
        
        # Filter by media type
        self.df = full_df[full_df['ismovie'] == ismovie_value].copy()
        
        if self.df.empty:
            raise ValueError(f"No data found for media type '{self.media_type}'")
        
        print(f"🎬 Found {len(self.df):,} {self.media_type} items")
        
        # Filter by language if specified
        if self.lang_id is not None:
            initial_count = len(self.df)
            self.df = self.df[self.df['lang_id'] == self.lang_id].copy()
            filtered_count = len(self.df)
            
            lang_name = self.language_map.get(self.lang_id, f'Language {self.lang_id}')
            print(f"🔍 Filtered from {initial_count:,} to {filtered_count:,} items for {lang_name}")
            
            if filtered_count == 0:
                raise ValueError(f"No {self.media_type} items found for language '{lang_name}' (ID: {self.lang_id})")
        
        # Reset index for consistency
        self.df = self.df.reset_index(drop=True)
        
        # Load or create embeddings
        self._load_embeddings()
        
        print(f"✅ Loaded {len(self.df)} {self.media_type} items")
    
    def _load_embeddings(self):
        """Load or create embeddings for the filtered dataset"""
        # Create a unique identifier for this combination
        lang_suffix = f"_lang{self.lang_id}" if self.lang_id is not None else ""
        embeddings_file = f'{self.media_type}_embeddings{lang_suffix}.pt'
        
        try:
            self.embeddings = torch.load(embeddings_file, weights_only=True)
            print(f"📥 Loaded existing embeddings from {embeddings_file}")
            
            # Verify embeddings match current dataset
            if len(self.embeddings) != len(self.df):
                print(f"⚠️ Embeddings size mismatch ({len(self.embeddings)} vs {len(self.df)}). Regenerating...")
                raise FileNotFoundError("Size mismatch")
                
        except FileNotFoundError:
            print(f"🧠 Generating new embeddings for {self.media_type}...")
            self._generate_embeddings(embeddings_file)
    
    def _generate_embeddings(self, embeddings_file):
        """Generate embeddings for the current dataset"""
        # Create enhanced text for embeddings
        enhanced_texts = []
        for idx, row in self.df.iterrows():
            enhanced_text = self._create_enhanced_text(row)
            enhanced_texts.append(enhanced_text)
            
            if len(enhanced_texts) % 500 == 0:
                print(f"   Created enhanced text for {len(enhanced_texts)} items...")
        
        # Generate embeddings
        print(f"🔄 Encoding {len(enhanced_texts)} texts to embeddings...")
        self.embeddings = self.model.encode(
            enhanced_texts, 
            convert_to_tensor=True, 
            show_progress_bar=True,
            batch_size=16,
            normalize_embeddings=True
        )
        
        # Save embeddings for future use
        torch.save(self.embeddings, embeddings_file)
        print(f"💾 Saved embeddings to {embeddings_file}")
    
    def _create_enhanced_text(self, row):
        """Create enhanced text representation for embeddings with improved semantic matching"""
        components = []
        
        # Title (triple weight for better semantic matching)
        title = str(row['title']) if pd.notna(row['title']) else ''
        if title:
            components.extend([title] * 3)
            # Extract key terms from title for better matching
            title_words = title.lower().split()
            key_terms = [word for word in title_words if len(word) > 3 and word not in ['the', 'and', 'for', 'with']]
            if key_terms:
                components.extend(key_terms)
        
        # Description (quadruple weight - most important for content similarity)
        description = str(row['description']) if pd.notna(row['description']) else ''
        if description and description.lower() != 'nan':
            # Clean and enhance description
            clean_desc = re.sub(r'[^\w\s]', ' ', description.lower())
            components.extend([description] * 4)
            # Extract key phrases from description
            desc_words = clean_desc.split()
            important_words = [word for word in desc_words if len(word) > 4][:10]
            components.extend(important_words)
        
        # Genres with enhanced semantic context
        genres = str(row['genres']) if pd.notna(row['genres']) else ''
        if genres and genres.lower() != 'nan':
            genre_list = [genre.strip() for genre in genres.split(',')]
            # Add genres multiple times for strong genre matching
            components.extend(genre_list * 2)
            
            # Add genre-specific semantic context
            for genre in genre_list:
                genre_lower = genre.lower().strip()
                if genre_lower == 'animation':
                    components.extend(['animated', 'cartoon', 'anime', 'animated film', 'animation movie'])
                elif genre_lower == 'action':
                    components.extend(['fight', 'battle', 'combat', 'adventure', 'intense'])
                elif genre_lower == 'comedy':
                    components.extend(['funny', 'humor', 'laugh', 'entertaining', 'hilarious'])
                elif genre_lower == 'drama':
                    components.extend(['emotional', 'dramatic', 'serious', 'character driven'])
                elif genre_lower == 'thriller':
                    components.extend(['suspense', 'tension', 'mystery', 'exciting'])
                elif genre_lower == 'horror':
                    components.extend(['scary', 'frightening', 'terror', 'supernatural'])
                elif genre_lower == 'romance':
                    components.extend(['love', 'romantic', 'relationship', 'love story'])
                elif genre_lower == 'sci-fi' or genre_lower == 'science fiction':
                    components.extend(['futuristic', 'technology', 'space', 'future'])
                elif genre_lower == 'fantasy':
                    components.extend(['magical', 'mystical', 'supernatural', 'mythical'])
                elif genre_lower == 'musical':
                    components.extend(['music', 'songs', 'singing', 'musical numbers'])
            
            components.append(f"movie with {genres.lower()} themes and elements")
        
        # Enhanced rating context with more semantic meaning
        if pd.notna(row.get('imdb_rating', 0)) and row.get('imdb_rating', 0) > 0:
            rating = row['imdb_rating']
            if rating >= 9.0:
                components.extend(["masterpiece", "outstanding", "exceptional quality", "must watch"])
            elif rating >= 8.5:
                components.extend(["excellent", "highly acclaimed", "top rated", "critically praised"])
            elif rating >= 8.0:
                components.extend(["very good", "well made", "high quality", "recommended"])
            elif rating >= 7.5:
                components.extend(["good quality", "well received", "solid film"])
            elif rating >= 7.0:
                components.extend(["decent", "watchable", "enjoyable"])
            elif rating >= 6.0:
                components.extend(["average", "okay", "moderate quality"])
            else:
                components.extend(["low rated", "poor quality"])
        
        # Media type context with more semantic information
        if 'ismovie' in row and pd.notna(row['ismovie']):
            if row['ismovie'] == 1:  # Movie
                components.extend(["feature film", "movie", "cinema", "theatrical release"])
            elif row['ismovie'] == 0:  # Series
                components.extend(["tv series", "television show", "episodic", "series"])
            elif row['ismovie'] == 2:  # Short Drama
                components.extend(["short film", "short drama", "brief story"])
        
        # Add character/theme extraction from title for better semantic matching
        title_lower = title.lower() if title else ''
        if any(word in title_lower for word in ['slayer', 'demon', 'hunter']):
            components.extend(['supernatural fighter', 'demon hunter', 'supernatural combat', 'monster slayer'])
        if any(word in title_lower for word in ['anime', 'manga', 'otaku']):
            components.extend(['japanese animation', 'anime style', 'manga adaptation'])
        if any(word in title_lower for word in ['war', 'battle', 'fight']):
            components.extend(['warfare', 'combat', 'military', 'battle scenes'])
        
        return ' '.join(components)
    
    def _preprocess_data(self):
        """Enhanced preprocessing with better handling"""
        # Handle missing values
        self.df['genres'] = self.df['genres'].fillna('')
        self.df['description'] = self.df['description'].fillna('')
        self.df['imdb_rating'] = self.df['imdb_rating'].fillna(self.df['imdb_rating'].mean())
        
        # Ensure required columns exist
        if 'views' not in self.df.columns:
            self.df['views'] = 1000  # Default view count
        
        # Normalize views with log transformation for better distribution
        self.df['log_views'] = np.log1p(self.df['views'])
        scaler = MinMaxScaler()
        self.df['normalized_views'] = scaler.fit_transform(self.df[['log_views']])
        
        # Create improved genre vectors (using old script approach)
        self._create_improved_genre_vectors()
        
        # Create rating buckets for better similarity
        self._create_rating_buckets()
    
    def _get_clicked_languages(self, clicked_ids):
        """Get the languages of clicked items"""
        clicked_df = self.df[self.df['id'].isin(clicked_ids)]
        languages = clicked_df['lang_id'].unique()
        return languages
    
    def _create_improved_genre_vectors(self):
        """Create weighted genre vectors with better similarity calculation (from old script)"""
        # Extract all unique genres
        all_genres = set()
        for genres in self.df['genres']:
            if genres:
                all_genres.update([g.strip().lower() for g in genres.split(',')])
        
        self.all_genres = sorted(list(all_genres))
        
        # Create simple binary vectors (no IDF weighting to reduce bias) - from old script
        genre_vectors = []
        for genres in self.df['genres']:
            vector = [0.0] * len(self.all_genres)
            if genres:
                movie_genres = [g.strip().lower() for g in genres.split(',')]
                for i, genre in enumerate(self.all_genres):
                    if genre in movie_genres:
                        vector[i] = 1.0  # Simple binary encoding
            genre_vectors.append(vector)
        
        self.genre_vectors = np.array(genre_vectors)
    
    def _create_rating_buckets(self):
        """Create rating similarity buckets for more nuanced comparison"""
        def get_rating_bucket(rating):
            if rating >= 8.5: return 'excellent'
            elif rating >= 7.5: return 'very_good' 
            elif rating >= 6.5: return 'good'
            elif rating >= 5.5: return 'average'
            elif rating >= 4.0: return 'below_average'
            else: return 'poor'
        
        self.df['rating_bucket'] = self.df['imdb_rating'].apply(get_rating_bucket)
    
    def _extract_unique_genres_from_clicked(self, clicked_ids):
        """Extract unique genres from clicked items (from old script)"""
        clicked_df = self.df[self.df['id'].isin(clicked_ids)]
        all_genres = set()
        genre_groups = []
        
        for _, row in clicked_df.iterrows():
            if row['genres']:
                movie_genres = set(g.strip().lower() for g in row['genres'].split(','))
                genre_groups.append(movie_genres)
                all_genres.update(movie_genres)
        
        return list(all_genres), genre_groups
    
    def _calculate_enhanced_genre_similarity(self, clicked_ids):
        """Enhanced genre similarity with better semantic understanding"""
        clicked_indices = self.df[self.df['id'].isin(clicked_ids)].index.tolist()
        clicked_genre_vectors = self.genre_vectors[clicked_indices]
        
        # Get genre context from clicked items
        clicked_df = self.df[self.df['id'].isin(clicked_ids)]
        clicked_genres_flat = []
        for _, row in clicked_df.iterrows():
            if row['genres']:
                clicked_genres_flat.extend([g.strip().lower() for g in row['genres'].split(',')])
        
        similarities = []
        for idx, candidate_vector in enumerate(self.genre_vectors):
            candidate_row = self.df.iloc[idx]
            candidate_genres = []
            if candidate_row['genres']:
                candidate_genres = [g.strip().lower() for g in candidate_row['genres'].split(',')]
            
            max_similarity = 0.0
            
            # Calculate similarity with each clicked item
            for clicked_vector in clicked_genre_vectors:
                if np.linalg.norm(clicked_vector) == 0 or np.linalg.norm(candidate_vector) == 0:
                    sim = 0.0
                else:
                    # Use Jaccard similarity
                    intersection = np.sum(np.minimum(clicked_vector, candidate_vector))
                    union = np.sum(np.maximum(clicked_vector, candidate_vector))
                    jaccard_sim = intersection / union if union > 0 else 0.0
                    
                    # Add semantic bonus for related genres
                    semantic_bonus = 0.0
                    for clicked_genre in clicked_genres_flat:
                        for candidate_genre in candidate_genres:
                            # Special bonuses for semantically related genres
                            if clicked_genre == 'animation' and candidate_genre in ['anime', 'cartoon', 'family', 'fantasy']:
                                semantic_bonus += 0.3
                            elif clicked_genre == 'action' and candidate_genre in ['adventure', 'thriller', 'war']:
                                semantic_bonus += 0.2
                            elif clicked_genre == 'comedy' and candidate_genre in ['family', 'romance']:
                                semantic_bonus += 0.2
                            elif clicked_genre == 'drama' and candidate_genre in ['romance', 'thriller']:
                                semantic_bonus += 0.15
                            elif clicked_genre == candidate_genre:
                                semantic_bonus += 0.5  # Exact match bonus
                    
                    sim = min(1.0, jaccard_sim + (semantic_bonus * 0.3))  # Apply semantic bonus
                
                max_similarity = max(max_similarity, sim)
            
            similarities.append(max_similarity)
        
        return np.array(similarities)
    
    def _calculate_semantic_similarity(self, clicked_ids):
        """Enhanced semantic similarity with better averaging and thresholds"""
        clicked_indices = self.df[self.df['id'].isin(clicked_ids)].index.tolist()
        clicked_embeddings = self.embeddings[clicked_indices]
        
        # Use weighted average instead of just maximum for better accuracy
        similarities = []
        for candidate_embedding in self.embeddings:
            similarities_to_clicked = []
            
            for clicked_embedding in clicked_embeddings:
                sim = util.cos_sim(candidate_embedding, clicked_embedding)[0][0].item()
                similarities_to_clicked.append(sim)
            
            # Use weighted average: 70% max similarity + 30% average similarity
            if similarities_to_clicked:
                max_sim = max(similarities_to_clicked)
                avg_sim = sum(similarities_to_clicked) / len(similarities_to_clicked)
                weighted_sim = 0.7 * max_sim + 0.3 * avg_sim
                similarities.append(weighted_sim)
            else:
                similarities.append(0.0)
        
        return np.array(similarities)
    
    def _calculate_improved_rating_similarity(self, clicked_ids):
        """Improved rating similarity with bucket-based approach (from old script)"""
        clicked_df = self.df[self.df['id'].isin(clicked_ids)]
        clicked_ratings = clicked_df['imdb_rating'].tolist()
        
        similarities = []
        for _, row in self.df.iterrows():
            candidate_rating = row['imdb_rating']
            
            # Calculate similarity to best matching clicked item
            best_similarity = 0.0
            for clicked_rating in clicked_ratings:
                rating_diff = abs(candidate_rating - clicked_rating)
                similarity = max(0, 1 - (rating_diff / 5.0))
                best_similarity = max(best_similarity, similarity)
            
            similarities.append(best_similarity)
        
        return np.array(similarities)
    
    def _add_content_similarity_boost(self, clicked_ids, candidate_id, base_score):
        """Add boost for content with similar themes, characters, or franchises"""
        clicked_df = self.df[self.df['id'].isin(clicked_ids)]
        candidate_row = self.df[self.df['id'] == candidate_id].iloc[0]
        
        boost = 0.0
        
        # Get text content for analysis
        clicked_texts = []
        for _, row in clicked_df.iterrows():
            text_content = f"{row['title']} {row['description'] if pd.notna(row['description']) else ''}"
            clicked_texts.append(text_content.lower())
        
        candidate_text = f"{candidate_row['title']} {candidate_row['description'] if pd.notna(candidate_row['description']) else ''}".lower()
        
        # Check for thematic similarities
        anime_keywords = ['anime', 'manga', 'demon', 'slayer', 'hunter', 'supernatural', 'japanese', 'animation']
        action_keywords = ['fight', 'battle', 'combat', 'warrior', 'sword', 'martial', 'hero']
        fantasy_keywords = ['magic', 'mystical', 'powers', 'supernatural', 'fantasy', 'mythical']
        
        # Calculate keyword overlap boost
        for clicked_text in clicked_texts:
            # Anime/Animation boost
            clicked_anime_count = sum(1 for keyword in anime_keywords if keyword in clicked_text)
            candidate_anime_count = sum(1 for keyword in anime_keywords if keyword in candidate_text)
            if clicked_anime_count > 0 and candidate_anime_count > 0:
                boost += 0.1 * min(clicked_anime_count, candidate_anime_count)
            
            # Action/Combat boost
            clicked_action_count = sum(1 for keyword in action_keywords if keyword in clicked_text)
            candidate_action_count = sum(1 for keyword in action_keywords if keyword in candidate_text)
            if clicked_action_count > 0 and candidate_action_count > 0:
                boost += 0.05 * min(clicked_action_count, candidate_action_count)
            
            # Fantasy boost
            clicked_fantasy_count = sum(1 for keyword in fantasy_keywords if keyword in clicked_text)
            candidate_fantasy_count = sum(1 for keyword in fantasy_keywords if keyword in candidate_text)
            if clicked_fantasy_count > 0 and candidate_fantasy_count > 0:
                boost += 0.05 * min(clicked_fantasy_count, candidate_fantasy_count)
        
        return min(boost, 0.2)  # Cap the boost
    
    def _apply_language_filter(self, recommendations, clicked_ids):
        """Filter recommendations to match clicked item languages"""
        if not self.enforce_language_matching:
            return recommendations
        
        clicked_languages = self._get_clicked_languages(clicked_ids)
        
        # Filter recommendations to only include items in the same languages
        filtered_recommendations = []
        for rec in recommendations:
            if rec['lang_id'] in clicked_languages:
                filtered_recommendations.append(rec)
        
        return filtered_recommendations
    
    def _split_recommendations_by_genre(self, recommendations, clicked_ids):
        """Split recommendations based on clicked item genres (from old script)"""
        unique_genres, genre_groups = self._extract_unique_genres_from_clicked(clicked_ids)
        num_unique_genres = len(genre_groups)
        
        if num_unique_genres <= 1:
            return recommendations[:self.top_k]
        
        # Calculate split ratios
        if num_unique_genres == 2:
            split_ratios = [5, 5]
        elif num_unique_genres == 3:
            split_ratios = [3, 3, 4]
        else:
            # For more than 3 genres, distribute evenly
            base_count = self.top_k // num_unique_genres
            remainder = self.top_k % num_unique_genres
            split_ratios = [base_count] * num_unique_genres
            for i in range(remainder):
                split_ratios[i] += 1
        
        # Group recommendations by their best matching genre group
        genre_recommendations = [[] for _ in range(num_unique_genres)]
        unmatched_recommendations = []
        
        for rec in recommendations:
            rec_idx = self.df[self.df['id'] == rec['id']].index[0]
            rec_genres = set(g.strip().lower() for g in str(self.df.iloc[rec_idx]['genres']).split(','))
            
            best_match_group = -1
            best_overlap = 0
            
            for i, genre_group in enumerate(genre_groups):
                overlap = len(rec_genres.intersection(genre_group))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match_group = i
            
            if best_match_group >= 0 and best_overlap > 0:
                genre_recommendations[best_match_group].append(rec)
            else:
                unmatched_recommendations.append(rec)
        
        # Fill each genre group up to its target count
        final_recommendations = []
        used_recommendations = set()
        
        for i, target_count in enumerate(split_ratios):
            genre_recs = genre_recommendations[i]
            added_count = 0
            
            for rec in genre_recs:
                if rec['id'] not in used_recommendations and added_count < target_count:
                    final_recommendations.append(rec)
                    used_recommendations.add(rec['id'])
                    added_count += 1
        
        # Fill remaining slots with unmatched or remaining recommendations
        remaining_slots = self.top_k - len(final_recommendations)
        all_remaining = unmatched_recommendations + [
            rec for rec in recommendations 
            if rec['id'] not in used_recommendations
        ]
        
        for rec in all_remaining[:remaining_slots]:
            if rec['id'] not in used_recommendations:
                final_recommendations.append(rec)
                used_recommendations.add(rec['id'])
        
        return final_recommendations
    
    def get_recommendations(self, clicked_ids, apply_genre_split=True):
        """Get improved recommendations with multi-genre support"""
        clicked_df = self.df[self.df['id'].isin(clicked_ids)]
        
        if clicked_df.empty:
            print("⚠️ No clicked items found in the current dataset.")
            return []
        
        # Get clicked languages for matching
        clicked_languages = self._get_clicked_languages(clicked_ids)
        
        # Show user interaction info
        unique_genres, genre_groups = self._extract_unique_genres_from_clicked(clicked_ids)
        print(f"\n🎯 User interacted with {len(clicked_ids)} {self.media_type_names.get(self.media_type_mapping.get(self.media_type), self.media_type)}(s)")
        
        # Display clicked items with details
        for _, row in clicked_df.iterrows():
            lang_name = self.language_map.get(row['lang_id'], f"Lang {row['lang_id']}")
            print(f"📊 Clicked: {row['title']} (ID: {row['id']}) | {lang_name} | {row['genres']}")
        
        print(f"🎭 Detected {len(genre_groups)} different genre groups")
        
        # Show language matching info
        if self.enforce_language_matching:
            lang_names = [self.language_map.get(lang_id, f'Lang {lang_id}') for lang_id in clicked_languages]
            print(f"🔒 Language matching enabled: Will recommend only {', '.join(lang_names)} content")
        
        # Calculate similarity scores using enhanced methods
        semantic_scores = self._calculate_semantic_similarity(clicked_ids)
        genre_scores = self._calculate_enhanced_genre_similarity(clicked_ids)
        rating_scores = self._calculate_improved_rating_similarity(clicked_ids)
        
        # Prepare recommendations
        recommendations = []
        for idx, (semantic_score, genre_score, rating_score) in enumerate(
            zip(semantic_scores, genre_scores, rating_scores)
        ):
            item = self.df.iloc[idx]
            item_id = item['id']
            views = item.get('views', 1000)
            
            # Skip clicked items and low-view items
            if item_id in clicked_ids or views < self.view_threshold:
                continue
            
            # Apply language filtering if enabled
            if self.enforce_language_matching and item['lang_id'] not in clicked_languages:
                continue
            
            # Calculate content similarity boost
            content_boost = self._add_content_similarity_boost(clicked_ids, item_id, 0)
            
            # Calculate weighted final score
            popularity_score = item['normalized_views']
            
            base_score = (
                self.weights['semantic_similarity'] * semantic_score +
                self.weights['genre_similarity'] * genre_score +
                self.weights['rating_similarity'] * rating_score +
                self.weights['popularity_boost'] * popularity_score
            )
            
            final_score = base_score + content_boost
            
            recommendations.append({
                'id': item_id,
                'title': item['title'],
                'final_score': final_score,
                'semantic_score': semantic_score,
                'genre_score': genre_score,
                'rating_score': rating_score,
                'popularity_score': popularity_score,
                'content_boost': content_boost,
                'views': views,
                'imdb_rating': item['imdb_rating'],
                'genres': item['genres'],
                'lang_id': item.get('lang_id', 'Unknown'),
                'language_name': self.language_map.get(item.get('lang_id'), 'Unknown'),
                'ismovie': item['ismovie'],
                'media_type_name': self.media_type_names.get(item['ismovie'], 'Unknown')
            })
        
        # Sort by final score
        recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Apply genre splitting if requested and multiple genres detected
        if apply_genre_split and len(genre_groups) > 1:
            print(f"🎭 Applying genre diversity to ensure balanced recommendations...")
            final_recommendations = self._split_recommendations_by_genre(
                recommendations[:self.top_k * 3], clicked_ids  # Get more candidates for splitting
            )
        else:
            final_recommendations = recommendations[:self.top_k]
        
        return final_recommendations
    
    def print_detailed_recommendations(self, recommendations, clicked_ids):
        """Print detailed recommendation results"""
        clicked_languages = self._get_clicked_languages(clicked_ids)
        unique_genres, genre_groups = self._extract_unique_genres_from_clicked(clicked_ids)
        
        media_type_name = self.media_type_names.get(self.media_type_mapping.get(self.media_type), self.media_type)
        
        print(f"\n🎬 Top {len(recommendations)} {media_type_name} Recommendations:")
        
        # Show filtering info
        if self.enforce_language_matching:
            lang_names = [self.language_map.get(lang_id, f'Lang {lang_id}') for lang_id in clicked_languages]
            print(f"🔒 Language Filter: Matching clicked items ({', '.join(lang_names)})")
        
        if len(genre_groups) > 1:
            print(f"📊 Genre Split Strategy: {len(genre_groups)} genre groups detected")
            for i, group in enumerate(genre_groups):
                print(f"   Group {i+1}: {list(group)}")
        
        print("=" * 100)
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']} (ID: {rec['id']})")
            print(f"   🎬 Media Type: {rec['media_type_name']}")
            print(f"   🌍 Language: {rec['language_name']}")
            print(f"   📊 Final Score: {rec['final_score']:.4f}")
            print(f"   🎭 Genres: {rec['genres']}")
            print(f"   ⭐ IMDB Rating: {rec['imdb_rating']:.1f}")
            print(f"   👀 Views: {rec['views']:,}")
            
            # Score breakdown
            content_info = f" (+{rec['content_boost']:.3f} content)" if rec['content_boost'] > 0 else ""
            print(f"   📈 Score Breakdown: Semantic={rec['semantic_score']:.3f}, "
                  f"Genre={rec['genre_score']:.3f}, Rating={rec['rating_score']:.3f}, "
                  f"Popularity={rec['popularity_score']:.3f}{content_info}")
            print("-" * 100)
    
    def print_simple_recommendations(self, recommendations):
        """Print recommendations in simple format"""
        media_type_name = self.media_type_names.get(self.media_type_mapping.get(self.media_type), self.media_type)
        
        print(f"\n🎬 Simple Format - Top {len(recommendations)} {media_type_name} Recommendations:")
        
        for i, rec in enumerate(recommendations, 1):
            lang_name = rec['language_name']
            print(f"{i}. {rec['title']} (ID: {rec['id']}) | {lang_name} | Score: {rec['final_score']:.4f} | Views: {rec['views']:,}")
    
    def print_genre_analysis(self, recommendations, clicked_ids):
        """Print genre analysis of recommendations vs clicked items"""
        print(f"\n📊 Genre Analysis:")
        print("=" * 60)
        
        # Clicked items genres
        clicked_df = self.df[self.df['id'].isin(clicked_ids)]
        print("🎯 Clicked Items Genres:")
        for _, row in clicked_df.iterrows():
            print(f"  {row['title']} (ID: {row['id']}) → {row['genres']}")
        
        # Recommended items genre distribution
        print("\n🎭 Recommended Items Genre Distribution:")
        genre_counts = {}
        for rec in recommendations:
            genres = rec['genres'].lower() if rec['genres'] else ''
            for genre in genres.split(','):
                genre = genre.strip()
                if genre:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        if genre_counts:
            sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
            for genre, count in sorted_genres:
                percentage = (count / len(recommendations)) * 100
                print(f"  {genre.title()}: {count} items ({percentage:.1f}%)")
        
        # Language distribution
        print("\n🌍 Language Distribution in Recommendations:")
        lang_counts = {}
        for rec in recommendations:
            lang = rec['language_name']
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(recommendations)) * 100
            print(f"  {lang}: {count} items ({percentage:.1f}%)")
    
    def validate_clicked_ids(self, clicked_ids):
        """Validate that clicked IDs exist in current dataset"""
        print(f"🔍 Validating clicked IDs...")
        
        valid_ids = []
        invalid_ids = []
        
        for clicked_id in clicked_ids:
            if clicked_id in self.df['id'].values:
                valid_ids.append(clicked_id)
            else:
                invalid_ids.append(clicked_id)
        
        if valid_ids:
            print(f"✅ Valid IDs: {valid_ids}")
        
        if invalid_ids:
            print(f"❌ Invalid IDs: {invalid_ids}")
            print(f"💡 Suggested valid IDs from current dataset:")
            sample_ids = self.df['id'].head(10).tolist()
            for sample_id in sample_ids:
                sample_row = self.df[self.df['id'] == sample_id].iloc[0]
                lang_name = self.language_map.get(sample_row['lang_id'], f"Lang {sample_row['lang_id']}")
                print(f"   ID: {sample_id} - {sample_row['title']} ({lang_name}) - {sample_row['genres']}")
        
        return valid_ids


# Main execution
if __name__ == "__main__":
    # Configuration - Update these with your actual data
    CLICKED_IDS = [138]  # Replace with actual IDs from your dataset
    MEDIA_TYPE = 'series'  # Options: 'movie', 'series', 'short_drama'
    LANG_ID = None  # Set to specific language ID or None for all
    ENFORCE_LANGUAGE_MATCHING = True  # Enable language matching
    APPLY_GENRE_SPLIT = True  # Enable genre diversity
    
    print("🚀 Starting Enhanced Multi-Genre Recommender System")
    print("=" * 80)
    
    try:
        # Create improved recommender
        recommender = ImprovedMultiGenreRecommender(
            media_type=MEDIA_TYPE,
            lang_id=LANG_ID,
            top_k=10,
            view_threshold=10000,
            enforce_language_matching=ENFORCE_LANGUAGE_MATCHING
        )
        
        # Validate clicked IDs first
        valid_clicked_ids = recommender.validate_clicked_ids(CLICKED_IDS)
        
        if not valid_clicked_ids:
            print("\n❌ No valid clicked IDs found. Please update CLICKED_IDS with valid IDs from your dataset.")
        else:
            # Get recommendations
            recommendations = recommender.get_recommendations(
                clicked_ids=valid_clicked_ids,
                apply_genre_split=APPLY_GENRE_SPLIT
            )
            
            if recommendations:
                # Print detailed results
                recommender.print_detailed_recommendations(recommendations, valid_clicked_ids)
                
                # Print simple format
                recommender.print_simple_recommendations(recommendations)
                
                # Print genre analysis
                recommender.print_genre_analysis(recommendations, valid_clicked_ids)
                
                print(f"\n✅ Successfully generated {len(recommendations)} recommendations!")
                
            else:
                print("\n❌ No recommendations generated. Try lowering view_threshold or checking your data.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure final_dataset.csv exists")
        print("2. Check that your CLICKED_IDS exist in the dataset")
        print("3. Verify the media_type and lang_id values are correct")
        print("4. Ensure you have data for the selected media type and language combination")













































































#old code  

# enhanced_test_updated.py
# import pandas as pd
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer, util
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics.pairwise import cosine_similarity
# import re
# from collections import Counter, defaultdict

# class ImprovedMultiGenreRecommender:
#     def __init__(self, media_type='movie', lang_id=None, top_k=10, view_threshold=10000):
#         self.media_type = media_type
#         self.lang_id = lang_id
#         self.top_k = top_k
#         self.view_threshold = view_threshold
        
#         # Media type mapping
#         self.media_type_mapping = {
#             'trailer': 0,
#             'movie': 1,
#             'short_drama': 2
#         }
        
#         # Adjusted weights to reduce genre bias
#         self.weights = {
#             'semantic_similarity': 0.5,   # Increased for content-based similarity
#             'genre_similarity': 0.2,      # Reduced to prevent genre dominance
#             'rating_similarity': 0.2,     # Maintained
#             'popularity_boost': 0.1       # Maintained
#         }
        
#         # Load model and data
#         self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
#         print(f"📂 Loading data for media type: {media_type}")
#         if lang_id:
#             print(f"🌍 Filtering by language ID: {lang_id}")
        
#         # Load from the main dataset instead of separate files
#         self._load_and_filter_data()
        
#         # Preprocess data
#         self._preprocess_data()
    
#     def _load_and_filter_data(self):
#         """Load and filter data based on media type and language"""
#         # Load the main dataset
#         full_df = pd.read_csv('final_dataset.csv')
        
#         # Filter by media type
#         ismovie_value = self.media_type_mapping.get(self.media_type)
#         if ismovie_value is None:
#             raise ValueError(f"Invalid media type: {self.media_type}. Must be one of {list(self.media_type_mapping.keys())}")
        
#         self.df = full_df[full_df['ismovie'] == ismovie_value].copy()
        
#         # Filter by language if specified
#         if self.lang_id is not None:
#             initial_count = len(self.df)
#             self.df = self.df[self.df['lang_id'] == self.lang_id].copy()
#             filtered_count = len(self.df)
#             print(f"🔍 Filtered from {initial_count} to {filtered_count} items for language {self.lang_id}")
        
#         if self.df.empty:
#             raise ValueError(f"No data found for media type '{self.media_type}' and language '{self.lang_id}'")
        
#         # Reset index for consistency
#         self.df = self.df.reset_index(drop=True)
        
#         # Load or create embeddings
#         self._load_embeddings()
        
#         print(f"✅ Loaded {len(self.df)} {self.media_type} items")
    
#     def _load_embeddings(self):
#         """Load or create embeddings for the filtered dataset"""
#         # Create a unique identifier for this combination
#         lang_suffix = f"_lang{self.lang_id}" if self.lang_id is not None else ""
#         embeddings_file = f'{self.media_type}_embeddings{lang_suffix}.pt'
        
#         try:
#             self.embeddings = torch.load(embeddings_file, weights_only=True)
#             print(f"📥 Loaded existing embeddings from {embeddings_file}")
            
#             # Verify embeddings match current dataset
#             if len(self.embeddings) != len(self.df):
#                 print(f"⚠️ Embeddings size mismatch. Regenerating...")
#                 raise FileNotFoundError("Size mismatch")
                
#         except FileNotFoundError:
#             print(f"🧠 Generating new embeddings for {self.media_type}...")
#             self._generate_embeddings(embeddings_file)
    
#     def _generate_embeddings(self, embeddings_file):
#         """Generate embeddings for the current dataset"""
#         # Create enhanced text for embeddings
#         enhanced_texts = []
#         for idx, row in self.df.iterrows():
#             enhanced_text = self._create_enhanced_text(row)
#             enhanced_texts.append(enhanced_text)
        
#         # Generate embeddings
#         self.embeddings = self.model.encode(
#             enhanced_texts, 
#             convert_to_tensor=True, 
#             show_progress_bar=True,
#             batch_size=16,
#             normalize_embeddings=True
#         )
        
#         # Save embeddings for future use
#         torch.save(self.embeddings, embeddings_file)
#         print(f"💾 Saved embeddings to {embeddings_file}")
    
#     def _create_enhanced_text(self, row):
#         """Create enhanced text representation for embeddings"""
#         components = []
        
#         # Title
#         title = str(row['title']) if pd.notna(row['title']) else ''
#         if title:
#             components.extend([title] * 2)
        
#         # Description
#         description = str(row['description']) if pd.notna(row['description']) else ''
#         if description:
#             components.extend([description] * 2)
        
#         # Genres
#         genres = str(row['genres']) if pd.notna(row['genres']) else ''
#         if genres:
#             genre_list = [genre.strip() for genre in genres.split(',')]
#             components.extend(genre_list)
#             components.append(f"film with {genres.lower()} elements")
        
#         # Rating context
#         if pd.notna(row['imdb_rating']):
#             rating = row['imdb_rating']
#             if rating >= 8.5:
#                 components.append("exceptional highly acclaimed masterpiece")
#             elif rating >= 8.0:
#                 components.append("excellent highly rated film")
#             elif rating >= 7.5:
#                 components.append("very good well received film")
#             elif rating >= 7.0:
#                 components.append("good quality rated film")
#             elif rating >= 6.5:
#                 components.append("decent watchable film")
#             elif rating >= 6.0:
#                 components.append("average rated film")
#             else:
#                 components.append("low rated film")
        
#         return ' '.join(components)
    
#     def _preprocess_data(self):
#         """Enhanced preprocessing with better handling"""
#         # Handle missing values
#         self.df['genres'] = self.df['genres'].fillna('')
#         self.df['description'] = self.df['description'].fillna('')
#         self.df['imdb_rating'] = self.df['imdb_rating'].fillna(self.df['imdb_rating'].mean())
        
#         # Ensure required columns exist
#         if 'views' not in self.df.columns:
#             self.df['views'] = 1000  # Default view count
        
#         # Normalize views with log transformation for better distribution
#         self.df['log_views'] = np.log1p(self.df['views'])
#         scaler = MinMaxScaler()
#         self.df['normalized_views'] = scaler.fit_transform(self.df[['log_views']])
        
#         # Create improved genre vectors
#         self._create_improved_genre_vectors()
        
#         # Create rating buckets for better similarity
#         self._create_rating_buckets()
    
#     def _extract_unique_genres_from_clicked(self, clicked_ids):
#         """Extract unique genres from clicked items"""
#         clicked_df = self.df[self.df['id'].isin(clicked_ids)]
#         all_genres = set()
#         genre_groups = []
        
#         for _, row in clicked_df.iterrows():
#             if row['genres']:
#                 movie_genres = set(g.strip().lower() for g in row['genres'].split(','))
#                 genre_groups.append(movie_genres)
#                 all_genres.update(movie_genres)
        
#         return list(all_genres), genre_groups
    
#     def _create_improved_genre_vectors(self):
#         """Create weighted genre vectors with better similarity calculation"""
#         # Extract all unique genres
#         all_genres = set()
#         for genres in self.df['genres']:
#             if genres:
#                 all_genres.update([g.strip().lower() for g in genres.split(',')])
        
#         self.all_genres = sorted(list(all_genres))
        
#         # Create simple binary vectors (no IDF weighting to reduce bias)
#         genre_vectors = []
#         for genres in self.df['genres']:
#             vector = [0.0] * len(self.all_genres)
#             if genres:
#                 movie_genres = [g.strip().lower() for g in genres.split(',')]
#                 for i, genre in enumerate(self.all_genres):
#                     if genre in movie_genres:
#                         vector[i] = 1.0  # Simple binary encoding
#             genre_vectors.append(vector)
        
#         self.genre_vectors = np.array(genre_vectors)
    
#     def _create_rating_buckets(self):
#         """Create rating similarity buckets for more nuanced comparison"""
#         def get_rating_bucket(rating):
#             if rating >= 8.5: return 'excellent'
#             elif rating >= 7.5: return 'very_good' 
#             elif rating >= 6.5: return 'good'
#             elif rating >= 5.5: return 'average'
#             elif rating >= 4.0: return 'below_average'
#             else: return 'poor'
        
#         self.df['rating_bucket'] = self.df['imdb_rating'].apply(get_rating_bucket)
    
#     def _calculate_multi_genre_similarity(self, clicked_ids):
#         """Calculate genre similarity that handles multiple different genres"""
#         clicked_indices = self.df[self.df['id'].isin(clicked_ids)].index.tolist()
#         clicked_genre_vectors = self.genre_vectors[clicked_indices]
        
#         # Instead of averaging, we'll calculate maximum similarity to any clicked item
#         similarities = []
#         for candidate_vector in self.genre_vectors:
#             max_similarity = 0.0
            
#             for clicked_vector in clicked_genre_vectors:
#                 if np.linalg.norm(clicked_vector) == 0 or np.linalg.norm(candidate_vector) == 0:
#                     sim = 0.0
#                 else:
#                     # Use Jaccard similarity for better genre matching
#                     intersection = np.sum(np.minimum(clicked_vector, candidate_vector))
#                     union = np.sum(np.maximum(clicked_vector, candidate_vector))
#                     sim = intersection / union if union > 0 else 0.0
                
#                 max_similarity = max(max_similarity, sim)
            
#             similarities.append(max_similarity)
        
#         return np.array(similarities)
    
#     def _calculate_improved_rating_similarity(self, clicked_ids):
#         """Improved rating similarity with bucket-based approach"""
#         clicked_df = self.df[self.df['id'].isin(clicked_ids)]
#         clicked_ratings = clicked_df['imdb_rating'].tolist()
        
#         similarities = []
#         for _, row in self.df.iterrows():
#             candidate_rating = row['imdb_rating']
            
#             # Calculate similarity to best matching clicked item
#             best_similarity = 0.0
#             for clicked_rating in clicked_ratings:
#                 rating_diff = abs(candidate_rating - clicked_rating)
#                 similarity = max(0, 1 - (rating_diff / 5.0))
#                 best_similarity = max(best_similarity, similarity)
            
#             similarities.append(best_similarity)
        
#         return np.array(similarities)
    
#     def _calculate_semantic_similarity(self, clicked_ids):
#         """Calculate semantic similarity using maximum similarity approach"""
#         clicked_indices = self.df[self.df['id'].isin(clicked_ids)].index.tolist()
#         clicked_embeddings = self.embeddings[clicked_indices]
        
#         # Calculate maximum similarity to any clicked item
#         max_similarities = []
#         for candidate_embedding in self.embeddings:
#             max_sim = 0.0
#             for clicked_embedding in clicked_embeddings:
#                 sim = util.cos_sim(candidate_embedding, clicked_embedding)[0][0].item()
#                 max_sim = max(max_sim, sim)
#             max_similarities.append(max_sim)
        
#         return np.array(max_similarities)
    
#     def _calculate_franchise_boost(self, clicked_ids, candidate_title):
#         """Boost for movies in the same franchise/series"""
#         clicked_df = self.df[self.df['id'].isin(clicked_ids)]
#         boost = 0.0
        
#         for clicked_title in clicked_df['title']:
#             clicked_words = set(clicked_title.lower().split())
#             candidate_words = set(candidate_title.lower().split())
            
#             common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an']
#             significant_clicked = clicked_words - set(common_words)
#             significant_candidate = candidate_words - set(common_words)
            
#             overlap = len(significant_clicked.intersection(significant_candidate))
#             if overlap >= 2:
#                 boost += 0.05 * overlap  # Reduced franchise boost
        
#         return min(boost, 0.15)  # Reduced cap
    
#     def _split_recommendations_by_genre(self, recommendations, clicked_ids):
#         """Split recommendations based on clicked item genres"""
#         unique_genres, genre_groups = self._extract_unique_genres_from_clicked(clicked_ids)
#         num_unique_genres = len(genre_groups)
        
#         if num_unique_genres <= 1:
#             return recommendations[:self.top_k]
        
#         # Calculate split ratios
#         if num_unique_genres == 2:
#             split_ratios = [5, 5]
#         elif num_unique_genres == 3:
#             split_ratios = [3, 3, 4]
#         else:
#             # For more than 3 genres, distribute evenly
#             base_count = self.top_k // num_unique_genres
#             remainder = self.top_k % num_unique_genres
#             split_ratios = [base_count] * num_unique_genres
#             for i in range(remainder):
#                 split_ratios[i] += 1
        
#         # Group recommendations by their best matching genre group
#         genre_recommendations = [[] for _ in range(num_unique_genres)]
#         unmatched_recommendations = []
        
#         for rec in recommendations:
#             rec_idx = self.df[self.df['title'] == rec['title']].index[0]
#             rec_genres = set(g.strip().lower() for g in str(self.df.iloc[rec_idx]['genres']).split(','))
            
#             best_match_group = -1
#             best_overlap = 0
            
#             for i, genre_group in enumerate(genre_groups):
#                 overlap = len(rec_genres.intersection(genre_group))
#                 if overlap > best_overlap:
#                     best_overlap = overlap
#                     best_match_group = i
            
#             if best_match_group >= 0 and best_overlap > 0:
#                 genre_recommendations[best_match_group].append(rec)
#             else:
#                 unmatched_recommendations.append(rec)
        
#         # Fill each genre group up to its target count
#         final_recommendations = []
#         used_recommendations = set()
        
#         for i, target_count in enumerate(split_ratios):
#             genre_recs = genre_recommendations[i]
#             added_count = 0
            
#             for rec in genre_recs:
#                 if rec['title'] not in used_recommendations and added_count < target_count:
#                     final_recommendations.append(rec)
#                     used_recommendations.add(rec['title'])
#                     added_count += 1
        
#         # Fill remaining slots with unmatched or remaining recommendations
#         remaining_slots = self.top_k - len(final_recommendations)
#         all_remaining = unmatched_recommendations + [
#             rec for rec in recommendations 
#             if rec['title'] not in used_recommendations
#         ]
        
#         for rec in all_remaining[:remaining_slots]:
#             if rec['title'] not in used_recommendations:
#                 final_recommendations.append(rec)
#                 used_recommendations.add(rec['title'])
        
#         return final_recommendations
    
#     def get_recommendations(self, clicked_ids, apply_genre_split=True, diversity_threshold=0.75):
#         """Get improved recommendations with multi-genre support"""
#         clicked_df = self.df[self.df['id'].isin(clicked_ids)]
        
#         if clicked_df.empty:
#             print("⚠️ No clicked items found. Check your CLICKED_IDS.")
#             return []
        
#         # Verify clicked items are in the same media type and language
#         clicked_media_types = clicked_df['ismovie'].unique()
#         clicked_lang_ids = clicked_df['lang_id'].unique() if 'lang_id' in clicked_df.columns else [None]
        
#         if len(clicked_media_types) > 1:
#             print(f"⚠️ Warning: Clicked items contain multiple media types: {clicked_media_types}")
#         if self.lang_id is not None and len(clicked_lang_ids) > 1:
#             print(f"⚠️ Warning: Clicked items contain multiple languages: {clicked_lang_ids}")
        
#         unique_genres, genre_groups = self._extract_unique_genres_from_clicked(clicked_ids)
        
#         print(f"🎯 User interacted with {len(clicked_ids)} {self.media_type.title()}(s)")
#         items = ', '.join([f"{row['title']} (ID: {row['id']})" for _, row in clicked_df.iterrows()])
#         print(f"📊 Clicked items: {items}")   
#         print(f"🎭 Detected {len(genre_groups)} different genre groups: {[list(g) for g in genre_groups]}")
#         if self.lang_id is not None:
#             print(f"🌍 Language filter: {self.lang_id}")
        
#         # Calculate similarity scores using max similarity approach
#         semantic_scores = self._calculate_semantic_similarity(clicked_ids)
#         genre_scores = self._calculate_multi_genre_similarity(clicked_ids)
#         rating_scores = self._calculate_improved_rating_similarity(clicked_ids)
        
#         # Prepare recommendations
#         recommendations = []
#         for idx, (semantic_score, genre_score, rating_score) in enumerate(
#             zip(semantic_scores, genre_scores, rating_scores)
#         ):
#             movie = self.df.iloc[idx]
#             movie_id = movie['id']
#             views = movie.get('views', 1000)  # Default if views column doesn't exist
            
#             # Skip clicked items and low-view items
#             if movie_id in clicked_ids or views < self.view_threshold:
#                 continue
            
#             # Calculate franchise boost
#             franchise_boost = self._calculate_franchise_boost(clicked_ids, movie['title'])
            
#             # Calculate weighted final score
#             popularity_score = movie['normalized_views']
            
#             base_score = (
#                 self.weights['semantic_similarity'] * semantic_score +
#                 self.weights['genre_similarity'] * genre_score +
#                 self.weights['rating_similarity'] * rating_score +
#                 self.weights['popularity_boost'] * popularity_score
#             )
            
#             final_score = base_score + franchise_boost
            
#             recommendations.append({
#                 'id': movie_id,
#                 'title': movie['title'],
#                 'final_score': final_score,
#                 'semantic_score': semantic_score,
#                 'genre_score': genre_score,
#                 'rating_score': rating_score,
#                 'popularity_score': popularity_score,
#                 'franchise_boost': franchise_boost,
#                 'views': views,
#                 'imdb_rating': movie['imdb_rating'],
#                 'genres': movie['genres'],
#                 'lang_id': movie.get('lang_id', 'Unknown'),
#                 'ismovie': movie['ismovie']
#             })
        
#         # Sort by final score
#         recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        
#         # Apply genre splitting if requested and multiple genres detected
#         if apply_genre_split and len(genre_groups) > 1:
#             final_recommendations = self._split_recommendations_by_genre(
#                 recommendations[:self.top_k * 3], clicked_ids  # Get more candidates for splitting
#             )
#         else:
#             final_recommendations = recommendations[:self.top_k]
        
#         return final_recommendations
    
#     def print_detailed_recommendations(self, recommendations, clicked_ids):
#         """Print detailed recommendation results with genre split info"""
#         unique_genres, genre_groups = self._extract_unique_genres_from_clicked(clicked_ids)
        
#         media_type_names = {0: 'Trailer', 1: 'Movie', 2: 'Short Drama'}
        
#         print(f"\n🎬 Top {len(recommendations)} Multi-Genre {self.media_type.title()} Recommendations:")
#         if self.lang_id is not None:
#             print(f"🌍 Language Filter: {self.lang_id}")
#         if len(genre_groups) > 1:
#             print(f"📊 Split Strategy: {len(genre_groups)} genre groups detected")
#             for i, group in enumerate(genre_groups):
#                 print(f"   Group {i+1}: {list(group)}")
#         print("=" * 100)
        
#         for i, rec in enumerate(recommendations, 1):
#             media_name = media_type_names.get(rec['ismovie'], 'Unknown')
#             print(f"{i}. {rec['title']} (ID: {rec['id']})")
#             print(f"   🎬 Media Type: {media_name}")
#             print(f"   🌍 Language ID: {rec['lang_id']}")
#             print(f"   📊 Final Score: {rec['final_score']:.4f}")
#             print(f"   🎭 Genres: {rec['genres']}")
#             print(f"   ⭐ IMDB: {rec['imdb_rating']:.1f}")
#             print(f"   👀 Views: {rec['views']:,}")
#             franchise_info = f" (+{rec['franchise_boost']:.3f} franchise)" if rec['franchise_boost'] > 0 else ""
#             print(f"   📈 Breakdown: Semantic={rec['semantic_score']:.3f}, "
#                   f"Genre={rec['genre_score']:.3f}, Rating={rec['rating_score']:.3f}, "
#                   f"Popularity={rec['popularity_score']:.3f}{franchise_info}")
#             print("-" * 100)


# # Usage
# if __name__ == "__main__":
#     # Configuration - using the IDs that gave good results
#     CLICKED_IDS = [138]  # Example IDs
#     MEDIA_TYPE = 'movie'  # Options: 'movie', 'trailer', 'short_drama'
#     LANG_ID = None  # Set to specific language ID or None for all languages
    
#     try:
#         # Create improved recommender
#         recommender = ImprovedMultiGenreRecommender(
#             media_type=MEDIA_TYPE,
#             lang_id=LANG_ID,
#             top_k=10,
#             view_threshold=10000
#         )
        
#         # Get recommendations with genre splitting
#         recommendations = recommender.get_recommendations(
#             clicked_ids=CLICKED_IDS,
#             apply_genre_split=True,
#             diversity_threshold=0.75
#         )
        
#         # Print results
#         recommender.print_detailed_recommendations(recommendations, CLICKED_IDS)
        
#         # Print simple format
#         print(f"\n🎬 Simple Format - Top {len(recommendations)} Recommendations:")
#         for rec in recommendations:
#             print(f"- {rec['title']} (ID: {rec['id']}, Score: {rec['final_score']:.4f}, Views: {rec['views']:,})")
        
#         # Print genre distribution
#         print(f"\n📊 Genre Distribution:")
#         clicked_df = recommender.df[recommender.df['id'].isin(CLICKED_IDS)]
#         for _, row in clicked_df.iterrows():
#             print(f"Clicked: {row['title']} (ID: {row['id']}) -> {row['genres']}")
        
#         genre_counts = {}
#         for rec in recommendations:
#             genres = rec['genres'].lower()
#             for genre in genres.split(','):
#                 genre = genre.strip()
#                 if genre:
#                     genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
#         print(f"Recommended genres distribution: {dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True))}")
        
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         print("💡 Make sure your CLICKED_IDS exist in the dataset and match the selected media type and language.")
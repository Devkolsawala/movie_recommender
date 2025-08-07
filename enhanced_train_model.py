# enhanced_train_model_updated.py
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import re
from sklearn.preprocessing import StandardScaler
import os

class EnhancedTextProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load language mapping
        self.language_map = self._load_language_mapping()
        
        # Media type mapping
        self.media_type_map = {
            0: 'series',
            1: 'movie',
            2: 'short_drama'
        }
        
        self.media_type_names = {
            0: 'Series/TV Show',
            1: 'Movie', 
            2: 'Short Drama'
        }
    
    def _load_language_mapping(self):
        """Load language mapping from CSV file"""
        try:
            lang_df = pd.read_csv('language_tbl.csv', header=None)
            lang_df.columns = ['lang_id', 'language_name', 'priority']
            lang_mapping = dict(zip(lang_df['lang_id'], lang_df['language_name']))
            print(f"🌍 Loaded {len(lang_mapping)} languages from language_tbl.csv")
            return lang_mapping
        except FileNotFoundError:
            print("⚠️ language_tbl.csv not found. Using default language mapping.")
            return {
                1: "English", 2: "हिन्दी", 3: "ગુજરાતી", 4: "Español", 5: "தமிழ்",
                6: "Other", 7: "മലയാളം", 10: "Français", 12: "ਪੰਜਾਬੀ", 13: "বাংলা",
                14: "मराठी", 15: "ಕನ್ನಡ", 23: "Chinese", 24: "Korean", 25: "తెలుగు",
                49: "عربي", 55: "Deutsch", 67: "Português"
            }
    
    def clean_text(self, text):
        """Enhanced text cleaning"""
        if pd.isna(text) or text == '':
            return ''
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', str(text).strip())
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\-\.\,\!\?\:]', ' ', text)
        
        return text
    
    def create_balanced_enhanced_text(self, row):
        """Create balanced text representation with language and media type awareness"""
        components = []
        
        # Title (most important - give it appropriate weight)
        title = self.clean_text(row['title'])
        if title:
            components.extend([title] * 2)
        
        # Description (increase weight for semantic content)
        description = self.clean_text(row['description'])
        if description:
            components.extend([description] * 2)
            
            # Extract key themes from description
            themes = self.extract_themes(description)
            if themes:
                components.extend(themes)
        
        # Genres (balanced emphasis to prevent genre bias)
        genres = self.clean_text(row['genres'])
        if genres:
            genre_list = [genre.strip() for genre in genres.split(',')]
            components.extend(genre_list)
            components.append(f"film with {genres.lower()} elements")
        
        # IMDB Rating context (enhanced)
        if pd.notna(row['imdb_rating']) and row['imdb_rating'] > 0:
            rating = row['imdb_rating']
            if rating >= 8.5:
                components.append("exceptional highly acclaimed masterpiece")
            elif rating >= 8.0:
                components.append("excellent highly rated content")
            elif rating >= 7.5:
                components.append("very good well received content")
            elif rating >= 7.0:
                components.append("good quality rated content")
            elif rating >= 6.5:
                components.append("decent watchable content")
            elif rating >= 6.0:
                components.append("average rated content")
            else:
                components.append("low rated content")
        
        # Popularity context based on views (enhanced)
        if pd.notna(row.get('views')) and row['views'] > 0:
            views = row['views']
            if views > 500000000:  # 500M+
                components.append("mega blockbuster worldwide phenomenon")
            elif views > 100000000:  # 100M+
                components.append("blockbuster extremely popular content")
            elif views > 50000000:  # 50M+
                components.append("very popular mainstream hit")
            elif views > 10000000:  # 10M+
                components.append("popular well known content")
            elif views > 1000000:  # 1M+
                components.append("moderately popular content")
            else:
                components.append("niche independent content")
        
        # Media type context (Enhanced)
        if 'ismovie' in row and pd.notna(row['ismovie']):
            media_type_name = self.media_type_names.get(row['ismovie'], 'content')
            media_type_key = self.media_type_map.get(row['ismovie'], 'content')
            
            # Add specific context for each media type
            if row['ismovie'] == 1:  # Movie
                components.append("feature length movie film cinema")
            elif row['ismovie'] == 0:  # Series
                components.append("television series episodic show")
            elif row['ismovie'] == 2:  # Short Drama
                components.append("short drama brief story")
            
            components.append(f"{media_type_key} entertainment content")
        
        # Language context (Enhanced)
        if 'lang_id' in row and pd.notna(row['lang_id']):
            lang_id = int(row['lang_id'])
            language_name = self.language_map.get(lang_id, 'Unknown')
            
            # Add language context
            components.append(f"{language_name} language content")
            components.append(f"language {lang_id} entertainment")
            
            # Add regional context for major languages
            regional_context = {
                1: "international global English content",  # English
                2: "bollywood hindi indian content",  # Hindi
                3: "gujarati regional indian content",  # Gujarati
                4: "spanish latino hispanic content",  # Spanish
                5: "tamil south indian content",  # Tamil
                7: "malayalam south indian content",  # Malayalam
                23: "chinese asian content",  # Chinese
                24: "korean asian k-content",  # Korean
                25: "telugu south indian content",  # Telugu
            }
            
            if lang_id in regional_context:
                components.append(regional_context[lang_id])
        
        # Add content-based descriptors from title analysis
        title_lower = title.lower() if title else ""
        if any(word in title_lower for word in ['love', 'heart', 'romance', 'prema']):
            components.append("romantic emotional story")
        if any(word in title_lower for word in ['war', 'battle', 'fight', 'action']):
            components.append("intense action sequences")
        if any(word in title_lower for word in ['family', 'home', 'mother', 'father']):
            components.append("family oriented story")
        if any(word in title_lower for word in ['comedy', 'funny', 'humor']):
            components.append("comedy entertainment")
        
        # Add cross-cultural appeal indicators
        if 'lang_id' in row and row['lang_id'] == 1:  # English content
            components.append("international crossover appeal")
        elif 'views' in row and row['views'] > 10000000:  # High view content in non-English
            components.append("regional blockbuster crossover potential")
        
        return ' '.join(components)
    
    def extract_themes(self, description):
        """Enhanced theme extraction with more nuanced detection"""
        if not description:
            return []
        
        themes = []
        description_lower = description.lower()
        
        # Enhanced theme mapping with more specific keywords
        theme_mapping = {
            'romance': {
                'keywords': ['love', 'romance', 'romantic', 'relationship', 'couple', 'marry', 'wedding', 'heart', 'passion', 'prema'],
                'description': 'romantic emotional storyline'
            },
            'action': {
                'keywords': ['fight', 'battle', 'war', 'combat', 'violence', 'chase', 'explosion', 'martial', 'warrior', 'action'],
                'description': 'intense action packed sequences'
            },
            'comedy': {
                'keywords': ['funny', 'comedy', 'humor', 'laugh', 'comic', 'hilarious', 'amusing', 'joke'],
                'description': 'humorous entertaining comedy'
            },
            'drama': {
                'keywords': ['family', 'emotional', 'drama', 'life', 'struggle', 'conflict', 'serious', 'intense'],
                'description': 'dramatic emotional storytelling'
            },
            'thriller': {
                'keywords': ['mystery', 'suspense', 'thriller', 'investigation', 'crime', 'murder', 'detective'],
                'description': 'suspenseful thrilling mystery'
            },
            'horror': {
                'keywords': ['horror', 'scary', 'ghost', 'haunted', 'fear', 'terror', 'nightmare'],
                'description': 'frightening horror elements'
            },
            'adventure': {
                'keywords': ['journey', 'adventure', 'travel', 'quest', 'explore', 'expedition'],
                'description': 'adventurous journey story'
            },
            'family': {
                'keywords': ['family', 'children', 'parent', 'kid', 'father', 'mother', 'son', 'daughter'],
                'description': 'family friendly content'
            },
            'friendship': {
                'keywords': ['friend', 'friendship', 'buddy', 'companion', 'loyalty', 'brotherhood'],
                'description': 'friendship and camaraderie'
            },
            'coming_of_age': {
                'keywords': ['young', 'youth', 'growing', 'teenage', 'adolescent', 'maturity'],
                'description': 'coming of age story'
            }
        }
        
        # Count theme occurrences
        theme_scores = {}
        for theme, info in theme_mapping.items():
            score = sum(1 for keyword in info['keywords'] if keyword in description_lower)
            if score > 0:
                theme_scores[theme] = score
        
        # Add themes based on strength
        for theme, score in sorted(theme_scores.items(), key=lambda x: x[1], reverse=True):
            if score >= 2:  # Strong theme presence
                themes.append(theme_mapping[theme]['description'])
            elif score == 1:  # Weak theme presence
                themes.append(f"elements of {theme}")
        
        # Limit to top 3 themes to avoid over-tagging
        return themes[:3]

def process_dataset_by_language_and_media():
    """Process the dataset with comprehensive language and media type separation"""
    # Load full dataset
    try:
        df = pd.read_csv('final_dataset.csv')
        print(f"✅ Loaded final_dataset.csv with {len(df):,} records")
    except FileNotFoundError:
        print("❌ final_dataset.csv not found. Please run prepare_dataset.py first.")
        return
    
    # Initialize text processor
    processor = EnhancedTextProcessor()
    
    # Get unique combinations
    unique_lang_ids = sorted(df['lang_id'].unique()) if 'lang_id' in df.columns else [None]
    unique_media_types = sorted(df['ismovie'].unique()) if 'ismovie' in df.columns else [1]
    
    print(f"🌍 Found {len(unique_lang_ids)} unique languages: {unique_lang_ids}")
    print(f"🎬 Found {len(unique_media_types)} media types: {[processor.media_type_names.get(mt, mt) for mt in unique_media_types]}")
    
    print("\n🔄 Processing dataset with comprehensive language and media type separation...")
    
    # Process each media type
    for media_value in unique_media_types:
        media_name = processor.media_type_map.get(media_value, f'type_{media_value}')
        media_display_name = processor.media_type_names.get(media_value, f'Type {media_value}')
        
        media_subset = df[df['ismovie'] == media_value].copy()
        if media_subset.empty:
            print(f"⚠️ No data found for {media_display_name}")
            continue
        
        print(f"\n📂 Processing {media_display_name} ({len(media_subset):,} items)...")
        
        # Create enhanced text for all items of this media type
        enhanced_texts = []
        for idx, row in media_subset.iterrows():
            enhanced_text = processor.create_balanced_enhanced_text(row)
            enhanced_texts.append(enhanced_text)
            
            # Progress indicator
            if len(enhanced_texts) % 1000 == 0:
                print(f"   Processed {len(enhanced_texts):,} items...")
        
        # Update the text column
        media_subset['enhanced_text'] = enhanced_texts
        
        # Save the main media type file (all languages combined)
        output_filename = f'{media_name}s.csv'
        media_subset.to_csv(output_filename, index=False)
        print(f"💾 Saved {output_filename} with {len(media_subset):,} records")
        
        # Generate embeddings for the full media type (all languages)
        print(f"🧠 Generating embeddings for all {media_display_name}s...")
        embeddings = processor.model.encode(
            enhanced_texts, 
            convert_to_tensor=True, 
            show_progress_bar=True,
            batch_size=16,
            normalize_embeddings=True
        )
        
        # Save main embeddings file
        embeddings_filename = f'{media_name}_embeddings.pt'
        torch.save(embeddings, embeddings_filename)
        print(f"✅ Saved {embeddings_filename}")
        
        # Process by language if language information is available
        if 'lang_id' in media_subset.columns:
            lang_stats = media_subset['lang_id'].value_counts().sort_index()
            print(f"   📊 Language distribution for {media_display_name}:")
            
            for lang_id in unique_lang_ids:
                if pd.isna(lang_id):
                    continue
                    
                lang_subset = media_subset[media_subset['lang_id'] == lang_id].copy()
                if lang_subset.empty:
                    continue
                
                lang_name = processor.language_map.get(lang_id, f'Language {lang_id}')
                item_count = len(lang_subset)
                
                print(f"      🌍 {lang_name} (ID: {lang_id}): {item_count:,} items")
                
                # Get embeddings for this language subset
                # Create a mapping from original index to position in media_subset
                media_index_mapping = {orig_idx: pos for pos, orig_idx in enumerate(media_subset.index)}
                lang_positions = [media_index_mapping[idx] for idx in lang_subset.index if idx in media_index_mapping]
                lang_embeddings = embeddings[lang_positions]
                
                # Save language-specific files if they have enough data
                min_items_threshold = 50  # Minimum items to create separate files
                if item_count >= min_items_threshold:
                    # Save CSV file
                    lang_csv_filename = f'{media_name}s_lang{lang_id}.csv'
                    lang_subset.to_csv(lang_csv_filename, index=False)
                    
                    # Save embeddings file
                    lang_embeddings_filename = f'{media_name}_embeddings_lang{lang_id}.pt'
                    torch.save(lang_embeddings, lang_embeddings_filename)
                    
                    print(f"         💾 Saved {lang_csv_filename} and {lang_embeddings_filename}")
                else:
                    print(f"         ⚠️ Too few items ({item_count}) for separate files (min: {min_items_threshold})")
        
        # Print sample enhanced text for verification
        print(f"\n📝 Sample enhanced texts for {media_display_name}:")
        sample_count = min(2, len(enhanced_texts))
        for i in range(sample_count):
            sample_row = media_subset.iloc[i]
            print(f"Sample {i+1}:")
            print(f"  ID: {sample_row['id']}")
            print(f"  Title: {sample_row['title']}")
            print(f"  Genres: {sample_row['genres']}")
            if 'lang_id' in sample_row:
                lang_name = processor.language_map.get(sample_row['lang_id'], 'Unknown')
                print(f"  Language: {lang_name} (ID: {sample_row['lang_id']})")
            print(f"  Media Type: {media_display_name}")
            print(f"  Enhanced Text: {enhanced_texts[i][:200]}...")
            print("-" * 60)
    
    # Create comprehensive summary statistics
    print(f"\n📊 Comprehensive Dataset Summary:")
    print("=" * 80)
    
    # Overall statistics
    total_items = len(df)
    print(f"📈 Total Items: {total_items:,}")
    
    # Media type breakdown
    print(f"\n🎬 Media Type Distribution:")
    for media_value in unique_media_types:
        media_name = processor.media_type_names.get(media_value, f'Type {media_value}')
        media_count = len(df[df['ismovie'] == media_value])
        percentage = (media_count / total_items) * 100
        print(f"  {media_name}: {media_count:,} ({percentage:.1f}%)")
        
        # Language breakdown for each media type
        if 'lang_id' in df.columns:
            media_df = df[df['ismovie'] == media_value]
            lang_counts = media_df['lang_id'].value_counts().sort_index()
            
            print(f"    Language breakdown:")
            for lang_id, count in lang_counts.head(5).items():  # Show top 5 languages
                lang_name = processor.language_map.get(lang_id, f'Language {lang_id}')
                percentage = (count / media_count) * 100
                print(f"      {lang_name}: {count:,} ({percentage:.1f}%)")
            
            if len(lang_counts) > 5:
                remaining = len(lang_counts) - 5
                print(f"      ... and {remaining} more languages")
        
        print("-" * 50)
    
    # Files created summary
    print(f"\n📁 Files Created:")
    for media_value in unique_media_types:
        media_name = processor.media_type_map.get(media_value, f'type_{media_value}')
        media_display_name = processor.media_type_names.get(media_value, f'Type {media_value}')
        
        print(f"  📂 {media_display_name}:")
        print(f"    - {media_name}s.csv (all languages)")
        print(f"    - {media_name}_embeddings.pt (all languages)")
        
        # Language-specific files
        if 'lang_id' in df.columns:
            media_df = df[df['ismovie'] == media_value]
            lang_counts = media_df['lang_id'].value_counts()
            
            lang_files_created = 0
            for lang_id, count in lang_counts.items():
                if count >= 50:  # Files created for languages with 50+ items
                    lang_name = processor.language_map.get(lang_id, f'Language {lang_id}')
                    print(f"    - {media_name}s_lang{lang_id}.csv ({lang_name})")
                    print(f"    - {media_name}_embeddings_lang{lang_id}.pt ({lang_name})")
                    lang_files_created += 1
            
            if lang_files_created == 0:
                print(f"    - No language-specific files (insufficient data)")
    
    print(f"\n✅ Processing completed successfully!")
    print(f"\n🔍 Key improvements implemented:")
    print("- ✨ Enhanced language-aware text processing")
    print("- 🎬 Media type specific context in embeddings")
    print("- 🌍 Regional and cultural context integration")
    print("- 📊 Comprehensive file organization by language and media type")
    print("- 🎯 Balanced semantic representation")
    print("- 🔄 Cross-cultural appeal detection")

def create_comprehensive_index():
    """Create a comprehensive index of all available combinations"""
    try:
        df = pd.read_csv('final_dataset.csv')
        processor = EnhancedTextProcessor()
        
        print("\n📋 Comprehensive Dataset Index:")
        print("=" * 80)
        
        total_items = len(df)
        print(f"📊 Total Records: {total_items:,}")
        
        # Media type analysis
        media_types = df['ismovie'].unique()
        for media_value in sorted(media_types):
            media_name = processor.media_type_names.get(media_value, f'Type {media_value}')
            media_subset = df[df['ismovie'] == media_value]
            media_count = len(media_subset)
            
            print(f"\n🎬 {media_name} (ismovie={media_value}): {media_count:,} items")
            
            if 'lang_id' in df.columns:
                lang_counts = media_subset['lang_id'].value_counts().sort_index()
                print(f"    📊 Available in {len(lang_counts)} languages:")
                
                for lang_id, count in lang_counts.items():
                    lang_name = processor.language_map.get(lang_id, f'Language {lang_id}')
                    percentage = (count / media_count) * 100
                    file_status = "✅ Separate files" if count >= 50 else "⚠️ Combined only"
                    print(f"      {lang_name} (ID: {lang_id}): {count:,} ({percentage:.1f}%) - {file_status}")
        
        print(f"\n🎯 Usage Examples:")
        print("# For all movies in any language:")
        print("recommender = ImprovedMultiGenreRecommender(media_type='movie', lang_id=None)")
        
        print("\n# For movies in specific language:")
        popular_langs = df['lang_id'].value_counts().head(3)
        for lang_id, count in popular_langs.items():
            lang_name = processor.language_map.get(lang_id, f'Language {lang_id}')
            print(f"recommender = ImprovedMultiGenreRecommender(media_type='movie', lang_id={lang_id})  # {lang_name}")
        
        print("\n# For other media types:")
        for media_value in sorted(media_types):
            if media_value != 1:  # Skip movie as it's shown above
                media_key = processor.media_type_map.get(media_value, f'type_{media_value}')
                media_name = processor.media_type_names.get(media_value, f'Type {media_value}')
                print(f"recommender = ImprovedMultiGenreRecommender(media_type='{media_key}', lang_id=1)  # {media_name} in English")
        
        print(f"\n📁 Available Files:")
        for media_value in sorted(media_types):
            media_key = processor.media_type_map.get(media_value, f'type_{media_value}')
            media_name = processor.media_type_names.get(media_value, f'Type {media_value}')
            
            print(f"\n  📂 {media_name}:")
            print(f"    - {media_key}s.csv")
            print(f"    - {media_key}_embeddings.pt")
            
            # Check for language-specific files
            media_subset = df[df['ismovie'] == media_value]
            lang_counts = media_subset['lang_id'].value_counts()
            
            for lang_id, count in lang_counts.items():
                if count >= 50:
                    lang_name = processor.language_map.get(lang_id, f'Language {lang_id}')
                    print(f"    - {media_key}s_lang{lang_id}.csv ({lang_name})")
                    print(f"    - {media_key}_embeddings_lang{lang_id}.pt ({lang_name})")
        
    except FileNotFoundError:
        print("❌ final_dataset.csv not found. Please run prepare_dataset.py first.")
    except Exception as e:
        print(f"❌ Error creating index: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--index':
        # Just create the index
        create_comprehensive_index()
    else:
        # Full processing
        process_dataset_by_language_and_media()
        create_comprehensive_index()






















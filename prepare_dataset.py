import pandas as pd

def load_language_mapping():
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

# Load your raw dataset
raw_df = pd.read_csv('post_tbl (1).csv')  # Replace with your actual file path

# Genre map (from cat_id to genre name)
genre_map = {
    1: 'Action', 2: 'Comedy', 3: 'Drama', 4: 'War', 5: 'Horror', 6: 'Mystery',
    10: 'Adventure', 11: 'Love Story', 19: 'Thriller', 21: 'Sci-fi',
    25: 'Fantasy', 28: 'Crime', 29: 'Political', 30: 'Romance', 36: 'Sport',
    37: 'Family', 38: 'History', 39: 'Biography', 40: 'Animation',
    41: 'Documentary', 42: 'Dance', 61: 'Western', 68: 'Reality',
    69: 'Musical', 72: 'Exclusive Drama', 73: 'Popular'
}

# Media type mapping
media_type_map = {
    0: 'Series/TV Show',
    1: 'Movie', 
    2: 'Short Drama'
}

# Load language mapping
language_map = load_language_mapping()

print(f"📊 Original data: {len(raw_df):,} records")
print(f"🌍 Available languages: {len(language_map)}")
print(f"🎬 Media types: {list(media_type_map.values())}")

# Function to convert '1,3,11' to 'Action,Drama,Love Story'
def map_cat_ids_to_genres(cat_ids_str):
    if pd.isna(cat_ids_str):
        return ''
    genres = []
    try:
        for cid in str(cat_ids_str).split(','):
            cid = cid.strip()
            if cid.isdigit():
                genre_name = genre_map.get(int(cid))
                if genre_name:
                    genres.append(genre_name)
    except Exception:
        return ''
    return ', '.join(genres)

# Apply genre mapping
raw_df['genres'] = raw_df['cat_id'].apply(map_cat_ids_to_genres)

# Clean title and description
raw_df['title'] = raw_df['title'].fillna('').astype(str).str.strip()
raw_df['caption'] = raw_df['caption'].fillna('').astype(str).str.strip()

# Handle lang_id - convert to numeric and handle missing values
raw_df['lang_id'] = pd.to_numeric(raw_df['lang_id'], errors='coerce')
# Fill missing lang_id with most common language (assuming English = 1)
most_common_lang = raw_df['lang_id'].mode().iloc[0] if not raw_df['lang_id'].mode().empty else 1
raw_df['lang_id'] = raw_df['lang_id'].fillna(most_common_lang)

# Handle ismovie - ensure it's numeric
raw_df['ismovie'] = pd.to_numeric(raw_df['ismovie'], errors='coerce').fillna(1)

# Drop rows with missing or empty title or genres
df = raw_df[(raw_df['title'] != '') & (raw_df['genres'] != '')].copy()
print(f"📝 After dropping rows without title or genres: {len(df):,}")

# Clean other fields
df['imdb_rating'] = pd.to_numeric(df['imdb_rating'], errors='coerce').fillna(0)
df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)

# Add language name for reference
df['language_name'] = df['lang_id'].map(language_map).fillna('Unknown')

# Add media type name for reference
df['media_type_name'] = df['ismovie'].map(media_type_map).fillna('Unknown')

# Rename caption to description
df = df.rename(columns={'caption': 'description'})

# Select and reorder important columns
final_df = df[[
    'id', 'title', 'description', 'genres', 'imdb_rating', 'views', 
    'ismovie', 'media_type_name', 'lang_id', 'language_name'
]].copy()

# Convert data types for consistency
final_df['id'] = final_df['id'].astype(int)
final_df['lang_id'] = final_df['lang_id'].astype(int)
final_df['ismovie'] = final_df['ismovie'].astype(int)

# Save the processed dataset
final_df.to_csv('final_dataset.csv', index=False)
print(f"✅ Final dataset saved as 'final_dataset.csv' with {len(final_df):,} records")

# Print statistics
print(f"\n📊 Dataset Statistics:")
print("=" * 50)

print(f"\n🎬 Media Type Distribution:")
media_stats = final_df['media_type_name'].value_counts()
for media_type, count in media_stats.items():
    print(f"  {media_type}: {count:,} ({count/len(final_df)*100:.1f}%)")

print(f"\n🌍 Language Distribution:")
lang_stats = final_df['language_name'].value_counts().head(10)
for lang, count in lang_stats.items():
    print(f"  {lang}: {count:,} ({count/len(final_df)*100:.1f}%)")

print(f"\n⭐ Rating Statistics:")
print(f"  Average IMDB Rating: {final_df['imdb_rating'].mean():.2f}")
print(f"  Highest Rated: {final_df['imdb_rating'].max():.1f}")
print(f"  Movies with rating > 7: {len(final_df[final_df['imdb_rating'] > 7]):,}")

print(f"\n👀 Views Statistics:")
print(f"  Total Views: {final_df['views'].sum():,}")
print(f"  Average Views: {final_df['views'].mean():.0f}")
print(f"  Most Viewed: {final_df['views'].max():,}")

print(f"\n🎭 Genre Statistics:")
# Count genre occurrences
all_genres = []
for genres_str in final_df['genres']:
    if genres_str:
        all_genres.extend([g.strip() for g in genres_str.split(',')])

genre_counts = pd.Series(all_genres).value_counts().head(10)
for genre, count in genre_counts.items():
    print(f"  {genre}: {count:,}")

print(f"\n🔍 Sample Records:")
print("-" * 80)
for i in range(min(3, len(final_df))):
    row = final_df.iloc[i]
    print(f"ID: {row['id']} | {row['media_type_name']} | {row['language_name']}")
    print(f"Title: {row['title']}")
    print(f"Genres: {row['genres']}")
    print(f"Rating: {row['imdb_rating']} | Views: {row['views']:,}")
    print("-" * 80)

print(f"\n✨ Dataset ready for recommendation system!")
print(f"💡 You can now use different combinations of media_type and lang_id for recommendations.")

















# old code for dataset preparation

# import pandas as pd

# # Load your raw dataset
# raw_df = pd.read_csv('D:\Dev\movie-demo\post_tbl (1).csv')  # <-- Replace with your real file

# # Genre map (from cat_id to genre name)
# genre_map = {
#     1: 'Action', 2: 'Comedy', 3: 'Drama', 4: 'War', 5: 'Horror', 6: 'Mystery',
#     10: 'Adventure', 11: 'Love Story', 19: 'Thriller', 21: 'Sci-fi',
#     25: 'Fantasy', 28: 'Crime', 29: 'Political', 30: 'Romance', 36: 'Sport',
#     37: 'Family', 38: 'History', 39: 'Biography', 40: 'Animation',
#     41: 'Documentary', 42: 'Dance', 61: 'Western', 68: 'Reality',
#     69: 'Musical', 72: 'Exclusive Drama', 73: 'Popular'
# }

# print(f"Original data: {len(raw_df)}")

# # Function to convert '1,3,11' to 'Action,Drama,Love Story'
# def map_cat_ids_to_genres(cat_ids_str):
#     if pd.isna(cat_ids_str):
#         return ''
#     genres = []
#     try:
#         for cid in str(cat_ids_str).split(','):
#             cid = cid.strip()
#             if cid.isdigit():
#                 genre_name = genre_map.get(int(cid))
#                 if genre_name:
#                     genres.append(genre_name)
#     except Exception:
#         return ''
#     return ', '.join(genres)

# # Apply genre mapping
# raw_df['genres'] = raw_df['cat_id'].apply(map_cat_ids_to_genres)

# # Clean title
# raw_df['title'] = raw_df['title'].fillna('').astype(str).str.strip()

# # Drop rows with missing or empty title or genres
# df = raw_df[(raw_df['title'] != '') & (raw_df['genres'] != '')]
# print(f"After dropping rows without title or genres: {len(df)}")

# # Filter only where ismovie == 1
# df = df[df['ismovie'] == 1]
# print(f"After filtering ismovie == 1: {len(df)}")

# # Clean other fields
# df['imdb_rating'] = pd.to_numeric(df['imdb_rating'], errors='coerce').fillna(0)
# df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)

# # Rename and select important columns
# df = df.rename(columns={'caption': 'description'})
# final_df = df[['id', 'title', 'description', 'genres', 'imdb_rating', 'views', 'ismovie']].copy()

# # Save
# final_df.to_csv('cleaned_movies.csv', index=False)
# print("✅ Cleaned dataset saved as 'cleaned_movies.csv'")

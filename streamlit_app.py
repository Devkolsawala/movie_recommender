import streamlit as st
from enhanced_test import ImprovedMultiGenreRecommender
import pandas as pd

@st.cache_data
def load_data():
    """Load the main dataset and language mapping"""
    df = pd.read_csv("final_dataset.csv")
    try:
        lang_df = pd.read_csv("language_tbl.csv", header=None, names=["lang_id", "language_name", "priority"])
    except FileNotFoundError:
        # Fallback language mapping if file doesn't exist
        lang_data = [
            [1, "English", 1], [2, "हिन्दी", 2], [3, "ગુજરાતી", 3], [4, "Español", 4], 
            [5, "தமிழ்", 5], [6, "Other", 6], [7, "മലയാളം", 7], [10, "Français", 8],
            [12, "ਪੰਜਾਬੀ", 9], [13, "বাংলা", 10], [14, "मराठी", 11], [15, "ಕನ್ನಡ", 12],
            [23, "Chinese", 13], [24, "Korean", 14], [25, "తెలుగు", 15], [49, "عربي", 16],
            [55, "Deutsch", 17], [67, "Português", 18]
        ]
        lang_df = pd.DataFrame(lang_data, columns=["lang_id", "language_name", "priority"])
    return df, lang_df

@st.cache_data
def get_language_name(lang_id, lang_df):
    """Get language name from language ID"""
    try:
        return lang_df[lang_df["lang_id"] == lang_id]["language_name"].iloc[0]
    except (IndexError, KeyError):
        return f"Language {lang_id}"

# Load data
df, lang_df = load_data()

# Page configuration
st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎬 AI Movie Recommender")
st.markdown("Select a few movies or series you've liked, and get smart AI recommendations.")

# Media type selection
media_type = st.selectbox("Select Media Type", ["movie", "series", "short_drama"])

# Language selection
lang_name_to_id = dict(zip(lang_df["language_name"], lang_df["lang_id"]))
language = st.selectbox("Select Language", ["All"] + list(lang_name_to_id.keys()))

# Filter dataset based on selections
media_type_mapping = {"movie": 1, "series": 0, "short_drama": 2}
filtered_df = df[df["ismovie"] == media_type_mapping[media_type]].copy()

lang_id = None
if language != "All":
    lang_id = lang_name_to_id[language]
    filtered_df = filtered_df[filtered_df["lang_id"] == lang_id]

# Multi-select for choosing liked items
clicked_titles = st.multiselect(
    "Choose a few items you liked",
    options=filtered_df["title"].tolist()
)

# Single button for recommendations
if st.button("🔮 Get AI Recommendations"):
    if clicked_titles:
        with st.spinner("Getting AI recommendations..."):
            try:
                # Get clicked item IDs
                clicked_ids = filtered_df[filtered_df["title"].isin(clicked_titles)]["id"].tolist()
                
                # Create recommender with explicit top_k=20
                recommender = ImprovedMultiGenreRecommender(
                    media_type=media_type, 
                    lang_id=lang_id, 
                    top_k=20,
                    equal_genre_distribution=False  # Disable to ensure consistent count
                )
                
                # Get recommendations
                recs = recommender.get_recommendations(clicked_ids=clicked_ids)
                
                # Ensure exactly 20 recommendations
                if len(recs) > 20:
                    recs = recs[:20]
                
                if recs:
                    st.subheader("✨ Recommended For You:")
                    for rec in recs:
                        # Get language name for the recommendation
                        rec_lang_name = get_language_name(rec['lang_id'], lang_df)
                        
                        st.markdown(f"**{rec['title']}**")
                        st.write(f"Genres: {rec['genres']}")
                        st.write(f"Language: {rec_lang_name}")
                        st.divider()
                else:
                    st.warning("No recommendations found. Try selecting different items.")
            
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
    else:
        st.info("Select at least one title to get recommendations.")












# # streamlit_app.py

# import streamlit as st
# from enhanced_test import ImprovedMultiGenreRecommender
# import pandas as pd

# @st.cache_data
# def load_data():
#     df = pd.read_csv("final_dataset.csv")
#     lang_df = pd.read_csv("language_tbl.csv", header=None, names=["lang_id", "language_name", "priority"])
#     return df, lang_df

# df, lang_df = load_data()

# st.set_page_config(page_title="AI Movie Recommender", layout="wide")
# st.title("🎬 AI Movie Recommender")
# st.markdown("Select a few movies or series you’ve liked, and get smart AI recommendations.")

# media_type = st.selectbox("Select Media Type", ["movie", "series", "short_drama"])
# lang_name_to_id = dict(zip(lang_df["language_name"], lang_df["lang_id"]))
# language = st.selectbox("Select Language", ["All"] + list(lang_name_to_id.keys()))

# filtered_df = df[df["ismovie"] == {"movie": 1, "series": 0, "short_drama": 2}[media_type]]
# lang_id = None
# if language != "All":
#     lang_id = lang_name_to_id[language]
#     filtered_df = filtered_df[filtered_df["lang_id"] == lang_id]

# clicked_titles = st.multiselect(
#     "Choose a few items you liked",
#     options=filtered_df["title"].tolist()
# )

# if st.button("🔮 Get AI Recommendations") and clicked_titles:
#     clicked_ids = filtered_df[filtered_df["title"].isin(clicked_titles)]["id"].tolist()
#     recommender = ImprovedMultiGenreRecommender(media_type=media_type, lang_id=lang_id)
#     recs = recommender.get_recommendations(clicked_ids=clicked_ids, apply_genre_split=True)

#     st.subheader("✨ Recommended For You:")
#     for rec in recs:
#         st.markdown(f"**{rec['title']}**")
#         st.write(f"Genres: {rec['genres']}")
#         st.write(f"IMDB Rating: {rec['imdb_rating']}, Views: {rec['views']}")
#         st.divider()
# elif not clicked_titles:
#     st.info("Select at least one title to get recommendations.")
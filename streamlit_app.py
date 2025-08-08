import streamlit as st
from enhanced_test_improved_updated_full import ImprovedMultiGenreRecommender
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
st.markdown("Select a few movies or series you've liked, and get smart AI recommendations based on genres, content, and language preferences.")

# Create two columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    media_type = st.selectbox(
        "🎭 Select Media Type", 
        ["movie", "series", "short_drama"],
        help="Choose the type of content you want recommendations for"
    )

with col2:
    # Language selection
    lang_name_to_id = dict(zip(lang_df["language_name"], lang_df["lang_id"]))
    language = st.selectbox(
        "🌍 Select Language", 
        ["All"] + list(lang_name_to_id.keys()),
        help="Filter content by language, or select 'All' for any language"
    )

# Filter dataset based on selections
media_type_mapping = {"movie": 1, "series": 0, "short_drama": 2}
filtered_df = df[df["ismovie"] == media_type_mapping[media_type]].copy()

lang_id = None
if language != "All":
    lang_id = lang_name_to_id[language]
    filtered_df = filtered_df[filtered_df["lang_id"] == lang_id]

# Show dataset stats
st.info(f"📊 Available {media_type}s: {len(filtered_df):,} items" + 
        (f" in {language}" if language != "All" else " in all languages"))

# Multi-select for choosing liked items
clicked_titles = st.multiselect(
    f"🎯 Choose {media_type}s you liked (select 2-5 for better recommendations)",
    options=filtered_df["title"].tolist(),
    help=f"Select multiple {media_type}s that you enjoyed. The AI will find similar content based on genres, themes, and content similarity."
)

# Show selected items info
if clicked_titles:
    st.write(f"✅ Selected {len(clicked_titles)} {media_type}(s): {', '.join(clicked_titles[:3])}" + 
             (f" and {len(clicked_titles)-3} more..." if len(clicked_titles) > 3 else ""))

# Advanced options in an expander
with st.expander("⚙️ Advanced Options"):
    col3, col4 = st.columns(2)
    with col3:
        enforce_language_matching = st.checkbox(
            "🔒 Language Matching", 
            value=True, 
            help="Only recommend content in the same language(s) as your selected items"
        )
    with col4:
        equal_genre_distribution = st.checkbox(
            "⚖️ Equal Genre Distribution", 
            value=True, 
            help="Distribute recommendations equally across all genres from your selected items"
        )
    
    top_k = st.slider("📊 Number of Recommendations", min_value=10, max_value=30, value=20, step=5)

# Get recommendations button
if st.button("🔮 Get AI Recommendations", type="primary") and clicked_titles:
    with st.spinner("🤖 AI is analyzing your preferences and finding the best recommendations..."):
        try:
            # Get clicked item IDs
            clicked_ids = filtered_df[filtered_df["title"].isin(clicked_titles)]["id"].tolist()
            
            # Create recommender with advanced options
            recommender = ImprovedMultiGenreRecommender(
                media_type=media_type, 
                lang_id=lang_id,
                top_k=top_k,
                enforce_language_matching=enforce_language_matching,
                equal_genre_distribution=equal_genre_distribution
            )
            
            # Get recommendations
            recs = recommender.get_recommendations(clicked_ids=clicked_ids)
            
            if recs:
                st.success(f"✨ Found {len(recs)} great recommendations for you!")
                
                # Show user's selected items with their details
                st.subheader("🎯 Your Selected Items")
                selected_items_df = filtered_df[filtered_df["title"].isin(clicked_titles)]
                
                for _, item in selected_items_df.iterrows():
                    lang_name = get_language_name(item["lang_id"], lang_df)
                    st.write(f"📌 **{item['title']}** | {lang_name} | {item['genres']}")
                
                st.divider()
                
                # Display recommendations in a clean card format
                st.subheader("🎬 Recommended For You")
                st.write("Based on your preferences, here are movies/shows you might love:")
                
                # Create a more visually appealing layout
                for idx, rec in enumerate(recs, start=1):
                    # Get language name for the recommendation
                    rec_lang_name = get_language_name(rec['lang_id'], lang_df)
                    
                    # Create a card-like display using containers
                    with st.container():
                        # Title row with numbering
                        st.markdown(f"### {idx}. {rec['title']}")
                        
                        # Create columns for better information layout
                        info_col1, info_col2, info_col3 = st.columns([2, 1, 1])
                        
                        with info_col1:
                            st.markdown(f"**🎭 Genres:** {rec['genres']}")
                        
                        with info_col2:
                            st.markdown(f"**🌍 Language:** {rec_lang_name}")
                        
                        with info_col3:
                            # Show match score with color coding
                            score = rec['final_score']
                            if score >= 0.7:
                                score_color = "🟢"
                            elif score >= 0.6:
                                score_color = "🟡" 
                            else:
                                score_color = "🔴"
                            st.markdown(f"**📊 Match:** {score_color} {score:.2f}")
                        
                        # Add target genre info if available (for equal distribution)
                        if 'target_genre' in rec:
                            st.markdown(f"**🎯 Target Genre:** {rec['target_genre'].title()}")
                        
                        # Divider between recommendations
                        if idx < len(recs):  # Don't add divider after last item
                            st.divider()
                
                # Show summary statistics
                st.subheader("📊 Recommendation Summary")
                
                # Genre distribution
                genre_counts = {}
                language_counts = {}
                
                for rec in recs:
                    # Count genres
                    if rec['genres']:
                        for genre in rec['genres'].split(','):
                            genre = genre.strip()
                            genre_counts[genre] = genre_counts.get(genre, 0) + 1
                    
                    # Count languages
                    rec_lang = get_language_name(rec['lang_id'], lang_df)
                    language_counts[rec_lang] = language_counts.get(rec_lang, 0) + 1
                
                # Display stats in columns
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.write("**🎭 Genre Distribution:**")
                    for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                        percentage = (count / len(recs)) * 100
                        st.write(f"• {genre}: {count} ({percentage:.0f}%)")
                
                with stats_col2:
                    st.write("**🌍 Language Distribution:**")
                    for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / len(recs)) * 100
                        st.write(f"• {lang}: {count} ({percentage:.0f}%)")
                
            else:
                st.warning("😔 No recommendations found. Try selecting different items or adjusting the filters.")
        
        except Exception as e:
            st.error(f"❌ Error generating recommendations: {str(e)}")
            st.write("💡 Please try again or check if your dataset files are properly loaded.")

elif st.button("🔮 Get AI Recommendations", type="primary") and not clicked_titles:
    st.warning("⚠️ Please select at least one movie or show that you liked before getting recommendations.")

# Footer with additional info
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
🤖 Powered by AI content similarity analysis, genre matching, and semantic understanding.<br>
Select items you enjoyed to get personalized recommendations based on content, themes, and preferences.
</div>
""", unsafe_allow_html=True)












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
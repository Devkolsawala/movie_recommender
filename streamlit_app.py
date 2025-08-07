import streamlit as st
from enhanced_test import ImprovedMultiGenreRecommender
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_csv("final_dataset.csv")
    lang_df = pd.read_csv("language_tbl.csv", header=None, names=["lang_id", "language_name", "priority"])
    return df, lang_df

df, lang_df = load_data()

st.set_page_config(page_title="AI Movie Recommender", layout="wide")
st.title("🎬 AI Movie Recommender")
st.markdown("Select a few movies or series you’ve liked, and get smart AI recommendations.")

media_type = st.selectbox("Select Media Type", ["movie", "series", "short_drama"])
lang_name_to_id = dict(zip(lang_df["language_name"], lang_df["lang_id"]))
language = st.selectbox("Select Language", ["All"] + list(lang_name_to_id.keys()))

filtered_df = df[df["ismovie"] == {"movie": 1, "series": 0, "short_drama": 2}[media_type]]
lang_id = None
if language != "All":
    lang_id = lang_name_to_id[language]
    filtered_df = filtered_df[filtered_df["lang_id"] == lang_id]

clicked_titles = st.multiselect(
    "Choose a few items you liked",
    options=filtered_df["title"].tolist()
)

if st.button("🔮 Get AI Recommendations") and clicked_titles:
    clicked_ids = filtered_df[filtered_df["title"].isin(clicked_titles)]["id"].tolist()
    recommender = ImprovedMultiGenreRecommender(media_type=media_type, lang_id=lang_id)
    recs = recommender.get_recommendations(clicked_ids=clicked_ids)

    st.subheader("✨ Recommended 20 Movies For You based on your current movie preferences:")
    st.write(f"Showing {len(recs)} recommendations based on your selection.")

    for idx, rec in enumerate(recs, start=1):
        with st.expander(f"{idx}. {rec['title']}"):
            st.markdown(f"**Genres:** {rec['genres']}")
            st.markdown(f"**IMDB Rating:** {rec['imdb_rating']}")
            st.markdown(f"**Views:** {rec['views']}")











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
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Netflix Recommender 🎬",
    page_icon="🍿",
    layout="wide"
)

st.markdown("<h1 style='text-align:center;color:#E50914;'>🍿 Netflix Recommendation App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Find your next binge-worthy show!</p>", unsafe_allow_html=True)
st.write("---")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/shreyadas/Desktop/netflix_titles.csv")
    df["description"] = df["description"].fillna("")
    df["listed_in"] = df["listed_in"].fillna("Unknown")
    return df

df = load_data()

# ---------------------------
# TF-IDF MODEL
# ---------------------------
@st.cache_resource
def build_tfidf():
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["description"])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_tfidf()

# ---------------------------
# FILTERS
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    type_filter = st.selectbox("Select Type", ["All"] + sorted(df["type"].dropna().unique().tolist()))
with col2:
    genre_filter = st.selectbox("Select Genre", ["All"] + sorted(set(",".join(df["listed_in"].dropna()).split(", "))))

filtered_df = df.copy()
if type_filter != "All":
    filtered_df = filtered_df[filtered_df["type"] == type_filter]
if genre_filter != "All":
    filtered_df = filtered_df[filtered_df["listed_in"].str.contains(genre_filter, case=False, na=False)]

# ---------------------------
# SEARCH BAR
# ---------------------------
search_title = st.selectbox(
    "🎥 Search or Select a Movie/Show:",
    options=filtered_df["title"].values,
    index=None,
    placeholder="Start typing..."
)

# ---------------------------
# RECOMMENDATION LOGIC
# ---------------------------
def get_recommendations(title, n=5):
    idx = df[df["title"] == title].index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = sim_scores.argsort()[-n-1:][::-1]
    recs = df.iloc[similar_indices[1:]]
    return recs

if search_title:
    if st.button("🍿 Show Recommendations"):
        st.subheader(f"Because you watched **{search_title}**...")
        recommendations = get_recommendations(search_title, 5)

        for _, row in recommendations.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(
                        "https://cdn-icons-png.flaticon.com/512/4149/4149643.png",
                        width=100
                    )
                with col2:
                    st.markdown(f"### 🎬 {row['title']}")
                    st.markdown(f"**Type:** {row['type']}")
                    st.markdown(f"**Genre:** {row['listed_in']}")
                    st.markdown(f"**Description:** {row['description'][:300]}...")

        st.success("✅ Recommendations generated successfully!")

st.write("---")
st.markdown("<p style='text-align:center;font-size:12px;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)

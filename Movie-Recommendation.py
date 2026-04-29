import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# 1. DATASET  (built-in, no download needed)
# =============================================================================
movies = pd.DataFrame({
    "title": [
        "The Dark Knight", "Inception", "Interstellar", "The Prestige",
        "Memento", "Batman Begins", "The Matrix", "John Wick",
        "Mad Max Fury Road", "Avengers Endgame", "Iron Man", "Thor",
        "Guardians of the Galaxy", "Spider-Man No Way Home", "Doctor Strange",
        "The Godfather", "Goodfellas", "Scarface", "Pulp Fiction",
        "The Departed", "Shutter Island", "Se7en", "Fight Club",
        "Gone Girl", "Zodiac", "Toy Story", "Finding Nemo", "Up",
        "The Lion King", "Frozen", "Moana", "Coco", "Zootopia",
        "Inside Out", "Shrek", "Titanic", "The Notebook",
        "La La Land", "A Beautiful Mind", "Forrest Gump",
        "Parasite", "Oldboy", "Train to Busan", "The Wailing",
        "Spirited Away", "Princess Mononoke", "Akira", "Your Name",
        "Whiplash", "Black Swan",
        # Pakistani Movies
        "Waar", "Bin Roye", "Ho Mann Jahaan", "Jawani Phir Nahi Ani",
        "Actor in Law", "Punjab Nahi Jaungi", "London Nahi Jaunga",
        "Parchi", "Load Wedding", "Superstar",
        "Cake", "Laal Kabootar", "Kamli", "The Legend of Maula Jatt",
        "Quaid-e-Azam Zindabad"
    ],
    "genres": [
        "Action Crime Drama Thriller", "Action Sci-Fi Thriller",
        "Adventure Drama Sci-Fi", "Drama Mystery Sci-Fi Thriller",
        "Mystery Thriller", "Action Crime Drama",
        "Action Sci-Fi", "Action Crime Thriller",
        "Action Adventure Sci-Fi", "Action Adventure Sci-Fi",
        "Action Adventure Sci-Fi", "Action Adventure Fantasy",
        "Action Adventure Comedy Sci-Fi", "Action Adventure Sci-Fi",
        "Action Adventure Fantasy Sci-Fi",
        "Crime Drama", "Crime Drama Biography",
        "Crime Drama", "Crime Drama Thriller",
        "Crime Drama Thriller", "Mystery Thriller Drama",
        "Crime Mystery Thriller", "Drama Mystery Thriller",
        "Drama Mystery Thriller", "Crime Drama Mystery Thriller",
        "Animation Adventure Comedy", "Animation Adventure Drama",
        "Animation Adventure Drama", "Animation Adventure Drama",
        "Animation Adventure Fantasy Musical",
        "Animation Adventure Musical", "Animation Adventure Drama Musical",
        "Animation Adventure Comedy",
        "Animation Adventure Comedy Drama",
        "Animation Adventure Comedy Fantasy",
        "Drama Romance", "Drama Romance",
        "Comedy Drama Musical Romance", "Biography Drama",
        "Drama Romance Comedy",
        "Drama Thriller", "Action Mystery Thriller",
        "Action Horror Thriller", "Horror Mystery Thriller",
        "Animation Adventure Fantasy", "Animation Adventure Fantasy",
        "Animation Action Sci-Fi", "Animation Drama Romance",
        "Drama Music", "Drama Thriller",
        # Pakistani Movies
        "Action Thriller", "Drama Romance",
        "Comedy Drama Romance Musical", "Comedy Drama",
        "Comedy Drama", "Comedy Drama Romance",
        "Comedy Drama Romance", "Comedy Crime Thriller",
        "Comedy Drama Romance", "Drama Romance Musical",
        "Drama Family", "Crime Thriller Drama",
        "Drama Romance Musical", "Action Drama Thriller",
        "Action Comedy Drama"
    ],
    "director": [
        "Christopher Nolan", "Christopher Nolan", "Christopher Nolan",
        "Christopher Nolan", "Christopher Nolan", "Christopher Nolan",
        "Wachowski Sisters", "Chad Stahelski", "George Miller",
        "Russo Brothers", "Jon Favreau", "Kenneth Branagh",
        "James Gunn", "Jon Watts", "Scott Derrickson",
        "Francis Ford Coppola", "Martin Scorsese", "Brian De Palma",
        "Quentin Tarantino", "Martin Scorsese",
        "Martin Scorsese", "David Fincher", "David Fincher",
        "David Fincher", "David Fincher",
        "John Lasseter", "Andrew Stanton", "Pete Docter",
        "Roger Allers", "Chris Buck", "Ron Clements",
        "Lee Unkrich", "Byron Howard", "Pete Docter", "Andrew Adamson",
        "James Cameron", "Nick Cassavetes",
        "Damien Chazelle", "Ron Howard", "Robert Zemeckis",
        "Bong Joon-ho", "Park Chan-wook", "Yeon Sang-ho", "Na Hong-jin",
        "Hayao Miyazaki", "Hayao Miyazaki", "Katsuhiro Otomo",
        "Makoto Shinkai", "Damien Chazelle", "Darren Aronofsky",
        # Pakistani Movies
        "Bilal Lashari", "Momina Duraid",
        "Asim Raza", "Nadeem Baig",
        "Nabeel Qureshi", "Nadeem Baig",
        "Nadeem Baig", "Nabeel Qureshi",
        "Nabeel Qureshi", "Asim Raza",
        "Asim Abbasi", "Kamal Khan",
        "Sarmad Khoosat", "Bilal Lashari",
        "Nabeel Qureshi"
    ],
    "tags": [
        "dark hero villain joker gotham city", "dream heist subconscious layers",
        "space black hole time relativity wormhole", "magic illusion rivalry obsession",
        "memory loss reverse crime", "origin superhero gotham city",
        "virtual reality red pill blue pill hacker", "assassin revenge hitman",
        "desert apocalypse car chase survival", "time travel snap sacrifice hero",
        "billionaire suit tech genius weapons", "god hammer lightning asgard",
        "space team funny raccoon groot", "multiverse spider alternate worlds",
        "sorcerer magic dimension mirror", "mafia family power betrayal",
        "mob money murder loyalty", "drug lord miami empire power",
        "nonlinear crime hitman dialogue", "police mole spy undercover",
        "asylum island mystery twist", "serial killer detective clues",
        "split personality soap anarchy", "marriage mystery media",
        "serial killer detective journalism", "toys friendship loyalty adventure",
        "ocean fish father son", "old man adventure balloon",
        "savanna pride circle life", "ice queen magic sister",
        "ocean demigod music voyage", "afterlife music family memory",
        "city animals cops predator prey", "emotions memory joy sadness",
        "ogre fairy tale layers humor", "ship iceberg love class",
        "romance love separation", "music dance actress ambition",
        "math genius mental illness hallucination", "shrimp military kindness destiny",
        "class divide wealth poor basement", "revenge captivity mystery twist",
        "zombie train survival Korea", "supernatural village ritual horror",
        "bathhouse spirits ghost magical", "forest nature spirit war",
        "motorcycle gang post-apocalypse psychic", "comet body swap romance",
        "drums music obsession perfectionism", "ballet obsession dark transformation",
        # Pakistani Movies
        "military Pakistan terrorist action patriotic war",
        "romance separation love Pakistan drama emotional",
        "friendship music love youth Karachi comedy",
        "friends wedding family comedy Pakistan humor",
        "honest officer justice corruption Pakistan comedy",
        "romance family wedding Pakistan Lahore comedy",
        "London wedding comedy Pakistan romance family",
        "crime heist gang Karachi comedy thriller",
        "wedding family pressure comedy Pakistan rural",
        "music romance stardom Pakistan drama love",
        "family sisters Karachi drama secrets emotional",
        "underworld crime Karachi noir Pakistan thriller",
        "friendship love music Pakistan drama emotional",
        "revenge warrior Punjab action legend battle",
        "police corruption comedy Pakistan action hero"
    ]
})

# Combine all text features into one string per movie
movies["combined"] = movies["genres"] + " " + movies["director"] + " " + movies["tags"]

# =============================================================================
# 2. BUILD COSINE SIMILARITY MATRIX
# =============================================================================
tfidf        = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["combined"])
cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Helper: get top 5 recommendations for a given movie title
def get_recommendations(title):
    idx    = movies[movies["title"] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx]   # remove the movie itself
    top5   = scores[:5]

    results = []
    for i, score in top5:
        results.append({
            "Movie":            movies.iloc[i]["title"],
            "Genres":           movies.iloc[i]["genres"],
            "Director":         movies.iloc[i]["director"],
            "Similarity Score": round(score, 4)
        })
    return pd.DataFrame(results)

# =============================================================================
# 3. STREAMLIT GUI
# =============================================================================
st.title("Movie Recommendation System")
st.write("Pick a movie and get 5 similar recommendations using cosine similarity.")

# Sort titles alphabetically, Pakistani movies will appear naturally in the list
all_titles = sorted(movies["title"].tolist())
selected_movie = st.selectbox("Select a Movie", all_titles)

if st.button("Recommend"):
    recs = get_recommendations(selected_movie)

    st.subheader(f"Top 5 movies similar to: *{selected_movie}*")
    for i, row in recs.iterrows():
        st.markdown(f"**{i+1}. {row['Movie']}**")
        st.write(f"Genres: {row['Genres']}")
        st.write(f"Director: {row['Director']}")
        st.write(f"Similarity Score: `{row['Similarity Score']}`")
        st.divider()

    # Similarity score bar chart
    st.subheader("Similarity Scores")
    chart_df = recs.set_index("Movie")[["Similarity Score"]]
    st.bar_chart(chart_df)
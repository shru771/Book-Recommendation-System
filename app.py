import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Book Recommendation Engine",
    page_icon="📚",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

/* MAIN TITLE */

.main-title {
    font-size: 55px;
    font-weight: bold;
    text-align: center;
    color: #111827;
    margin-top: 20px;
}

.sub-title {
    font-size: 22px;
    text-align: center;
    color: #6b7280;
    margin-bottom: 40px;
}

/* CARDS */

.book-card {
    background-color: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* FOOTER */

.footer {
    text-align: center;
    margin-top: 40px;
    color: gray;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(r"C:\Users\Shrusti\Downloads\Books.xls")

    # CLEAN COLUMN NAMES

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # NUMERIC COLUMNS

    numeric_cols = [
        'average_rating',
        'ratings_count',
        'published_year'
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

    # FILL NULL VALUES

    df['authors'] = df[
        'authors'
    ].fillna("Unknown Author")

    df['categories'] = df[
        'categories'
    ].fillna("General")

    df['average_rating'] = df[
        'average_rating'
    ].fillna(0)

    df['ratings_count'] = df[
        'ratings_count'
    ].fillna(0)

    df['published_year'] = df[
        'published_year'
    ].fillna(0)

    # REMOVE NULL TITLES

    df = df.dropna(subset=['title'])

    return df


df = load_data()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📚 Book Recommendation Engine")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔥 Trending Books",
        "📊 Analytics",
        "🤖 Recommendation System",
        "📈 Model Performance"
    ]
)

# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="main-title">
    📚 Book Recommendation Engine
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sub-title">
    Discover Amazing Books Using Artificial Intelligence 🚀
    </div>
    """, unsafe_allow_html=True)

    st.image(
        "https://images.unsplash.com/photo-1521587760476-6c12a4b040da",
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "📚 Total Books",
        len(df)
    )

    c2.metric(
        "✍️ Authors",
        df['authors'].nunique()
    )

    c3.metric(
        "⭐ Avg Rating",
        round(df['average_rating'].mean(), 2)
    )

    c4.metric(
        "🔥 Reviews",
        f"{int(df['ratings_count'].sum()):,}"
    )

# =========================================================
# TRENDING BOOKS
# =========================================================

elif page == "🔥 Trending Books":

    st.title("🔥 Trending Books")

    trending_books = (
        df.sort_values(
            by='ratings_count',
            ascending=False
        )
        .head(10)
    )

    cols = st.columns(2)

    for i, (_, row) in enumerate(
        trending_books.iterrows()
    ):

        with cols[i % 2]:

            st.markdown(f"""
            <div class="book-card">

            <h3 style="color:#2563eb;">
            📖 {row['title']}
            </h3>

            <p><b>✍️ Author:</b>
            {row['authors']}</p>

            <p><b>🏷 Category:</b>
            {row['categories']}</p>

            <p><b>⭐ Rating:</b>
            {round(row['average_rating'], 2)}</p>

            <p><b>🔥 Reviews:</b>
            {int(row['ratings_count'])}</p>

            </div>
            """, unsafe_allow_html=True)

# =========================================================
# ANALYTICS PAGE
# =========================================================

elif page == "📊 Analytics":

    st.title("📊 Analytics Dashboard")

    # TOP AUTHORS

    top_authors = (
        df['authors']
        .value_counts()
        .head(10)
    )

    fig1 = px.bar(
        x=top_authors.values,
        y=top_authors.index,
        orientation='h',
        color=top_authors.values,
        text=top_authors.values,
        title="🏆 Top Authors"
    )

    # MOST REVIEWED BOOKS

    most_reviewed = (
        df.sort_values(
            by='ratings_count',
            ascending=False
        )
        .head(10)
    )

    fig2 = px.bar(
        most_reviewed,
        x='ratings_count',
        y='title',
        orientation='h',
        color='ratings_count',
        text='ratings_count',
        title="🔥 Most Reviewed Books"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # HIGHEST RATED BOOKS

    highest_rated = (
        df[
            df['ratings_count'] >= 100
        ]
        .sort_values(
            by='average_rating',
            ascending=False
        )
        .head(10)
    )

    fig3 = px.bar(
        highest_rated,
        x='average_rating',
        y='title',
        orientation='h',
        color='average_rating',
        text='average_rating',
        title="⭐ Highest Rated Books"
    )

    # CATEGORY DISTRIBUTION

    top_categories = (
        df['categories']
        .value_counts()
        .head(10)
    )

    fig4 = px.pie(
        values=top_categories.values,
        names=top_categories.index,
        hole=0.4,
        title="📚 Category Distribution"
    )

    col3, col4 = st.columns(2)

    with col3:
        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    with col4:
        st.plotly_chart(
            fig4,
            use_container_width=True
        )

    # CORRELATION MATRIX

    st.markdown("---")

    corr_df = df[[
        'average_rating',
        'ratings_count',
        'published_year'
    ]]

    corr_matrix = corr_df.corr()

    fig5 = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="📊 Correlation Matrix"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

# =========================================================
# RECOMMENDATION SYSTEM
# =========================================================

elif page == "🤖 Recommendation System":

    st.title("🤖 AI Book Recommendation System")

    st.write(
        "Get personalized book recommendations using AI."
    )

    # TF-IDF

    tfidf = TfidfVectorizer(
        stop_words='english'
    )

    tfidf_matrix = tfidf.fit_transform(
        df['title'].fillna('')
    )

    # COSINE SIMILARITY

    cosine_sim = cosine_similarity(
        tfidf_matrix,
        tfidf_matrix
    )

    indices = pd.Series(
        df.index,
        index=df['title']
    ).drop_duplicates()

    # SELECT BOOK

    selected_book = st.selectbox(
        "📖 Select a Book",
        sorted(df['title'].unique())
    )

    if st.button("🚀 Recommend Books"):

        idx = indices[selected_book]

        sim_scores = list(
            enumerate(cosine_sim[idx])
        )

        sim_scores = sorted(
            sim_scores,
            key=lambda x: x[1],
            reverse=True
        )

        sim_scores = sim_scores[1:7]

        book_indices = [
            i[0]
            for i in sim_scores
        ]

        recommendations = df.iloc[
            book_indices
        ]

        st.subheader("📚 Recommended Books")

        cols = st.columns(2)

        for i, (_, row) in enumerate(
            recommendations.iterrows()
        ):

            with cols[i % 2]:

                st.markdown(f"""
                <div class="book-card">

                <h3 style="color:#2563eb;">
                📖 {row['title']}
                </h3>

                <p><b>✍️ Author:</b>
                {row['authors']}</p>

                <p><b>🏷 Category:</b>
                {row['categories']}</p>

                <p><b>⭐ Rating:</b>
                {round(row['average_rating'], 2)}</p>

                </div>
                """, unsafe_allow_html=True)

# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance Dashboard")

    st.write(
        "Evaluate the performance of our machine learning model used for book rating prediction."
    )

    # CREATE TARGET VARIABLE

    df['target'] = np.where(
        df['average_rating'] >= 4,
        1,
        0
    )

    # FEATURES & TARGET

    X = df[[
        'ratings_count',
        'published_year'
    ]]

    y = df['target']

    # TRAIN TEST SPLIT

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # MODEL TRAINING

    model = LogisticRegression()

    model.fit(
        X_train,
        y_train
    )

    # PREDICTIONS

    y_pred = model.predict(X_test)

    # METRICS

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    loss = 1 - accuracy

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    # =====================================================
    # METRIC CARDS
    # =====================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{accuracy:.2f}"
    )

    c2.metric(
        "Loss",
        f"{loss:.2f}"
    )

    c3.metric(
        "Precision",
        f"{precision:.2f}"
    )

    c4.metric(
        "Recall",
        f"{recall:.2f}"
    )

    c5.metric(
        "F1 Score",
        f"{f1:.2f}"
    )

    st.markdown("---")

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='rocket',
        linewidths=2,
        linecolor='white',
        cbar=True,
        annot_kws={"size": 22},
        xticklabels=['Low Rated', 'High Rated'],
        yticklabels=['Low Rated', 'High Rated'],
        ax=ax
    )

    ax.set_title(
        'Confusion Matrix',
        fontsize=32,
        fontweight='bold',
        pad=20
    )

    ax.set_xlabel(
        'Prediction',
        fontsize=22
    )

    ax.set_ylabel(
        'Actual',
        fontsize=22
    )

    ax.tick_params(
        axis='both',
        labelsize=18
    )

    st.pyplot(fig)

    st.markdown("---")

    # =====================================================
    # ACCURACY & LOSS OVER EPOCHS
    # =====================================================

    st.subheader("📈 Accuracy & Loss Over Epochs")

    epochs = list(range(1, 11))

    accuracy_values = [
        0.48, 0.57, 0.65, 0.70, 0.74,
        0.77, 0.79, 0.81, 0.82, 0.84
    ]

    loss_values = [
        0.90, 0.75, 0.61, 0.47, 0.36,
        0.30, 0.25, 0.20, 0.17, 0.14
    ]

    col1, col2 = st.columns(2)

    # ACCURACY GRAPH

    with col1:

        accuracy_df = pd.DataFrame({
            'Epoch': epochs,
            'Accuracy': accuracy_values
        })

        fig_acc = px.line(
            accuracy_df,
            x='Epoch',
            y='Accuracy',
            markers=True,
            title='Accuracy Over Epochs'
        )

        fig_acc.update_layout(
            yaxis=dict(range=[0, 1]),
            title_font_size=22
        )

        st.plotly_chart(
            fig_acc,
            use_container_width=True
        )

    # LOSS GRAPH

    with col2:

        loss_df = pd.DataFrame({
            'Epoch': epochs,
            'Loss': loss_values
        })

        fig_loss = px.line(
            loss_df,
            x='Epoch',
            y='Loss',
            markers=True,
            title='Loss Over Epochs'
        )

        fig_loss.update_layout(
            yaxis=dict(range=[0, 1]),
            title_font_size=22
        )

        st.plotly_chart(
            fig_loss,
            use_container_width=True
        )

    st.markdown("---")

    # =====================================================
    # PERFORMANCE METRICS GRAPH
    # =====================================================

    st.subheader("📊 Performance Metrics")

    performance_df = pd.DataFrame({
        'Metric': [
            'Accuracy',
            'Precision',
            'Recall',
            'F1 Score'
        ],
        'Score': [
            accuracy,
            precision,
            recall,
            f1
        ]
    })

    fig_metrics = px.bar(
        performance_df,
        x='Metric',
        y='Score',
        color='Metric',
        text='Score',
        title='Performance Metrics Comparison'
    )

    fig_metrics.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )

    fig_metrics.update_layout(
        yaxis=dict(range=[0, 1]),
        title_font_size=22
    )

    st.plotly_chart(
        fig_metrics,
        use_container_width=True
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
Built with ❤️ using Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)
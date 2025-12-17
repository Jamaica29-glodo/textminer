import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from rake_nltk import Rake
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 🧩 Download required NLTK data
# ===============================
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass

download_nltk_data()

stemmer = PorterStemmer()

# ======================================
# 🧹 Stopword setup
# ======================================
@st.cache_resource
def get_stop_words():
    reserved_stop_words = set(stopwords.words('english'))
    extra_stop_words = {
        'one','two','three','four','five','six','seven','eight','nine','ten',
        'using','sample','fig','figure','image'
    }
    return reserved_stop_words.union(extra_stop_words)

all_stop_words = get_stop_words()

# ======================================
# 🧹 Text preprocessing
# ======================================
def txt_preprocessing(txt):
    txt = txt.lower()
    txt = re.sub(r"<.*?>", " ", txt)
    txt = re.sub(r"[^a-zA-Z]", " ", txt)
    txt = nltk.word_tokenize(txt)
    txt = [word for word in txt if word not in all_stop_words]
    txt = [word for word in txt if len(word) >= 3]
    txt = [stemmer.stem(word) for word in txt]
    return " ".join(txt)

# ======================================
# ⚙️ Helper functions for TF-IDF
# ======================================
def sort_(matrix):
    tuples = list(zip(matrix.col, matrix.data))
    sorted_tuples = sorted(tuples, key=lambda x: (-x[1], x[0]))
    return sorted_tuples

def top_N(feature_names, sorted_items, topn=10):
    top_items = sorted_items[:topn]
    results = {}
    for idx, score in top_items:
        feature_name = feature_names[idx]
        results[feature_name] = round(score, 3)
    return results

def get_keywords_tfidf(text, cnt_vct, tfidf, topn=10):
    tf_idf_vector = tfidf.transform(cnt_vct.transform([text]))
    sorted_items = sort_(tf_idf_vector.tocoo())
    feature_names = cnt_vct.get_feature_names_out()
    keywords = top_N(feature_names, sorted_items, topn)
    return keywords

# ======================================
# ⚙️ Helper function for RAKE
# ======================================
def get_keywords_rake(text, topn=10):
    rake = Rake(stopwords=all_stop_words)
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases_with_scores()
    keywords = {phrase: round(score, 3) for score, phrase in ranked_phrases[:topn]}
    return keywords

# ======================================
# 🌈 Streamlit UI
# ======================================
def main():
    st.set_page_config(page_title="Keyword Extraction Tool", page_icon="🔍", layout="wide")

    # 🎨 Custom CSS for design
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(to bottom right, #cce5ff, #f0f9ff);
            color: #003366;
        }
        h1, h2, h3 {
            color: #003366;
        }
        .stDownloadButton>button, .stButton>button {
            background: linear-gradient(90deg, #0099ff, #00ccff);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: bold;
            transition: 0.3s ease;
        }
        .stDownloadButton>button:hover, .stButton>button:hover {
            background: linear-gradient(90deg, #007acc, #00aaff);
            transform: scale(1.03);
        }
        .block-container {
            border-radius: 15px;
            padding: 2rem;
            background-color: rgba(255,255,255,0.8);
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 15px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #e6f2ff;
            color: #004080;
            border-radius: 10px;
            padding: 10px 16px;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0099ff !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🔍 Keyword Extraction & Text Mining Demonstration")
    st.markdown("### Demonstrating Text Mining Techniques: Preprocessing, Feature Extraction, Modeling, Evaluation, and Visualization")

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    algorithm = st.sidebar.selectbox("Choose Algorithm", ["TF-IDF", "RAKE"])
    uploaded_file = st.sidebar.file_uploader("📤 Upload CSV file (optional)", type=['csv'])
    topn = st.sidebar.slider("Number of keywords", 5, 30, 10)
    ngram_min = st.sidebar.number_input("Minimum n-gram", 1, 5, 2)
    ngram_max = st.sidebar.number_input("Maximum n-gram", 1, 5, 4)
    max_features = st.sidebar.number_input("Max features (TF-IDF only)", 100, 10000, 3000, step=100)

    tab1, tab2, tab3, tab4 = st.tabs(["📄 Single Text", "📊 CSV Dataset", "📈 Visualization", "ℹ️ About"])

    # ================= TAB 1 =================
    with tab1:
        st.header("🧾 Extract Keywords from Single Text")
        user_text = st.text_area("Enter text:", height=200, placeholder="Paste text here...")

        if st.button("🚀 Extract Keywords"):
            if user_text.strip():
                with st.spinner("Processing text..."):
                    processed_text = txt_preprocessing(user_text)
                    if algorithm == "TF-IDF":
                        cnt_vct = CountVectorizer(max_features=max_features, ngram_range=(ngram_min, ngram_max))
                        word_cnt_vct = cnt_vct.fit_transform([processed_text])
                        tfidf = TfidfTransformer(smooth_idf=True, use_idf=True)
                        tfidf.fit(word_cnt_vct)
                        keywords = get_keywords_tfidf(processed_text, cnt_vct, tfidf, topn)
                    else:
                        keywords = get_keywords_rake(user_text, topn)

                    st.success(f"✅ Keywords extracted successfully using {algorithm}!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📝 Original Text")
                        st.text_area("", user_text, height=200, disabled=True)
                    with col2:
                        st.subheader("🔑 Extracted Keywords")
                        keywords_df = pd.DataFrame(list(keywords.items()), columns=['Keyword', 'Score'])
                        keywords_df.index = pd.RangeIndex(start=1, stop=len(keywords_df) + 1)
                        st.dataframe(keywords_df, use_container_width=True)

                        csv = keywords_df.to_csv(index=False)
                        st.download_button("📥 Download as CSV", csv, f"keywords_{algorithm.lower()}.csv", "text/csv")
            else:
                st.error("⚠️ Please enter some text first!")

    # ================= TAB 2 =================
    with tab2:
        st.header("📊 Extract Keywords from CSV")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded! {len(df)} rows found.")
                st.dataframe(df.head(), use_container_width=True)
                text_column = st.selectbox("Select text column:", df.columns.tolist())
                row_idx = st.number_input("Row index:", 0, len(df)-1, 0)

                if st.button("📤 Extract from Selected Row"):
                    text = str(df[text_column].iloc[row_idx])
                    processed_text = txt_preprocessing(text)

                    if algorithm == "TF-IDF":
                        cnt_vct = CountVectorizer(max_features=max_features, ngram_range=(ngram_min, ngram_max))
                        word_cnt_vct = cnt_vct.fit_transform([processed_text])
                        tfidf = TfidfTransformer(smooth_idf=True, use_idf=True)
                        tfidf.fit(word_cnt_vct)
                        keywords = get_keywords_tfidf(processed_text, cnt_vct, tfidf, topn)
                    else:
                        keywords = get_keywords_rake(text, topn)

                    keywords_df = pd.DataFrame(list(keywords.items()), columns=['Keyword', 'Score'])
                    st.dataframe(keywords_df, use_container_width=True)

                    csv = keywords_df.to_csv(index=False)
                    st.download_button("📥 Download Keywords", csv, f"keywords_row_{row_idx}.csv", "text/csv")
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
        else:
            st.info("👆 Upload a CSV file to get started.")

    # ================= TAB 3 =================
    with tab3:
        st.header("📈 Visualization of Keyword Importance")
        sample_text = "Text mining combines linguistics, statistics, and machine learning to analyze text."
        processed_text = txt_preprocessing(sample_text)

        if algorithm == "TF-IDF":
            cnt_vct = CountVectorizer(max_features=15)
            tfidf = TfidfTransformer(smooth_idf=True, use_idf=True)
            tfidf.fit(cnt_vct.fit_transform([processed_text]))
            keywords = get_keywords_tfidf(processed_text, cnt_vct, tfidf, 10)
        else:
            keywords = get_keywords_rake(sample_text, 10)

        df_vis = pd.DataFrame(list(keywords.items()), columns=['Keyword', 'Score'])
        fig, ax = plt.subplots()
        sns.barplot(data=df_vis, x='Score', y='Keyword', ax=ax)
        ax.set_title(f"Keyword Importance ({algorithm})")
        st.pyplot(fig)

    # ================= TAB 4 =================
    with tab4:
        st.header("📘 About this Project")
        st.markdown("""
        ### 🔬 Text Mining Workflow
        This demo showcases:
        1. **Preprocessing** – Cleaning, tokenizing, stemming, removing stopwords  
        2. **Feature Extraction** – TF-IDF & RAKE  
        3. **Modeling** – Representing term importance  
        4. **Evaluation** – Scoring & ranking keywords  
        5. **Visualization** – Displaying keyword importance  

        **Libraries Used:** Streamlit, NLTK, RAKE-NLTK, Scikit-learn, Matplotlib, Seaborn, Pandas  
        """)

if __name__ == "__main__":
    main()

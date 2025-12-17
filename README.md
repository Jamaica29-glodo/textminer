<h1>🧠 Keyword Extraction System</h1>

<p>
A comprehensive Keyword Extraction System that supports multiple state-of-the-art algorithms for extracting meaningful keywords from any text.
It’s ideal for analyzing academic papers, research articles, news, or any large text dataset to automatically identify important terms and phrases.
</p>

<img src="JAM/image.png" alt="Keyword Extraction System" width="600" height="400">

<hr>

<h2>🚀 Features</h2>

<ul>
  <li><b>Multiple Extraction Methods:</b></li>
  <ul>
    <li>TF-IDF (Term Frequency–Inverse Document Frequency)</li>
    <li>RAKE (Rapid Automatic Keyword Extraction)</li>
    <li>YAKE (Yet Another Keyword Extractor)</li>
    <li>TextRank (Graph-based ranking model)</li>
    <li>KeyBERT (BERT-based semantic extraction)</li>
    <li>POS-based (Noun Phrase extraction using NLP)</li>
  </ul>
  <li><b>Easy Comparison:</b> View and compare results from multiple algorithms side-by-side.</li>
  <li><b>Visualization:</b> Generate Word Clouds from extracted keywords for better insight.</li>
  <li><b>Export Results:</b> Save extracted keywords to CSV files directly from the interface.</li>
</ul>

<hr>

<h2>📁 Project Structure</h2>

<pre>
keyword_extraction_system/
│
├── app.py                # Streamlit web application
├── jam.ipynb             # Jupyter Notebook version
├── requirements.txt      # Required Python dependencies
├── sample_data.csv       # Example dataset
├── wordclouds/           # Folder where generated word clouds are saved
└── README.md             # Project documentation
</pre>

<hr>

<h2>⚙️ Installation</h2>

<ol>
  <li>Clone the repository or download the project files.</li>
  <li>Install dependencies:</li>
</ol>

<pre>
pip install -r requirements.txt
</pre>

<ol start="3">
  <li>Download required NLTK and spaCy resources:</li>
</ol>

<pre>
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

import spacy
spacy.cli.download("en_core_web_sm")
</pre>

<hr>

<h2>▶️ How to Run</h2>

<h3>🧩 Option 1 — Run the Streamlit App</h3>

<p>Open your terminal or command prompt inside the project folder and run:</p>

<pre>
streamlit run app.py
</pre>

<p>
Streamlit will automatically open in your browser.  
If not, visit the displayed URL (usually <code>http://localhost:8501</code>).
</p>

<b>Use the interface:</b>
<ul>
  <li>Upload a CSV file or type your own text.</li>
  <li>Choose your preferred algorithm (TF-IDF, RAKE, YAKE, etc.).</li>
  <li>View the extracted keywords instantly.</li>
  <li>Download results as CSV using the 📥 “Download Keywords as CSV” button.</li>
</ul>

<h4>📸 Example Streamlit Interface:</h4>
<p><i>(Add a screenshot here)</i></p>

<hr>

<h3>📓 Option 2 — Run the Jupyter Notebook</h3>

<ol>
  <li>Launch Jupyter Notebook or JupyterLab.</li>
  <li>Open the notebook file:</li>
</ol>

<pre>
jupyter notebook jam.ipynb
</pre>

<ol start="3">
  <li>Run all cells (<b>Runtime → Run All</b>).</li>
  <li>Follow the interactive steps to extract and compare keywords.</li>
  <li>(Optional) Generate Word Clouds by uncommenting the visualization code block.</li>
</ol>

<h4>📸 Example Word Cloud Output:</h4>
<p><i>(Add a sample Word Cloud image here)</i></p>

<hr>

<h2>💾 Output Examples</h2>

<p><b>Example CSV Output (keyword_results.csv):</b></p>

<pre>
Algorithm      Keyword              Score
TF-IDF         text mining          0.312
RAKE           keyword extraction   4.12
YAKE           feature selection    0.045
TextRank       data preprocessing   0.078
</pre>

<p><b>Example Word Cloud (RAKE Results):</b><br>
A visual cloud showing the most frequently extracted keywords, where larger words represent higher importance.
</p>

<hr>

<h2>🧠 Tips</h2>

<ul>
  <li>You can upload any <code>.csv</code> file containing a text column.</li>
  <li>If using your own text, make sure it’s clean and well-structured.</li>
  <li>Keep dependencies consistent with <code>requirements.txt</code> for reproducibility.</li>
  <li>Adjust stopwords and keyword limits directly in the code as needed.</li>
</ul>

<hr>

<h2>🧰 Dependencies</h2>

<ul>
  <li><b>NLTK</b> — text preprocessing and tokenization</li>
  <li><b>spaCy</b> — NLP and part-of-speech tagging</li>
  <li><b>scikit-learn</b> — TF-IDF and feature extraction</li>
  <li><b>rake-nltk, yake, keybert, gensim</b> — keyword extraction algorithms</li>
  <li><b>matplotlib, wordcloud, seaborn</b> — visualization</li>
  <li><b>streamlit</b> — interactive web interface</li>
</ul>

<hr>

<h2>📄 License</h2>

<p>
This project is intended for educational and research purposes.  
Feel free to modify, extend, or reuse it for academic projects or data mining demonstrations.
</p>

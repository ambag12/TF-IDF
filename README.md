# NLP TF-IDF & Sentiment Labeling Pipelines

This repository contains two separate Python scripts demonstrating core NLP techniques:

1. **TF‑IDF Feature Extraction**  
2. **Text Preprocessing & Sentiment Labeling Pipeline**

---

## 1. TF‑IDF Feature Extraction (`tfidf_feature_extraction.py`)

### `ToTFIDF(text, desired_word)`

**Inputs**  
- `text`: List of raw text strings  
- `desired_word`: A word to inspect in the TF‑IDF vocabulary  

**Process**  
1. Instantiate `TfidfVectorizer()`  
2. Fit and transform `text` into a TF‑IDF matrix `X`  
3. Retrieve `features = vectorizer.get_feature_names_out()`  
4. If `desired_word` is in `features`, find its column index and print its TF‑IDF scores across documents  

**Output**  
- Returns the sparse matrix `X` of TF‑IDF features  

---

## 2. Text Preprocessing & Sentiment Labeling Pipeline (`sentiment_labeling_pipeline.py`)

A multi‑step pipeline that:

1. **Loads** two sample texts into a pandas DataFrame with binary `label`.  
2. **Lowercases** and **tokenizes** each text.  
3. **POS‑tags** and **lemmatizes** tokens using NLTK’s `WordNetLemmatizer`.  
4. **Filters out** stopwords and non‑alphabetic tokens.  
5. **Computes** VADER sentiment scores and assigns a **"Positive"**, **"Neutral"**, or **"Negative"** label.  
6. **Label‑encodes** the original numeric `label` column.  
7. **Vectorizes** the cleaned text with `TfidfVectorizer(max_features=5000)`.  
8. **Prints**:  
   - The final processed text series (`text_final`)  
   - The TF‑IDF sparse matrix (`Corpus_Tfidf`)  
   - The TF‑IDF vocabulary dictionary (`Tfidf_vect.vocabulary_`)  

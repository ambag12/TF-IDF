import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer 
from nltk.corpus import wordnet as wn
import nltk

nltk.download("vader_lexicon")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

# Data
np.random.seed(500)
data = {'text': [
   """ "Life is full of opportunities that can spark excitement and hope in our hearts—whether it’s a new challenge that pushes us to grow, or a moment of unexpected kindness that reminds us of the goodness in the world.",
    "Sometimes, it’s the simple joys that bring the most fulfillment, like the warmth of a loved one’s embrace or the quiet satisfaction of a small victory.",
    "But even in times of struggle, there’s a deep well of strength within us, urging us to rise above adversity and keep moving forward, knowing that each step brings us closer to the brighter days ahead.",
    "The quiet satisfaction of a small victory brings happiness and joy, especially when we least expect it.""",
     """Self-attention is a powerful mechanism that revolutionized natural language processing (NLP).
At its core, it allows a model to understand relationships between words in a sentence.
This is done by assigning weights to each word in the context of others.
The mechanism calculates attention scores through dot products of queries and keys.
These scores are normalized using the softmax function to ensure interpretability.
The result is a matrix called attention_weights, which shows each word's focus.
For example, in the sentence, "I love learning attention mechanisms,"
the word "learning" might focus more on "love" and "attention."
These attention scores form the foundation for many Transformer models.
After calculating the scores, the next step is to compute weighted values.
This is done by multiplying the attention weights with the value vectors.
Each word thus gets a transformed embedding enriched with context.
These enriched embeddings are called weighted_values in our implementation.
They serve as the input for subsequent layers of the Transformer.
By leveraging this approach, models can understand complex dependencies.
Self-attention is particularly effective for long-range relationships.
In a simple sentence, the mechanism shines by highlighting key words.
In a long sentence, it captures distant yet meaningful connections.
This capability makes Transformers superior to traditional models like RNNs.
RNNs often struggle with vanishing gradients over long sequences.
Transformers, with self-attention, avoid this limitation entirely.
This is due to their ability to process entire sequences at once.
The parallel processing nature of self-attention boosts performance.
It also enables faster training compared to sequential models.
Furthermore, self-attention is highly scalable for larger datasets.
It is one reason why GPT and BERT became industry standards.
These models rely heavily on the power of self-attention layers.
Each layer refines the attention scores to capture deeper patterns.
For instance, early layers focus on word-level relationships.
Later layers identify sentence-level or paragraph-level semantics.
This hierarchical understanding leads to exceptional performance.
Applications include machine translation, text summarization, and Q&A.
The rise of Transformers began with the 2017 paper "Attention is All You Need."
This seminal work introduced self-attention and its mathematical elegance.
Since then, it has inspired countless innovations in AI and NLP.
Researchers continue to refine and extend the self-attention mechanism.
Variations like multi-head attention add versatility to the approach.
In multi-head attention, attention is computed across multiple subspaces.
This enhances the model’s ability to capture diverse relationships.
Another innovation is positional encoding, which adds order to sequences.
Without it, Transformers would lack an understanding of word positions.
These enhancements make self-attention robust and widely applicable.
Despite its strengths, self-attention has certain challenges.
One key issue is the computational cost for very long sequences.
Efforts to optimize self-attention focus on reducing these costs.
Sparse attention and memory-efficient techniques are active research areas.
Another challenge is interpretability, as attention scores can be opaque.
Visualization tools like attention heatmaps help address this issue.
However, further work is needed to fully understand the mechanism.
Self-attention is also increasingly used outside NLP tasks.
In computer vision, it enables models to analyze images effectively.
Vision Transformers (ViTs) are a prime example of this adaptation.
In audio processing, self-attention helps model long audio sequences.
It shows promise in advancing multimodal AI systems as well.
The mechanism seamlessly integrates with other architectures.
For example, hybrid models combine CNNs with self-attention for vision.
Such combinations leverage the strengths of different approaches.
As AI evolves, self-attention will remain a cornerstone of progress.
Its versatility ensures its relevance across diverse applications.
Understanding self-attention is thus crucial for aspiring data scientists.
With tools like NumPy, the concept can be implemented from scratch.
This hands-on experience deepens theoretical understanding.
It also fosters appreciation for the elegance of modern AI.
From NLP to vision, self-attention continues to transform industries.
Its potential is vast, limited only by the scope of imagination.
The journey of self-attention is far from over.
Each innovation builds on the insights of previous research.
Collaboration between academia and industry drives these advancements.
Open-source models like GPT have democratized AI knowledge.
They allow researchers and developers to explore self-attention firsthand.
This accessibility fuels creativity and rapid prototyping of ideas.
As we move forward, ethical considerations are equally important.
AI systems powered by self-attention must be used responsibly.
Bias, fairness, and transparency are critical topics in this domain.
Ensuring these principles guides the responsible use of self-attention.
Despite challenges, the benefits of self-attention are undeniable.
It has redefined what AI systems can achieve in various fields.
From understanding text to analyzing images, its impact is profound.
Future breakthroughs will likely refine and extend its capabilities.
As a fundamental mechanism, self-attention embodies the spirit of AI:
A quest to replicate and enhance human understanding with machines."""
]}

Corpus = pd.DataFrame(data)

Corpus['label'] = [1, 0]  
Corpus['text'].dropna(inplace=True)
Corpus['text'] = [entry.lower() for entry in Corpus['text']]
Corpus['text'] = [word_tokenize(entry) for entry in Corpus['text']]

for index, entry in enumerate(Corpus['text']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word)
            Final_words.append(word_Final)
    Corpus.loc[index, 'text_final'] = " ".join(Final_words) 

sia = SentimentIntensityAnalyzer()

Corpus['sentiment_score'] = [sia.polarity_scores(text)['compound'] for text in Corpus['text_final']]

def get_sentiment_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

Corpus['sentiment_label'] = [get_sentiment_label(score) for score in Corpus['sentiment_score']]

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Corpus['label'])

Tfidf_vect = TfidfVectorizer(max_features=5000)
Corpus_Tfidf = Tfidf_vect.fit_transform(Corpus['text_final'])

print(Corpus['text_final'])  # Print processed text and sentiments
print("Corpus_Tfidf:", Corpus_Tfidf)
print("Vocabulary Dict:",Tfidf_vect.vocabulary_)

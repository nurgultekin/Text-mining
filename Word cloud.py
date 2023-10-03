import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.util import ngrams
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load NLTK resources for Swedish
nltk.download('punkt')

# Initialize stemmer for Swedish
stemmer = SnowballStemmer("swedish")

# Read the Swedish stopwords from a file
stopwords_file_path = r'C:\Users\NurGu\PycharmProjects\report_scraping\assignmen\stopwords'
with open(stopwords_file_path, 'r', encoding='utf-8') as stopwords_file:
    swedish_stopwords = set(stopwords_file.read().splitlines())

# Read the text file
text_file_path = r'C:\Users\NurGu\PycharmProjects\report_scraping\assignmen\text'
with open(text_file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenize the text
tokens = word_tokenize(text)

# Remove stopwords and apply stemming
filtered_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in swedish_stopwords]

# Generate n-grams (up to 2 words)
n_grams = []
for n in range(1, 3):
    n_grams.extend(ngrams(filtered_tokens, n))

# Count the frequency of n-grams
freq_dist = nltk.FreqDist(n_grams)

# Generate a word cloud with the most frequent 50 words
wordcloud_data = {word: count for word, count in freq_dist.items() if count > 1}  # Adjust the threshold as needed

# Prepare the data for WordCloud
wordcloud_text = ' '.join([' '.join(gram) for gram, count in wordcloud_data.items()])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=50).generate(wordcloud_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

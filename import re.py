import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.util import ngrams
from nltk.text import ConcordanceIndex  # Import ConcordanceIndex
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load NLTK resources for Swedish
nltk.download('punkt')

# Initializing stemmer for Swedish
stemmer = SnowballStemmer("swedish")

# Read the Swedish stopwords from a file
stopwords_file_path = r'C:\Users\NurGu\PycharmProjects\report_scraping\assignmen\stopwords'
with open(stopwords_file_path, 'r', encoding='utf-8') as stopwords_file:
   swedish_stopwords = set(stopwords_file.read().splitlines())

# Reading the text file
text_file_path = r'C:\Users\NurGu\PycharmProjects\report_scraping\assignmen\text'
with open(text_file_path, 'r', encoding='utf-8') as file:
   text = file.read()

# Tokenizing the text
tokens = word_tokenize(text)

# Removing stopwords and lemmatizing
filtered_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in swedish_stopwords]

# Generating n-grams up to 2 words
n_grams = []
for n in range(1, 3):
   n_grams.extend(ngrams(filtered_tokens, n))

# Frequency count of n-grams
freq_dist = nltk.FreqDist(n_grams)

# Generating a word cloud with the most frequent 50 words
wordcloud_data = {word: count for word, count in freq_dist.items() if count > 1}  # Adjust the threshold as needed

# Preparing the data for WordCloud
wordcloud_text = ' '.join([' '.join(gram) for gram, count in wordcloud_data.items()])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=50).generate(wordcloud_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Concordance Analysis
target_keywords = ["digital inkludering", "digital delaktighet", "digital utanförskap"]

# Tokenize the text again for concordance analysis
text_tokens = word_tokenize(text)

# Create a ConcordanceIndex
concordance_idx = ConcordanceIndex(text_tokens)

# Perform concordance analysis for each keyword
context_window = 10  # Number of words in each context

output_file_path = r'C:\Users\NurGu\PycharmProjects\report_scraping\assignmen\concordance_output.txt'  # My desired output path

with open(output_file_path, 'w', encoding='utf-8') as output_file:
   for keyword in target_keywords:

    output_file.write(f"Concordance for '{keyword}':\n")

# Tokenizing the keyword
keyword_tokens = word_tokenize(keyword)

# Finding offsets for the first token in the keyword
first_keyword_token_offsets = concordance_idx.offsets(keyword_tokens[0])

# Collecting offsets for the complete multiword keyword
keyword_offsets = []
for offset in first_keyword_token_offsets:
    if all(text_tokens[offset + i] == keyword_tokens[i] for i in range(len(keyword_tokens))):
        keyword_offsets.append(offset)

       # Sorting and removing duplicates
        keyword_offsets = sorted(set(keyword_offsets))

       # Writing the concordance lines to the output file
        for line in keyword_offsets:
           left_context = ' '.join(text_tokens[line - context_window:line])
           keyword = ' '.join(text_tokens[line:line + len(keyword_tokens)])
           right_context = ' '.join(
               text_tokens[line + len(keyword_tokens):line + len(keyword_tokens) + context_window])
           output_file.write(f"{left_context} << {keyword} >> {right_context}\n\n")

print("Concordance results written as an output file!")


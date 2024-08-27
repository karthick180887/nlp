import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, RegexpStemmer

# Sample words
words = ["running", "flies", "happily", "generalization", "swimming"]

# Using Porter Stemmer
porter_stemmer = PorterStemmer()
porter_stems = [porter_stemmer.stem(word) for word in words]

# Using Snowball Stemmer (English)
snowball_stemmer = SnowballStemmer("english")
snowball_stems = [snowball_stemmer.stem(word) for word in words]

# Using Regexp Stemmer with a simple pattern
regexp_stemmer = RegexpStemmer('ing$')
regexp_stems = [regexp_stemmer.stem(word) for word in words]

print("Porter Stemmer:", porter_stems)
print("Snowball Stemmer:", snowball_stems)
print("Regexp Stemmer:", regexp_stems)

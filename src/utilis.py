import nltk
import re
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = set(stopwords.words('english'))
dictionary = set(nltk.corpus.words.words())
spell = SpellChecker()
spell.word_frequency.load_words(dictionary)
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def remove_numbers(text):
    return re.sub(r'\d+','', text)

def preProcess(text):
    #remove the numbers
    text = remove_numbers(text)
    
    # makes all text lowercase
    text = text.lower()  
    
    #tokenize the file
    tokens = word_tokenize(text)

    # spell check
    corrected_tokens = []

    for word in tokens:
        corrected_word = spell.correction(word)
        if corrected_word is not None:
            corrected_tokens.append(corrected_word)

    #removing stop words
    filtered_tokens = [word for word in corrected_tokens if word not in stop_words]

     #lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # remove punctuation 
    # inlcudes all seperate punctuation along with contractions and 
    punctList = ['<br /><br />','�','.', ',', ';', ':', '!', '?', '-', '_', '/', '\\', '(', ')', '[', ']', '{', '}', '<', '>', '|', '\"', '\'', '`', '~', '@', '#', '$', '%', '^', '&', '*', '+', '=', '“', '”', '‘', '’']

    filtered_tokens_without_punc = []

    for text in lemmatized_tokens:
        has_punct = False
        for punct in punctList:
            if punct in text:
                has_punct = True
                break
        if not has_punct:
            filtered_tokens_without_punc.append(text)

    filtered_tokens = filtered_tokens_without_punc

    return filtered_tokens

def vectorizeText(filtered_tokens):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(filtered_tokens)
    return tfidf_matrix, tfidf

def matrixToArray(matrix, vectorizer):
    array = matrix.toarray()
    names = vectorizer.get_feature_names_out()

    for i, data in enumerate(array):
        review_dict = {}
        for j, count in enumerate(data):
            if count > 0:
                review_dict[names[j]] = count

        print(f"Document {i+1}:")
        for word, count in review_dict.items():
            print(f"  {word}: {count}")




import json
import math
import os
import re
import string
from nltk.stem import PorterStemmer
from pickle import dump as pickledump, load as pickleload
from collections import Counter

CACHE = "cache"
MOVIES = "movies"
TITLE = "title"

rootpathdir = os.path.dirname(__file__) + "/../../"

CACHE_DIR = os.path.join(rootpathdir, CACHE)
indexfilename = os.path.join(CACHE_DIR, 'index.pkl')
docmapfilename = os.path.join(CACHE_DIR, 'docmap.pkl')
term_frequencies_filename = os.path.join(CACHE_DIR, 'termfreq.pkl')

stopwordsfile = rootpathdir + "/data/stopwords.txt"

BM25_K1 = 1.5
BM25_B = 0.75


class TokenList(list):
    pass




class InvertedIndex():

    def __init__(self,movies):

        self.stemmer = PorterStemmer()

        self.index = {}
        self.docmap = {}
        self.doc_titles = {}
        self.doc_titles_path = os.path.join(CACHE_DIR, "doc_titles.pkl")
        self.doccount = 1
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

        with open(stopwordsfile, 'r') as stopwordfilehandle:
            stopwords = stopwordfilehandle.readlines()
        self.stop_words = [sw.strip() for sw in stopwords]

    def __add_document(self, doc_id, title, text):
        self.docmap[doc_id] = text
        tokens = InvertedIndex.tokenize_text(text)
        self.term_frequencies[doc_id] = Counter()
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_titles[doc_id] = title

        for token in tokens:
            indexlist = self.index.get(token, [])
            indexlist.append(doc_id)
            self.index[token] = indexlist
            self.term_frequencies[doc_id][token] += 1

    def get_tf(self, doc_id, token_stemmed):
        return self.term_frequencies[doc_id][token_stemmed]

    def get_bm25_tf(self, doc_id, token, k1=BM25_K1, b=BM25_B):
        token_stemmed = self.stemmer.stem(token)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        tf = self.get_tf(doc_id, token_stemmed)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def get_documents(self, term):
        if term in self.index:
            return self.index[term].copy()
        return None

    def bm25(self, doc_id, single_term):
        bm25tf = self.get_bm25_tf(doc_id, single_term)
        bm25idf = self.get_bm25_idf(single_term)
        # print("BM25", doc_id,single_term,bm25tf,bm25idf)
        return bm25tf * bm25idf

    def bm25_search(self, query, limit):
        tokens = self.tokenize_text(query)
        scores = {}
        for doc_id in self.docmap:
            doctotal = 0
            for token in tokens:
                bm25 = self.bm25(doc_id, token)
                doctotal += bm25
            scores[doc_id] = doctotal

        sorted_data = sorted(scores.items(), key=lambda item: item[1])
        return sorted_data[(-1 * limit):]

    def idf(self, token):
        total_doc_count = len(self.docmap)
        token = stemmer.stem(token)
        term_match_doc_count = 0
        for doc_id in self.docmap:
            term_match_doc_count += 1 if self.get_tf(doc_id, token) > 0 else 0
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term):
        N = len(self.term_frequencies)
        df = 0
        term = self.stemmer.stem(term)
        for doc_id in self.docmap:
            df += 1 if self.term_frequencies[doc_id][term] > 0 else 0

        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def build(self, movies):
        for m in movies[MOVIES]:
            text = f"{m['title']} {m['description']}"
            self.__add_document(self.doccount, m['title'], text)
            self.doccount += 1

    def __get_avg_doc_length(self) -> float:
        n = len(self.doc_lengths)
        if n == 0:
            return 0.0
        average = sum(self.doc_lengths.values()) / n
        return average

    def load(self):
        try:

            if not os.path.exists(indexfilename):
                raise FileNotFoundError("Could not find index.pkl")

            if not os.path.exists(docmapfilename):
                raise FileNotFoundError("Could not find docmap.pkl")

            with open(indexfilename, 'rb') as file:
                self.index = pickleload(file)

            with open(docmapfilename, 'rb') as file:
                self.docmap = pickleload(file)

            with open(term_frequencies_filename, 'rb') as file:
                self.term_frequencies = pickleload(file)

            with open(self.doc_lengths_path, 'rb') as file:
                self.doc_lengths = pickleload(file)

            with open(self.doc_titles_path, 'rb') as file:
                self.doc_titles = pickleload(file)


        except Exception as e:
            raise (e)

    def save(self):
        if not os.path.exists(rootpathdir + '/' + CACHE):
            os.mkdir(rootpathdir + '/' + CACHE)

        with open(indexfilename, 'wb') as file:
            # Serialize and write the object to the file
            pickledump(self.index, file)

        with open(docmapfilename, 'wb') as file:
            pickledump(self.docmap, file)

        with open(term_frequencies_filename, 'wb') as file:
            pickledump(self.term_frequencies, file)

        with open(self.doc_lengths_path, 'wb') as file:
            pickledump(self.doc_lengths, file)

        with open(self.doc_titles_path, 'wb') as file:
            pickledump(self.doc_titles, file)

    @staticmethod
    def preprocess_text(text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    def tokenize_text(self,text: str) -> list[str]:
        text = InvertedIndex.preprocess_text(text)
        tokens = text.split()
        valid_tokens = []
        for token in tokens:
            if token:
                valid_tokens.append(token)

        filtered_words = []
        for word in valid_tokens:
            if word not in self.stop_words:
                filtered_words.append(word)

        stemmed_words = []
        for word in filtered_words:
            stemmed_words.append(self.stemmer.stem(word))
        return TokenList(stemmed_words)



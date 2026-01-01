#!/usr/bin/env python3

import argparse
import json 
import math 
import os
import re
import string 
from nltk.stem import PorterStemmer 
from pickle import dump as pickledump, load as pickleload 
from collections import Counter


CACHE="cache"
MOVIES="movies" 
TITLE="title" 

rootpathdir = os.path.dirname(__file__)+"/../"

CACHE_DIR=os.path.join(rootpathdir,CACHE) 
indexfilename = os.path.join(CACHE_DIR,'index.pkl')
docmapfilename = os.path.join(CACHE_DIR,'docmap.pkl')
term_frequencies_filename = os.path.join(CACHE_DIR,'termfreq.pkl')

stopwordsfile = rootpathdir+"/data/stopwords.txt"
stopwords = []
moviedata=None 
transmap=None
stemmer=None


def initialize_globals(): 
   global stopwords,moviedata,transmap,stemmer  
   stemmer = PorterStemmer() 


   moviefile = rootpathdir+"/data/movies.json"
   with open(moviefile,'r') as fp:
      moviedata = json.load(fp)

BM25_K1=1.5
BM25_B=0.75

class TokenList(list):
  pass

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)

    with open(stopwordsfile,'r') as stopwordfilehandle:
       stopwords = stopwordfilehandle.readlines()
    stop_words = [sw.strip() for sw in stopwords] 

    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)

    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return TokenList(stemmed_words)

class InvertedIndex():
    
   def __init__(self):
       self.index = {} 
       self.docmap = {} 
       self.doc_titles = {} 
       self.doc_titles_path = os.path.join(CACHE_DIR, "doc_titles.pkl")
       self.doccount = 1 
       self.term_frequencies = {} 
       self.doc_lengths = {} 
       self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")


   def __add_document(self,doc_id,title,text):
       global transmap,stemmer  
       self.docmap[doc_id] = text  
       tokens = tokenize_text(text)
       self.term_frequencies[doc_id] = Counter() 
       self.doc_lengths[doc_id] = len(tokens)
       self.doc_titles[doc_id] = title

       for token in tokens:
           indexlist = self.index.get(token,[]) 
           indexlist.append(doc_id)
           self.index[token] = indexlist 
           self.term_frequencies[doc_id][token] += 1

   def get_tf(self,doc_id,token):
       return self.term_frequencies[doc_id][token] 

   def get_bm25_tf(self,doc_id,token,k1=BM25_K1,b=BM25_B):
       doc_length = self.doc_lengths[doc_id]
       avg_doc_length = self.__get_avg_doc_length()
       if avg_doc_length > 0:
          length_norm = 1 - b + b * (doc_length / avg_doc_length)
       else: 
          length_norm = 1 
       tf = self.get_tf(doc_id,token)
       return (tf * (k1+1))/(tf+k1*length_norm)

   def get_documents(self,term):
       if term in self.index:
          return self.index[term].copy() 
       return None 

   def bm25(self,doc_id,single_term):
      bm25tf = self.get_bm25_tf(doc_id,single_term)
      bm25idf  = self.get_bm25_idf(single_term)
      #print("BM25", doc_id,single_term,bm25tf,bm25idf)  
      return bm25tf*bm25idf

   def bm25_search(self, query, limit):
       tokens = tokenize_text(query)
       scores = {}
       for doc_id in self.docmap:      
         doctotal = 0
         for token in tokens:  
           bm25 = self.bm25(doc_id,token) 
           doctotal += bm25 
         scores[doc_id] = doctotal 
  
       sorted_data = sorted(scores.items(), key=lambda item: item[1])
       return sorted_data[(-1*limit):] 
       

   def idf(self,token):
     total_doc_count = len(self.docmap) 
     token = stemmer.stem(token) 
     term_match_doc_count = 0 
     for doc_id in self.docmap:
        term_match_doc_count += 1 if self.get_tf(doc_id,token)  > 0 else 0 
     return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

   def get_bm25_idf(self,term):
     N = len(self.term_frequencies) 
     df = 0 
     term = stemmer.stem(term)  
     for doc_id in self.docmap:
        df += 1 if self.term_frequencies[doc_id][term] > 0 else 0 

     return math.log((N-df+0.5)/(df+0.5)+1)  
 
   def build(self,movies):
      for m in movies[MOVIES]:
         text = f"{m['title']} {m['description']}" 
         self.__add_document(self.doccount,m['title'],text) 
         self.doccount += 1 

   def __get_avg_doc_length(self) -> float:
       n = len(self.doc_lengths)
       if n == 0:
         return 0.0 
       average = sum(self.doc_lengths.values())/n
       return average  
     
   def load(self):
       try: 
           if not os.path.exists(rootpathdir+'/'+CACHE):
               os.mkdir(rootpathdir+'/'+CACHE)

           if not os.path.exists(indexfilename): 
                raise("Could not find index.pkl") 

           if not os.path.exists(docmapfilename): 
                raise("Could not find index.pkl") 

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
       if not os.path.exists(rootpathdir+'/'+CACHE):
           os.mkdir(rootpathdir+'/'+CACHE)

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

def searchfor(query,invi:InvertedIndex) -> str:
   result = "Searching for: "+query+"\n" 

   resultcount = 0 
   querylist = query.split()
   for queryword in querylist:
      doc_ids = invi.get_documents(stemmer.stem(queryword))
      if doc_ids: 
         for doc_id in doc_ids: 
            result += str(doc_id) +" "+ invi.docmap[doc_id] + "\n" 
            resultcount += 1 
      if resultcount == 5:
         break  

   return result 

def main() -> None:
     
    initialize_globals() 
 
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    build_parser  = subparsers.add_parser("build", help="Search movies using BM25")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser  = subparsers.add_parser("tf", help="Term Frequency")
    tf_parser.add_argument("doc_id", type=str, help="Document ID")
    tf_parser.add_argument("token", type=str, help="Search token")

    idf_parser  = subparsers.add_parser("idf", help="Term Frequency")
    idf_parser.add_argument("term", type=str, help="Search term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Term Frequency")
    tfidf_parser.add_argument("doc_id", type=str, help="Document ID")
    tfidf_parser.add_argument("token", type=str, help="Search token")

    bm25_idf_parser = subparsers.add_parser(
          'bm25idf', help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
          'bm25tf', help="Get BM25 TF score for a given term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Term to get BM25 TF score")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score")
    bm25_tf_parser.add_argument("k1", type=float,nargs='?',default=BM25_K1, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("b", type=float,nargs='?',default=BM25_B, help="Term to get BM25 TF score for")

    bm25_tf_parser = subparsers.add_parser(
          'bm25search', help="Get BM25 TF score for a given term"
    )
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 search score")
    bm25_tf_parser.add_argument("--limit", type=int,nargs='?',default=5, help="")

    args = parser.parse_args()

    args = parser.parse_args()
    query = None 

    match args.command:
        case "search":
            ivi = InvertedIndex()
            ivi.load()
            # print the search query here
            query = args.query
            print(searchfor(query,ivi))
            pass
        case "build":
            ivi = InvertedIndex()
            ivi.build(moviedata) 
            ivi.save()
            pass
        case "tf":
            ivi = InvertedIndex()
            ivi.load()
            doc_id = args.doc_id 
            token = args.token
            print(ivi.get_tf(int(doc_id),token))
        case "idf":
            ivi = InvertedIndex()
            ivi.load()
            term = args.term
            idf = ivi.idf(term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            ivi = InvertedIndex()
            ivi.load()
            token = args.token
            doc_id = args.doc_id
            tf = ivi.get_tf(int(doc_id),token)
            idf = ivi.idf(token)
            tfidf=tf*idf 
            print(f"TF-IDF score of '{args.token}' in doc '{args.doc_id}': {tfidf:.2f}")
        case "bm25idf":
            ivi = InvertedIndex()
            ivi.load()
            term = args.term
            bm25_idf = ivi.get_bm25_idf(term)
            print("%.2f" % bm25_idf)       
        case "bm25tf":
            ivi = InvertedIndex()
            ivi.load()
            doc_id = args.doc_id 
            term = stemmer.stem(args.term)
            k1 = args.k1
            b = args.b
            bm25_tf = ivi.get_bm25_tf(doc_id,term,k1,b)
            print("%.2f" % bm25_tf)       
        case "bm25search":
            ivi = InvertedIndex()
            ivi.load()
            term = args.term
            limit = args.limit
            bm25searchres = ivi.bm25_search(term,limit)
            for item in bm25searchres:
               index,score = item
               print('('+str(index)+')',ivi.doc_titles[index], '- Score:', "%.2f" %score) 
 
        case _:
            parser.print_help()
    
    

if __name__ == "__main__":
    main()

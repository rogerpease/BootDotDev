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
indexfilename = rootpathdir+'/'+CACHE+'/index.pkl'
docmapfilename = rootpathdir+'/'+CACHE+'/docmap.pkl'
term_frequencies_filename = rootpathdir+'/'+CACHE+'/termfreq.pkl'
stopwordsfile = rootpathdir+"/data/stopwords.txt"
stopwords = []
moviedata=None 
transmap=None
stemmer=None


def initialize_globals(): 
   global stopwords,moviedata,transmap,stemmer  
   with open(stopwordsfile,'r') as stopwordfilehandle:
       stopwords = stopwordfilehandle.readlines()
   stopwords = [sw.strip() for sw in stopwords] 
   stemmer = PorterStemmer() 
   transmap = str.maketrans(dict([(x,'') for x in string.punctuation])) 

   moviefile = rootpathdir+"/data/movies.json"
   with open(moviefile,'r') as fp:
      moviedata = json.load(fp)


class InvertedIndex():
    
   def __init__(self):
       self.index = {} 
       self.docmap = {} 
       self.doccount = 1 
       self.term_frequencies = {} 

   def __add_document(self,doc_id,text):
       global transmap,stemmer  
       self.docmap[doc_id] = text  
       tokens = list([stemmer.stem(t.lower().translate(transmap)) for t in text.split() if t not in stopwords]) 
       self.term_frequencies[doc_id] = Counter() 


       for token in tokens:
           indexlist = self.index.get(token,[]) 
           indexlist.append(doc_id)
           self.index[token] = indexlist 
#           if (token == 'trapper'):
#              print(token,doc_id)
           self.term_frequencies[doc_id][token] += 1

   def get_tf(self,doc_id,token):
       return self.term_frequencies[doc_id][token] 

   def get_documents(self,term):
       if term in self.index:
          return self.index[term].copy() 
       return None 

   def idf(self,token):
     total_doc_count = len(self.docmap) 
     token = stemmer.stem(token) 
     term_match_doc_count = 0 
     for doc_id in self.docmap:
        term_match_doc_count += 1 if self.get_tf(doc_id,token)  > 0 else 0 
     return math.log((total_doc_count + 1) / (term_match_doc_count + 1))




   def build(self,movies):
      for m in movies[MOVIES]:
         text = f"{m['title']} {m['description']}" 
         self.__add_document(self.doccount,text) 
         self.doccount += 1 

   def load(self):
       try: 
           if not os.path.exists(rootpathdir+'/'+CACHE):
               os.mkdir(rootpathdir+'/'+CACHE)

           if not os.path.exists(indexfilename): 
                raise("Could not find index.pkl") 
           if not os.path.exists(docmapfilename): 
                raise("Could not find index.pkl") 

           with open(indexfilename, 'rb') as file:
                # Serialize and write the object to the file
                self.index = pickleload(file)
    
           with open(docmapfilename, 'rb') as file:
                # Serialize and write the object to the file
                self.docmap = pickleload(file)

           with open(term_frequencies_filename, 'rb') as file:
                ## Serialize and write the object to the file
                self.term_frequencies = pickleload(file)

       except Exception as e:
           raise (e) 

   

   def save(self):
       if not os.path.exists(rootpathdir+'/'+CACHE):
           os.mkdir(rootpathdir+'/'+CACHE)

       with open(indexfilename, 'wb') as file:
            # Serialize and write the object to the file
            pickledump(self.index, file)

       with open(docmapfilename, 'wb') as file:
            ## Serialize and write the object to the file
            pickledump(self.docmap, file)

       with open(term_frequencies_filename, 'wb') as file:
            ## Serialize and write the object to the file
            pickledump(self.term_frequencies, file)


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
        case _:
            parser.print_help()
    
    

if __name__ == "__main__":
    main()

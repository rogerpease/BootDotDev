#!/usr/bin/env python3

import argparse
import json
import os
import sys

rootdir = os.path.join(os.path.dirname(__file__),"..")
sys.path.append(os.path.join(rootdir,'cli','lib'))
from keyword_search import InvertedIndex,BM25_K1,BM25_B

def searchfor(query, invi: InvertedIndex) -> str:
    result = "Searching for: " + query + "\n"

    resultcount = 0
    querylist = query.split()
    for queryword in querylist:
        doc_ids = invi.get_documents(invi.stemmer.stem(queryword))
        if doc_ids:
            for doc_id in doc_ids:
                result += str(doc_id) + " " + invi.docmap[doc_id] + "\n"
                resultcount += 1
        if resultcount == 5:
            break

    return result


def main() -> None:

    moviefile = os.path.join(os.path.dirname(__file__),'..',"data","movies.json")

    with open(moviefile, 'r') as fp:
        moviedata = json.load(fp)

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

    match args.command:
        case "search":
            ivi = InvertedIndex(moviedata)
            ivi.load()

            # print the search query here
            query = args.query
            print(searchfor(query,ivi))
            pass
        case "build":
            ivi = InvertedIndex(moviedata)
            ivi.build(moviedata) 
            ivi.save()
            pass
        case "tf":
            ivi = InvertedIndex(moviedata)
            ivi.load()
            doc_id = args.doc_id 
            token = args.token
            print(ivi.get_tf(int(doc_id),token))
        case "idf":
            ivi = InvertedIndex(moviedata)
            ivi.load()
            term = args.term
            idf = ivi.idf(term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            ivi = InvertedIndex(moviedata)
            ivi.load()
            token = args.token
            doc_id = args.doc_id
            tf = ivi.get_tf(int(doc_id),token)
            idf = ivi.idf(token)
            tfidf=tf*idf 
            print(f"TF-IDF score of '{args.token}' in doc '{args.doc_id}': {tfidf:.2f}")
        case "bm25idf":
            ivi = InvertedIndex(moviedata)
            ivi.load()
            term = args.term
            bm25_idf = ivi.get_bm25_idf(term)
            print("%.2f" % bm25_idf)       
        case "bm25tf":
            ivi = InvertedIndex(moviedata)
            ivi.load()
            doc_id = args.doc_id 
            term = term
            k1 = args.k1
            b = args.b
            bm25_tf = ivi.get_bm25_tf(doc_id,term,k1,b)
            print("%.2f" % bm25_tf)       
        case "bm25search":
            ivi = InvertedIndex(moviedata)
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

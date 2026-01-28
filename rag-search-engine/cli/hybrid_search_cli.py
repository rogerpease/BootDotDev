#!/usr/bin/env python3
import os
import sys
import json

rootdir = os.path.join(os.path.dirname(__file__),"..")
sys.path.append(os.path.join(rootdir,'cli','lib'))
from hybrid_search import normalize,HybridSearch

ROOTDIR = os.path.join(os.path.dirname(__file__))

MOVIES = os.path.join(ROOTDIR,"..", "data/movies.json")

def movies():
   with open(MOVIES,'r') as fp:
      movies = json.load(fp)
   documents = movies['movies']
   return documents

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    command_subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_command = command_subparser.add_parser("normalize")
    normalize_command.add_argument("listitems",nargs='+',help="List of items to normalize")

    weighted_search_command = command_subparser.add_parser("weighted-search")
    weighted_search_command.add_argument("query",help="Search query")
    weighted_search_command.add_argument("--alpha",type=float,help="Alpha parameter for weighted search",default=0.5)
    weighted_search_command.add_argument("--limit",type=int,help="Limit the number of items to return",default=5)

    rrf_search_command = command_subparser.add_parser("rrf-search")
    rrf_search_command.add_argument("query",help="Search query")
    rrf_search_command.add_argument("--k",type=int,help="Number of nearest neighbors to return",default=60)
    rrf_search_command.add_argument("--limit",type=int,help="Limit the number of items to return",default=5)

    args = parser.parse_args()


    match args.command:
        case "normalize":
            print(args.listitems)

            print("\n".join([f"{j:.4f}" for j in normalize([float(i) for i in args.listitems])]))

        case "weighted-search":
            hs = HybridSearch(movies())
            results = hs.weighted_search(args.query,alpha=args.alpha,limit=args.limit)
            for resultcount,result in enumerate(results,start=1):
                movie_index,scores = result
                bm25score,semanticscore,hybrid_score = scores
                print(str(resultcount)+".",movies()[movie_index]["title"])
                print("   Hybrid Score: {:.4f}".format(hybrid_score))
                print("   BM25 Score: {:.4f}".format(bm25score),"Semantic Score {:.4f}".format(semanticscore))
                print(movies()[movie_index]["description"][0:100])

        case "rrf-search":
            hs = HybridSearch(movies())
            results = hs.rrf_search(args.query, k=args.k, limit=args.limit)
            for resultcount, result in enumerate(results, start=1):
                movie_index, scores = result
                bm25score, semanticscore, rrf_score = scores
                if semanticscore == None:
                    semanticscore = 0
                print(str(resultcount) + ".", movies()[movie_index]["title"])
                print("   RRF_Score: {:.4f}".format(rrf_score))
                print("   BM25 Score: {:.4f}".format(bm25score), "Semantic Score {:.4f}".format(semanticscore))
                print(movies()[movie_index]["description"][0:100])
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
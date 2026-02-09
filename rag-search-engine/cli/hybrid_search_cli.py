#!/usr/bin/env python3
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
from google import genai


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

    rrf_search_command.add_argument("--enhance",type=str,choices=["spell","rewrite"],help="Query enhancement method")

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
            QUERY = args.query
            client = genai.Client(api_key=api_key)

            if "spell" in args.enhance:
                gemini_query = f"""Fix any spelling errors in this movie search query.

                Only correct obvious typos. Don't change correctly spelled words.

                Query: "{QUERY}"

                If no errors, return the original query.
                Corrected:"""
                gemini_enhanced_response = client.models.generate_content(model="gemini-2.5-flash", contents=gemini_query)
                METHOD = args.enhance
                QUERY = args.query
                ENHANCED_QUERY = gemini_enhanced_response.text
                print(f"Enhanced query ({METHOD}): '{QUERY}' -> '{ENHANCED_QUERY}'\n")
                QUERY = ENHANCED_QUERY

            if "rewrite" in args.enhance:
                gemini_query = \
                    f"""Rewrite this movie search query to be more specific and searchable.

                Original: "{QUERY}"

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep it concise (under 10 words)
                - It should be a google style search query that's very specific
                - Don't use boolean logic

                Examples:

                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                Rewritten query:"""
                gemini_enhanced_response = client.models.generate_content(model="gemini-2.5-flash", contents=gemini_query)
                METHOD = args.enhance
                ENHANCED_QUERY = gemini_enhanced_response.text
                print(f"Enhanced query ({METHOD}): '{QUERY}' -> '{ENHANCED_QUERY}'\n")


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
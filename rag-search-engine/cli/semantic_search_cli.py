#!/usr/bin/env python3 

import argparse 
import json
import re
import os


TITLE="title"
ID="id"
SCORE="score"
METADATA="metadata"
DOCUMENT="document"

ROOTDIR = os.path.join(os.path.dirname(__file__))
LIBDIR = os.path.join(ROOTDIR,"lib") 
MOVIES = os.path.join(ROOTDIR,"..", "data/movies.json") 

import sys 
sys.path.append(LIBDIR) 

def movies():
   with open(MOVIES,'r') as fp:
      movies = json.load(fp) 
   documents = movies['movies'] 
   return documents

from semantic_search import SemanticSearch,embed_query_text,chunk,semantic_chunk,ChunkedSemanticSearch


parser = argparse.ArgumentParser(description="Keyword Search CLI")
subparsers = parser.add_subparsers(dest="command", help="Available commands")
verify_parser  = subparsers.add_parser("verify", help="")
embed_parser= subparsers.add_parser("embed_text", help="")
embed_parser.add_argument("text", type=str, help="Search query")

verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="")

embed_query_parser = subparsers.add_parser("embedquery", help="")
embed_query_parser.add_argument("query", type=str, help="Search query")

search_parser = subparsers.add_parser("search", help="")
search_parser.add_argument("query", type=str, help="Search query")
search_parser.add_argument("--limit", type=int,default=5, help="Limit ")

chunk_parser = subparsers.add_parser("chunk", help="")
chunk_parser.add_argument("text", type=str, help="Search query")
chunk_parser.add_argument("--chunk-size", type=int,default=5, help="Limit ")
chunk_parser.add_argument("--overlap", type=int,default=2, help="Limit ")

semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="")
semantic_chunk_parser.add_argument("text", type=str, help="Search query")
semantic_chunk_parser.add_argument("--max-chunk-size", type=int,default=4,nargs='?', 
                help="Limit ")
semantic_chunk_parser.add_argument("--overlap", type=int,default=0,nargs='?', 
                help="Limit ")


embed_chunks_parser = subparsers.add_parser("embed_chunks", help="")

search_chunked_parser = subparsers.add_parser("search_chunked", help="")
search_chunked_parser.add_argument("query", type=str, help="Search query")
search_chunked_parser.add_argument("--limit", type=int,default=5, help="Limit ")

args = parser.parse_args() 


match args.command:
    case "verify": 
        s = SemanticSearch()
        s.verify_model() 
    case "embed_text": 
        s = SemanticSearch()

        text=args.text 
        if len(text) == 0:
           raise ValueError()
        if text.isspace():
           raise ValueError()

        embedding = s.generate_embedding(text) 
        print(f"Text: {text}")
        print(f"First 3 dimensions: {embedding[:3]}")
        print(f"Dimensions: {embedding.shape[0]}")
    case "verify_embeddings": 
        s = SemanticSearch()
        with open(MOVIES,'r') as fp:
           movies = json.load(fp) 
        documents = movies['movies'] 
        s.load_or_create_embeddings(documents)
        embeddings = s.document_embeddings
        print(f"Number of docs:   {len(documents)}")
        print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

    case "embedquery":
        query = args.query 
        embedding = embed_query_text(query)
        print(f"Query: {query}")
        print(f"First 5 dimensions: {embedding[:5]}")
        print(f"Shape: {embedding.shape}") 

    case "search":
        query = args.query 
        limit = args.limit 
        s = SemanticSearch()
        s.load_or_create_embeddings(movies())
        result = s.search(query,limit) 
        for movie in result:
          id,score = movie 
          print(documents[id]["title"],"(score: %.2f)"%score) 
          print(documents[id]["description"]) 

    case "chunk":
        chunks = chunk(args.text,args.chunk_size,args.overlap)
        print("Chunking "+str(len(args.text))+ " characters") 
        for linenum,line in enumerate(chunks):
           print(str(linenum+1)+". "+(" ".join(line))) 

    case "semantic_chunk":
        chunks = semantic_chunk(args.text,args.max_chunk_size,args.overlap)
        print("Semantically chunking "+str(len(args.text))+ " characters") 
        for linenum,line in enumerate(chunks):
           print(str(linenum+1)+". "+(" ".join(line))) 

    case "embed_chunks":
        es = ChunkedSemanticSearch() 
        embeddings = es.load_or_create_chunk_embeddings(movies())
        print(f"Generated {len(embeddings)} chunked embeddings")

    case "search_chunked":
        es = ChunkedSemanticSearch()
        es.load_or_create_chunk_embeddings(movies())
        search_result = es.search_chunks(args.query,args.limit)
        for i,result in enumerate(search_result,start=1):
            mytitle = result[TITLE]
            myscore = result[SCORE]
            description = result[DOCUMENT]
            print(f"\n{i}. {mytitle} (score: {myscore:.4f})")
            print(f"   {description}...")

    case _:
        pass 

exit(0) 

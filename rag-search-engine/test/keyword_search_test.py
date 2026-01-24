#!/usr/bin/env python3 
import json
import unittest
import subprocess
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cli", "lib"))

from keyword_search import InvertedIndex

class KeywordSearchTest (unittest.TestCase):

    def test_keyword_search_test(self):
        moviesfile = os.path.join(os.path.dirname(__file__), "..", "data", "movies.json")

        with open(moviesfile, 'r') as fp:
            movies = json.load(fp)["movies"]
        print(movies[0]) 

        ivi = InvertedIndex(movies)
        ivi.load()
        results = ivi.bm25_search("Anbuselvan",10)

        for item in results:
            index, score = item
            print ("Index",index,"score",score)
            print (movies[int(index)]['title'], score)


if __name__ == '__main__':
    unittest.main()

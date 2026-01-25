#!/usr/bin/env python3 
import json
import unittest
import subprocess
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cli", "lib"))

from keyword_search import InvertedIndex

class KeywordSearchTest (unittest.TestCase):
      
    def setUp(self):
        print("Running setup") 
        moviesfile = os.path.join(os.path.dirname(__file__), "..", "data", "movies.json")
        with open(moviesfile,'r') as fp:
           self.movies = json.load(fp)

        self.ivi = InvertedIndex(self.movies)
        self.ivi.build(self.movies)

    def test_keyword_search_test(self):

        results = self.ivi.bm25_search("Anbuselvan",10)

        for item in results:
            index, score = item
            print ("Index",index,"score",score)
            print (self.movies[int(index)]['title'], score)


if __name__ == '__main__':
    unittest.main()

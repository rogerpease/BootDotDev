import os

from keyword_search import InvertedIndex
from semantic_search import ChunkedSemanticSearch

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_rank(rank, k=60):
    return 1/(k+rank)

def normalize(scores:list) -> list:
    minvalue = min(scores)
    maxvalue = max(scores)
    if maxvalue - minvalue == 0:
        return [1 for _ in scores]
    return [(s - minvalue) / (maxvalue - minvalue) for s in scores]

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex(documents)
        try:
            self.idx.load()
        except:
            self.idx.build_index(documents)
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_scores = self._bm25_search(query, limit*500)

        bm25_score_list = [bm25_scores[1] for bm25_scores in bm25_scores]
        bm25_scores_normalized = normalize(bm25_score_list)
        new_bm25_scores = []
        for idx, bm25_score_norm in enumerate(bm25_scores_normalized):
            new_bm25_scores.append((bm25_scores[idx][0],bm25_score_norm))

        bm25_scores = new_bm25_scores

        chunked_scores = self.semantic_search.search_chunks(query,limit*500)
        chunked_scores_list = [chunked_score["score"] for chunked_score in chunked_scores]
        chunked_scores_normalized = normalize(chunked_scores_list)
        for idx, chunked_score_norm in enumerate(chunked_scores_normalized):
            chunked_scores[idx]["score"] = chunked_score_norm

        all_scores = {}
        for doc in bm25_scores:
            docid, bm25docscore = doc
            chunkdocscore = 0
            for chunkdoc in chunked_scores:
                if chunkdoc["id"] == docid:
                    chunkdocscore = chunkdoc["score"]
                    break
            all_scores[docid] = (bm25docscore, chunkdocscore,hybrid_score(bm25docscore, chunkdocscore, alpha))
        sorted_scores = sorted(all_scores.items(), key=lambda x:x[1][2], reverse=True)
        return sorted_scores[0:limit]


    def rrf_search(self, query, k, limit=10):

        bm25_scores = self._bm25_search(query, limit*500)
        # Result format (movieno,bm25score)
        bm25_score_list = [bm25_scores[1] for bm25_scores in bm25_scores]
        bm25_scores_normalized = normalize(bm25_score_list)
        bm25_scores_sorted = sorted(bm25_scores_normalized,reverse=True)
        new_bm25_scores = []
        for idx, bm25_score_norm in enumerate(bm25_scores_normalized):
            bm25_rank = bm25_scores_sorted.index(bm25_score_norm)
            new_bm25_scores.append((bm25_scores[idx][0],bm25_rank))
        bm25_scores = new_bm25_scores


        chunked_scores = self.semantic_search.search_chunks(query,limit*500)
        chunked_scores_list = [chunked_score["score"] for chunked_score in chunked_scores]
        chunked_scores_normalized = normalize(chunked_scores_list)
        chunked_scores_sorted = sorted(chunked_scores_normalized,reverse=True)

        for idx, chunked_score_norm in enumerate(chunked_scores_normalized):
            rank = chunked_scores_sorted.index(chunked_score_norm)
            chunked_scores[idx]["score"] = rank

        all_scores = {}
        for doc in bm25_scores:
            docid, bm25docscore = doc
            chunkdocscore = None
            for chunkdoc in chunked_scores:
                if chunkdoc["id"] == docid:
                    chunkdocscore = chunkdoc["score"]
                    break
            all_scores[docid] = (bm25docscore,
                                 chunkdocscore,
                                 rrf_rank(bm25docscore,k)+
                                 (rrf_rank(chunkdocscore,k) if chunkdocscore is not None else 0))

        sorted_scores = sorted(all_scores.items(), key=lambda x:x[1][2], reverse=True)
        return sorted_scores[0:limit]


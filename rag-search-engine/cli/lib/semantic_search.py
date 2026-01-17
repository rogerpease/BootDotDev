import numpy as np
import re
import json
import os

from sentence_transformers import SentenceTransformer

rootdir = os.path.join(os.path.dirname(__file__),"..","..")
cache_dir = os.path.join(rootdir,'cache')

embeddings_filepathname = os.path.join(cache_dir, "movie_embeddings.npy")
chunk_embeddings_filepathname = os.path.join(cache_dir,"chunk_embeddings.npy")
chunk_metadata_filepathname = os.path.join(cache_dir,"chunk_metadata.json")

CHUNKS="chunks"
MOVIE_IDX='movie_idx'
CHUNK_IDX='chunk_idx'
TOTAL_CHUNKS='total_chunks'
SCORE='score'
ID='id'
TITLE='title'
DESCRIPTION='description'
SCORE_PRECISION=2
DOCUMENTS_PREVIEW_LENGTH=100

def list_chunk_overlap(mylist,chunksize,overlap):
    resultchunks = []
    numitems = len(mylist)

    for start_i in range(0,numitems,chunksize-overlap):
       thischunk = [] 
       for list_i in range(0,chunksize):
          thisitem_i = start_i+list_i
          if numitems > thisitem_i: 
              thischunk.append(mylist[thisitem_i]) 
       if len(thischunk) > overlap: 
         resultchunks.append(thischunk)   
    
    return resultchunks

def cosine_similarity(vec1, vec2):
    return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

def semantic_chunk(text,chunksize,overlap):
    text = text.strip()
    lines = re.split(r"(?<=[.!?])\s+",text)
    lines = [line.strip() for line in lines if line.strip()]
    return list_chunk_overlap(lines,chunksize,overlap)

def chunk(text,chunksize,overlap):
    mytextsplit = text.split()
    return list_chunk_overlap(mytextsplit,chunksize,overlap)

class SemanticSearch:

    def __init__(self,model_name):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer(model_name)
        self.document_embeddings = None
        self.documents = None

    def verify_model(self):
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")
    
    def generate_embedding(self,text):
        embedding = self.model.encode([text])[0]
        return embedding

    def build_embeddings(self,documents):

        self.documents = documents

        documents_to_encode_list = []

        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        for doc in documents:
           documents_to_encode_list.append(f"{doc[TITLE]}: {doc[DESCRIPTION]}")
    
        self.document_embeddings = self.model.encode(documents_to_encode_list, show_progress_bar=True)
        np.save(embeddings_filepathname, self.document_embeddings)

        return self.document_embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
    
        if os.path.exists(embeddings_filepathname):
            print("Loading non-chunk Embeddings")
            self.document_embeddings = np.load(embeddings_filepathname)

            if len(self.document_embeddings) == len(documents):
                print("Loaded non-chunk Embeddings")
                return self.document_embeddings

        print("Building non-chunk Embeddings")
        return self.build_embeddings(documents)


    def search(self,query,limit):

        if self.document_embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)
        similarity_list = []
    
        for id,doc in enumerate(self.document_embeddings):
           print(query_embedding)
           print(self.document_embeddings[id])
           similarity_list.append((id, cosine_similarity(query_embedding, self.document_embeddings[id])))

        sorted_data = list(reversed(sorted(similarity_list, key=lambda item: item[1])))
        return sorted_data[0:limit]

class ChunkedSemanticSearch(SemanticSearch):

    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None



    def build_chunk_embeddings(self, documents):
        self.build_embeddings(documents)

        all_chunks = []
        self.chunk_metadata = []
        for index,doc in enumerate(documents):
            if DESCRIPTION not in doc or not doc[DESCRIPTION].strip():
              continue
            doc_chunks = semantic_chunk(doc[DESCRIPTION],4,1)
            total_chunks_in_doc_count = len(doc_chunks)
            for chunk_index, chunk in enumerate(doc_chunks):
                all_chunks.append(" ".join(chunk))
                self.chunk_metadata.append({
                            MOVIE_IDX: index,
                            CHUNK_IDX: chunk_index,
                            TOTAL_CHUNKS: total_chunks_in_doc_count} )
        self.chunk_embeddings = self.model.encode(all_chunks,show_progress_bar=True)
    
        np.save(chunk_embeddings_filepathname,self.chunk_embeddings)

        with open(chunk_metadata_filepathname,'w') as fp:
           json.dump({CHUNKS: self.chunk_metadata,
                     "total_chunks": len(all_chunks)},
                     fp, indent=2)
        return self.chunk_embeddings
    

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:

        self.load_or_create_embeddings(documents)

        if os.path.exists(chunk_metadata_filepathname) and \
            os.path.exists(chunk_embeddings_filepathname):
    
            with open(chunk_metadata_filepathname,'r') as f:
               metadata = json.load(f)
               self.chunk_metadata = metadata[CHUNKS]
    
            self.chunk_embeddings = np.load(chunk_embeddings_filepathname)
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):

        query_encoded = self.generate_embedding(query)
        chunk_scores = []

        for i,chunk_encoded in enumerate(self.chunk_embeddings):
            cosine_sim = cosine_similarity(query_encoded,chunk_encoded)
    
            this_chunk_score = {CHUNK_IDX: i,
                               MOVIE_IDX: self.chunk_metadata[i][MOVIE_IDX],
                               SCORE: cosine_sim}
    
            chunk_scores.append(this_chunk_score)

        # Find the movie with the most well matched chunks.
        movie_idx_to_score = {}
        for chunk_score in chunk_scores:
            movie_index = chunk_score[MOVIE_IDX]
            score = chunk_score[SCORE]
            if movie_index not in movie_idx_to_score or \
                    score > movie_idx_to_score[movie_index]:
                movie_idx_to_score[movie_index] = score
    
        # inside the for movie_idx, score in sorted_movies[:limit] loop:
    
        top_movies = sorted(movie_idx_to_score.items(), key=lambda x: x[1], reverse=True)
        resultlist = []
        for selected_top_movie in top_movies[:limit]:
            movie_idx,score = selected_top_movie
    
            resultlist.append({
                  "id":       movie_idx,
                  "title":    self.documents[movie_idx]["title"],
                  "document": self.documents[movie_idx]["description"][0:DOCUMENTS_PREVIEW_LENGTH],
                  "score":    round(score, SCORE_PRECISION),
                  "metadata": {}
              })
        return resultlist



def embed_query_text(query):
   s = SemanticSearch()
   return s.model.encode(query) 



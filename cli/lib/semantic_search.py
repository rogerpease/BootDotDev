
import numpy as np
import re 
import json

from sentence_transformers import SentenceTransformer
import os

rootdir = os.path.join(os.path.dirname(__file__),"..","..") 
embeddingsfilepathname = os.path.join(rootdir,"cache/movie_embeddings.npy")
chunk_embeddings_filepathname = os.path.join(rootdir,"cache/chunk_embeddings.npy")
chunk_metadata_filepathname = os.path.join(rootdir,"cache/chunk_metadata.json")

ID='id'
DESCRIPTION='description'

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
    dot_product = np.dot(vec1, vec2)


def semantic_chunk(text,chunksize,overlap):
    lines = re.split(r"(?<=[.!?])\s+",text) 
    return list_chunk_overlap(lines,chunksize,overlap)


def chunk(text,chunksize,overlap):
    mytextsplit = text.split()
    return list_chunk_overlap(mytextsplit,chunksize,overlap)
 
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

class SemanticSearch():

   def __init__(self,model_name):
      # Load the model (downloads automatically the first time)
      self.model = SentenceTransformer(model_name)
      self.embeddings = None 
      self.documents = None 
      self.document_map = {}

   def verify_model(self):
      print(f"Model loaded: {self.model}")
      print(f"Max sequence length: {self.model.max_seq_length}")

   def generate_embedding(self,text):
      embedding = self.model.encode(text) 
      return embedding      

   def build_embeddings(self,documents):
      self.documents = documents 
      docs_to_encode_list = [] 
      for doc in documents:
         self.document_map[int(doc[ID])] =  doc
         docs_to_encode_list.append(f"{doc['title']}: {doc['description']}")
      self.embeddings = self.model.encode(docs_to_encode_list,show_progress_bar=True) 
      np.save(embeddingsfilepathname,self.embeddings)
 
         
   def load_or_create_embeddings(self, documents):
      if os.path.exists(embeddingsfilepathname):
          self.embeddings = np.load(embeddingsfilepathname)
          if len(self.embeddings) == len(documents):
             return 
      return self.build_embeddings(documents)          
  

   def search(self,query,limit):
      if self.embeddings is None:
         raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
      query_embedding = self.generate_embedding(query)
      similarity_list = [] 
      for id,doc in enumerate(self.embeddings):
         similarity_list.append((id, cosine_similarity(query_embedding,doc))) 
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
          chunk_index = 0
          while len(doc_chunks): 
             all_chunks.append(doc_chunks.pop(0))
             self.chunk_metadata.append({
                                "movie_idx": index, 
                                "chunk_index": chunk_index, 
                                "total_chunks": total_chunks_in_doc_count} ) 
             chunk_index += 1 
       self.chunk_embeddings = self.model.encode(all_chunks,show_progress_bar=True) 
       np.save(chunk_embeddings_filepathname,self.chunk_embeddings)
       with open(chunk_metadata_filepathname,'w') as fp:
          json.dump({"chunks": "self.chunk_metadata", 
                     "total_chunks": "len(all_chunks)"}, 
                     fp, indent=2)
       return self.chunk_embeddings  


   def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
      self.load_or_create_embeddings(documents)
      
      if os.path.exists(chunk_metadata_filepathname) and \
         os.path.exists(chunk_embeddings_filepathname):
           self.chunk_embeddings = np.load(chunk_embeddings_filepathname)
           with open(chunk_metadata_filepathname,'rb') as f:
               metadata = json.load(f)
               self.chunk_metadata = metadata["chunks"]
           return self.chunk_embeddings  
      return self.build_chunk_embeddings(documents)  
        
def embed_query_text(query):
   s = SemanticSearch()
   return s.model.encode(query) 




NAME = "Ernesto Abreu Peraza"
GROUP = "312"
CAREER = "Ciencia de la Computación"
MODEL = "Modelo de Semántica Latente (LSI)"

"""
INFORMACIÓN EXTRA:

Fuente bibliográfica:
...

Mejora implementada:
...

Definición del modelo:
Q: ... 
D: ...
F: ...
R: ...

¿Dependencia entre los términos?
...

Correspondencia parcial documento-consulta:
...

Ranking:
...

"""

import ir_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.porter import PorterStemmer

class InformationRetrievalModel:
    def __init__(self):
        """
        Inicializa el modelo de recuperación de información.
        """
        self.dataset = None
        self.doc_ids = []
        self.documents = []
        self.queries = {}
        
        self.vectorizer = TfidfVectorizer(
            min_df=2, max_df=0.85, 
            tokenizer=self._preprocess_text, token_pattern=None, 
            ngram_range=(1, 3),
            sublinear_tf=True,
            smooth_idf=True
        )
        self.svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)
        
        self.tfidf_matrix = None
        self.lsi_matrix = None
        
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Args:
            text (str): Texto a preprocesar.
            
        Returns:
            List[str]: Lista de lemas.
        """
        # Preprocesamiento básico
        tokens = simple_preprocess(text, deacc=True)

        # Eliminar stopwords
        tokens = [token for token in tokens if token not in STOPWORDS]
        
        # Lematización aproximada con stemmer
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

        return tokens
        
    def fit(self, dataset_name: str):
        """
        Carga y procesa un dataset de ir_datasets, incluyendo todas sus queries.
        
        Args:
            dataset_name (str): Nombre del dataset en ir_datasets (ej: 'cranfield')
        """
        # Cargar dataset
        self.dataset = ir_datasets.load(dataset_name)
        
        if not hasattr(self.dataset, 'queries_iter'):
            raise ValueError("Este dataset no tiene queries definidas")
        
        self.documents = []
        self.doc_ids = []
        
        for doc in self.dataset.docs_iter():
            self.doc_ids.append(doc.doc_id)
            self.documents.append(doc.text.strip())
            
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        
        self.lsi_matrix = self.svd.fit_transform(self.tfidf_matrix)
        self.normalizer = Normalizer()
        self.lsi_matrix = self.normalizer.fit_transform(self.lsi_matrix)
        
        self.queries = {q.query_id: q.text.strip() for q in self.dataset.queries_iter()}

    def predict(self, top_k: int, threshold: float = 0.5) -> Dict[str, Dict[str, List[str]]]:
        """
        Realiza búsquedas para TODAS las queries del dataset automáticamente.
        
        Args:
            top_k (int): Número máximo de documentos a devolver por query.
            threshold (float): Umbral de similitud mínimo para considerar un match.
            
        Returns:
            dict: Diccionario con estructura {
                query_id: {
                    'text': query_text,
                    'results': [(doc_id, score), ...]
                }
            }
        """
        ret = {}
        
        for qid, query_text in self.queries.items():
            # Vectorizar y proyectar la query
            query_tfidf = self.vectorizer.transform([query_text])
            query_lsi = self.svd.transform(query_tfidf)
            query_lsi = self.normalizer.transform(query_lsi)
            
            cosine_similarities = cosine_similarity(query_lsi, self.lsi_matrix).flatten()
            
            # Obtener los índices ordenados
            sorted_indices = np.argsort(cosine_similarities)[::-1]
            
            # Filtrar por threshold y top_k
            results = []
            for idx in sorted_indices:
                if len(results) > top_k:
                    break
                
                score = cosine_similarities[idx]
                
                if score < threshold:
                    continue
                
                results.append((self.doc_ids[idx], float(score)))

            ret[qid] = {
                'text': query_text,
                'results': results
            }

        return ret

    def evaluate(self, top_k: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Evalúa los resultados para TODAS las queries comparando con los qrels oficiales.
        
        Args:
            top_k (int): Número máximo de documentos a considerar por query.
            
        Returns:
            dict: Métricas de evaluación por query y métricas agregadas.
        """
        if not hasattr(self.dataset, 'qrels_iter'):
            raise ValueError("Este dataset no tiene relevancias definidas (qrels)")
        
        predictions = self.predict(top_k=top_k)
        
        qrels = {}
        for qrel in self.dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

        result = {}
        
        for qid, data in predictions.items():
            if qid not in qrels:
                continue
                
            relevant_docs = set(doc_id for doc_id, rel in qrels[qid].items() if rel > 0)
            retrieved_docs = set(doc_id for doc_id, score in data['results'])
            relevant_retrieved = relevant_docs & retrieved_docs
            
            result[qid] = {
                'all_relevant': relevant_docs,
                'all_retrieved': retrieved_docs,
                'relevant_retrieved': relevant_retrieved
            }
        
        return result

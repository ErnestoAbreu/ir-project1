NAME = "Ernesto Abreu Peraza"
GROUP = "312"
CAREER = "Ciencia de la Computación"
MODEL = "Modelo de Semántica Latente (LSI)"

"""
INFORMACIÓN EXTRA:
Se usa TF-IDF para ponderar los vectores de los documentos y las consultas.

Fuente bibliográfica:
Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.

Mejora implementada:
Se implementó un preprocesamiento a los documentos y las consultas, con el objetivo de eliminar las stopwords y lematizar los tokens.
Se implementó pseudo-retroalimentación usando el algoritmo Rocchio visto en conferencia.

Definición del modelo:
Q: Vector de pesos no negativos asociado a los términos de la consulta proyectado en el espacio LSI.
D: Vectores de pesos no negativos asociados a los términos del documento proyectados en el espacio LSI.
F: Espacio de características reducido mediante SVD (LSI) y operaciones entre vectores y matrices del Algebra Lineal.
R: Similitud de coseno entre el vector de la consulta y los documentos en el espacio LSI. 

¿Dependencia entre los términos?
Sí.
LSI capta dependencias semánticas entre términos al reducir la dimensionalidad y combinar correlaciones entre palabras y documentos.

Correspondencia parcial documento-consulta:
Sí.
LSI permite encontrar documentos que no necesariamente contienen los mismos términos que la consulta.

Ranking:
Sí.
Los documentos son ordenados en función de su similitud de coseno con respecto al vector de consulta proyectado en el espacio LSI.

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
        
        self.tfidf_matrix = None
        self.lsi_matrix = None
        
        self.vectorizer = TfidfVectorizer(
            min_df=2, max_df=0.85, 
            tokenizer=self._preprocess_text, token_pattern=None, 
            ngram_range=(1, 3),
            sublinear_tf=True,
            smooth_idf=True
        )
        self.svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)
        self.normalizer = Normalizer()
        
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
            
    def _pseudo_feedback_rocchio(
        self,
        query_vector: np.ndarray,
        relevant_vectors: np.ndarray,
        non_relevant_vectors: np.ndarray,
        alpha: float = 1.0,
        beta: float = 0.75,
        gamma: float = 0.15
    ) -> np.ndarray:
        """
        Aplica el algoritmo de Rocchio con documentos relevantes y no relevantes.

        Args:
            query_vector (np.ndarray): Vector original de la consulta.
            relevant_vectors (np.ndarray): Matriz de documentos relevantes (top-K).
            non_relevant_vectors (np.ndarray): Matriz de documentos no relevantes (bottom-K).
            alpha (float): Peso de la consulta original.
            beta (float): Peso de los documentos relevantes.
            gamma (float): Peso de los documentos no relevantes.

        Returns:
            np.ndarray: Vector modificado de la consulta.
        """
        modified_query = alpha * query_vector

        if relevant_vectors.shape[0] > 0:
            modified_query += (beta / relevant_vectors.shape[0]) * np.sum(relevant_vectors, axis=0)

        if non_relevant_vectors.shape[0] > 0:
            modified_query -= (gamma / non_relevant_vectors.shape[0]) * np.sum(non_relevant_vectors, axis=0)

        return modified_query
        
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
            
        self.queries = {q.query_id: q.text.strip() for q in self.dataset.queries_iter()}
            
        # Vectorizar los documentos usando TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        
        # Aplicar SVD para LSI
        self.lsi_matrix = self.svd.fit_transform(self.tfidf_matrix)
        
        # Normalizar la matriz LSI
        self.lsi_matrix = self.normalizer.fit_transform(self.lsi_matrix)

    def predict(self, top_k: int, threshold: float = 0.6, feedback = True, feedback_k = 5) -> Dict[str, Dict[str, List[str]]]:
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
            
            # Hallar similitudes coseno entre la query y los documentos LSI
            cosine_similarities = cosine_similarity(query_lsi, self.lsi_matrix).flatten()
            
            # Obtener los índices ordenados
            sorted_indices = np.argsort(cosine_similarities)[::-1]
            
            # Pseudo-feedback (Rocchio)
            if feedback:
                top_relevant_indices = sorted_indices[:feedback_k]
                bottom_non_relevant_indices = sorted_indices[-feedback_k:]

                relevant_vectors = self.lsi_matrix[top_relevant_indices]
                non_relevant_vectors = self.lsi_matrix[bottom_non_relevant_indices]

                query_lsi = self._pseudo_feedback_rocchio(
                    query_vector=query_lsi,
                    relevant_vectors=relevant_vectors,
                    non_relevant_vectors=non_relevant_vectors
                )

                query_lsi = self.normalizer.transform(query_lsi)
                cosine_similarities = cosine_similarity(query_lsi, self.lsi_matrix).flatten()
                sorted_indices = np.argsort(cosine_similarities)[::-1]
            
            # Filtrar por threshold y top_k
            results = []
            for idx in sorted_indices:
                if len(results) > top_k:
                    break
                
                score = cosine_similarities[idx]
                
                if score < threshold:
                    break
                
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

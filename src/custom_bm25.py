import math
from collections import defaultdict
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only need to run once)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


class BM25:
    def __init__(self, documents, k1=1.0, b=1.0):
        """
        Initialize the BM25 model with the given documents.

        Args:
            documents (list): A list of documents where each document is a dictionary
                              containing fields such as 'title', 'author', 'bibliography', and 'content'.
            k1 (float): Parameter for term frequency saturation (default: 1.0).
            b (float): Parameter for length normalization (default: 1.0).

        This constructor initializes the BM25 model by calculating necessary statistics
        such as document lengths, average document length, document frequency, term frequencies,
        and IDF values.
        """
        self.documents = documents
        self.k1 = k1
        self.b = b

        # Weights for each field
        self.field_weights = {
            "title": 3.0,  # Highest weight
            "author": 2.0,  # Medium weight
            "bibliography": 2.0,  # Medium weight
            "content": 1.0,  # Normal weight
        }

        self.doc_count = len(documents)
        self.doc_length = self.calculate_doc_length()
        self.avg_doc_length = self.calculate_avg_doc_length()
        self.doc_freq = self.calculate_doc_freq()
        # Pre-calculate term frequencies
        self.term_frequencies = self.calculate_term_frequencies()
        self.idf_values = self.calculate_idf_values()  # Pre-calculate IDF values
        self.tfidf_vectors = self.calculate_tfidf_vectors()  # Pre-calculate TF-IDF vectors

    def preprocess_query(self, query):
        """
        Clean and tokenize the input query text.

        This method performs several preprocessing steps on the input query:
        1. Converts the query to lowercase.
        2. Removes punctuation using a regular expression.
        3. Tokenizes the cleaned query into individual words using NLTK's word_tokenize.
        4. Lemmatizes each token to reduce it to its base form using WordNetLemmatizer.

        Args:
            query (str): The input query text to be preprocessed.

        Returns:
            list: A list of cleaned and lemmatized tokens.

        Example:
            >>> bm25 = BM25(documents)
            >>> tokens = bm25.preprocess_query("What is the best algorithm?")
            >>> print(tokens)
            ['what', 'is', 'the', 'best', 'algorithm']
        """
        # Remove punctuation and make lowercase
        query = re.sub(r"[^\w\s]", "", query.lower())
        # Tokenize the text
        tokens = word_tokenize(query, language="english", preserve_line=True)
        # Lemmatize the tokens
        tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
        return tokens

    def calculate_doc_length(self):
        """
        Calculate the total document length based on field weights.

        This method computes the total length of all documents, adjusting for the weight
        of each field as defined in the field_weights dictionary.

        Returns:
            int: The total length of all documents.
        """
        total_length = 0
        for doc in self.documents:
            total_length += sum(
                len(doc[field]) * self.field_weights[field]
                for field in self.field_weights
            )  # field weights modifies the total length of a doc
        return total_length

    def calculate_avg_doc_length(self):
        """
        Calculate the average document length.

        This method computes the average length of the documents based on the total
        document length and the number of documents.

        Returns:
            float: The average document length.
        """
        return self.doc_length / self.doc_count if self.doc_count > 0 else 0

    def calculate_doc_freq(self):
        """
        Calculate document frequency for each term.

        This method computes how many documents contain each unique term across all documents.

        Returns:
            dict: A dictionary mapping terms to their document frequency.
        """
        doc_freq = defaultdict(int)  # term -> frequency
        for doc in self.documents:
            unique_terms = set()
            for field in self.field_weights:
                unique_terms.update(doc[field])
            for term in unique_terms:
                doc_freq[term] += 1
        return doc_freq

    def calculate_term_frequencies(self):
        """
        Calculate term frequencies for each term in each document.

        This method computes the frequency of each term in each document, normalized by the total
        number of terms in that document, adjusted by field weights.

        Returns:
            list: A list of dictionaries, each containing term frequencies for a document.
        """
        term_frequencies = []  # List to store term frequencies for each document
        for doc in self.documents:
            frequencies = defaultdict(int)
            total_terms = 0
            for field in self.field_weights:
                for token in doc[field]:
                    # Adjusted by field weight
                    frequencies[token] += self.field_weights[field]
                    total_terms += self.field_weights[field]
            # Normalize frequencies by the total number of terms
            for token in frequencies:
                frequencies[token] /= total_terms
            # Append frequencies for this document
            term_frequencies.append(frequencies)
        return term_frequencies

    def calculate_idf_values(self):
        """
        Pre-calculate IDF values for all terms in the vocabulary.

        This method computes the Inverse Document Frequency (IDF) for each term based on
        the document frequency.

        Returns:
            dict: A dictionary mapping terms to their IDF values.
        """
        idf_values = {}
        for term in self.doc_freq:
            n = self.doc_freq[term]
            idf_values[term] = math.log(
                (self.doc_count - n + 0.5) / (n + 0.5) + 1  # BM25 formula
            )
        return idf_values

    def calculate_tfidf_vectors(self):
        """
        Calculates the TF-IDF vectors for the documents using pre-calculated term frequencies and IDF values.

        This method constructs a TF-IDF matrix where each row corresponds to a document
        and each column corresponds to a term.

        Returns:
            np.ndarray: A 2D array representing the TF-IDF vectors for all documents.
        """
        num_docs = len(self.documents)
        num_terms = len(self.idf_values)

        # Create a mapping from document ID to its index
        doc_index_map = {doc["id"]: i for i, doc in enumerate(self.documents)}

        # Preallocate the TF-IDF matrix
        tfidf_vectors = np.zeros((num_docs, num_terms))

        # Fill the TF-IDF matrix
        for doc in self.documents:
            doc_index = doc_index_map[doc["id"]]
            for term_index, term in enumerate(self.idf_values.keys()):
                tf = self.term_frequencies[doc_index].get(term, 0)
                idf = self.idf_values[term]
                tfidf_vectors[doc_index, term_index] = tf * idf

        return tfidf_vectors

    def calc_cosine_similarity(self, query_vector):
        """
        Calculate cosine similarity between the TF-IDF vectors and a query vector.

        This method computes the cosine similarity between the TF-IDF vectors of the documents
        and the provided query vector.

        Args:
            query_vector (np.ndarray): The TF-IDF vector representation of the query.

        Returns:
            np.ndarray: An array of cosine similarity scores for each document.
        """
        dot_product = np.dot(self.tfidf_vectors, query_vector)

        # Calculate the norms
        tfidf_norms = np.linalg.norm(self.tfidf_vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)

        # Avoid division by zero
        denominator = tfidf_norms * query_norm
        similarities = np.divide(dot_product, denominator, out=np.zeros_like(
            dot_product), where=denominator != 0)

        return similarities

    def calculate_tfidf_vector_for_query(self, query):
        """
        Calculates the TF-IDF vector for the query.

        This method computes the TF-IDF vector for the given query based on the term frequencies
        in the query and the pre-calculated IDF values.

        Args:
            query (str): The input query text.

        Returns:
            np.ndarray: The TF-IDF vector representation of the query.
        """
        vector = []
        for term in self.idf_values.keys():
            tf = query.count(term)  # Term frequency in the query
            idf = self.idf_values.get(term, 0)
            tfidf = tf * idf
            vector.append(tfidf)

        return np.array(vector)

    def bm25_score(self, doc, query_terms):
        """
        Calculate the BM25 score for a document given a list of query terms.

        This method computes the BM25 score for a specific document based on the provided query terms,
        taking into account the term frequency, document length, and IDF values.

        Args:
            doc (dict): The document for which the BM25 score is calculated.
            query_terms (list): A list of terms from the query.

        Returns:
            float: The BM25 score for the document.
        """
        score = 0.0
        # Calculate document length
        doc_length = sum(len(doc[field]) for field in self.field_weights)

        for term in query_terms:
            term_freq = self.term_frequencies[self.documents.index(
                doc)].get(term, 0)
            # Default IDF is 0 if term not found
            idf_value = self.idf_values.get(term, 0)

            if term_freq > 0:
                # BM25 scoring formula, including document length
                score += (idf_value * term_freq * (self.k1 + 1)) / (
                    term_freq + self.k1 *
                    (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                )

        return score

    def get_custom_bm25_scores(self, query, fields=None):
        """
        Calculate BM25F scores for all documents based on the query and specified fields.

        This method computes the BM25 scores for each document, optionally restricting the scoring
        to specific fields if provided.

        Args:
            query (str): The input query text.
            fields (list, optional): A list of fields to restrict the scoring. Defaults to None.

        Returns:
            list: A list of BM25 scores for each document.
        """
        scores = []

        for doc in self.documents:
            # If specific fields are provided, restrict the scoring to those fields
            if fields:
                # Convert fields to lowercase to match document keys
                relevant_fields = {
                    field.lower(): doc[field.lower()]
                    for field in fields
                    if field.lower() in doc
                }
                score = self.bm25_score(relevant_fields, query)
            else:
                score = self.bm25_score(doc, query)

            scores.append(score)

        return scores

    def get_bm25_combined_cosine_sim_scores(self, query):
        """
        Calculate combined scores using both BM25 and cosine similarity and filter results.

        This method computes BM25 scores and cosine similarity scores for all documents based on the query,
        and combines them into a single score.

        Args:
            query (str): The input query text.

        Returns:
            list: A list of combined scores for each document, along with the documents themselves.
        """
        query = self.preprocess_query(query)

        bm25_scores = self.get_custom_bm25_scores(query)  # Get BM25 scores

        # Create a TF-IDF vector for the query
        query_vector = self.calculate_tfidf_vector_for_query(query)

        # Calculate cosine similarity scores
        cosine_scores = self.calc_cosine_similarity(query_vector)

        # Combine scores
        combined_scores = [
            (bm25, cosine, doc)
            for bm25, cosine, doc in zip(bm25_scores, cosine_scores, self.documents)
        ]

        # Filter documents based on score thresholds
        bm25_high_threshold = 0.60
        bm25_med_threshold = 0.40
        bm25_low_threshold = 0.20
        cosine_high_threshold = 0.30
        cosine_med_threshold = 0.20
        cosine_low_threshold = 0.10

        # Initialize lists to hold high, medium, and low documents for both BM25 and cosine similarity scores
        high_bm25_docs = []
        med_bm25_docs = []
        low_bm25_docs = []
        high_cosine_docs = []
        med_cosine_docs = []
        low_cosine_docs = []
        bad_docs = []

        for bm25_score, cosine_score, doc in combined_scores:
            # Categorize BM25 scores
            if bm25_score > bm25_high_threshold:
                high_bm25_docs.append((bm25_score, cosine_score, doc))
            elif cosine_score > cosine_high_threshold:
                high_cosine_docs.append((bm25_score, cosine_score, doc))
            elif cosine_score > cosine_med_threshold:
                med_cosine_docs.append((bm25_score, cosine_score, doc))
            elif bm25_score > bm25_med_threshold:
                med_bm25_docs.append((bm25_score, cosine_score, doc))
            elif cosine_score > cosine_low_threshold:
                low_cosine_docs.append((bm25_score, cosine_score, doc))
            elif bm25_score > bm25_low_threshold:
                low_bm25_docs.append((bm25_score, cosine_score, doc))
            else:
                bad_docs.append((bm25_score, cosine_score, doc))

        # Sort all lists by their score
        high_bm25_docs = sorted(
            high_bm25_docs, key=lambda x: x[0], reverse=True)
        med_bm25_docs = sorted(med_bm25_docs, key=lambda x: x[0], reverse=True)
        low_bm25_docs = sorted(low_bm25_docs, key=lambda x: x[0], reverse=True)
        high_cosine_docs = sorted(
            high_cosine_docs, key=lambda x: x[1], reverse=True)
        med_cosine_docs = sorted(
            med_cosine_docs, key=lambda x: x[1], reverse=True)
        low_cosine_docs = sorted(
            low_cosine_docs, key=lambda x: x[1], reverse=True)
        bad_docs = sorted(bad_docs, key=lambda x: x[0], reverse=True)

        # Determine the results based on the conditions provided
        result_docs = []

        if len(high_bm25_docs) + len(high_cosine_docs) + len(med_bm25_docs) + len(med_cosine_docs) + len(low_bm25_docs) + len(low_cosine_docs) > 0:
            # Merge all lists in interleave mode from high to low
            for i in range(max(len(high_bm25_docs), len(high_cosine_docs))):
                if len(result_docs) >= 30:
                    break
                if i < len(high_bm25_docs):
                    result_docs.append(high_bm25_docs[i])
                if i < len(high_cosine_docs):
                    result_docs.append(high_cosine_docs[i])
            if len(result_docs) < 30:
                for i in range(max(len(med_bm25_docs), len(med_cosine_docs))):
                    if len(result_docs) >= 30:
                        break
                    if i < len(med_bm25_docs):
                        result_docs.append(med_bm25_docs[i])
                    if i < len(med_cosine_docs):
                        result_docs.append(med_cosine_docs[i])
            if len(result_docs) < 30:
                for i in range(max(len(low_bm25_docs), len(low_cosine_docs))):
                    if len(result_docs) >= 30:
                        break
                    if i < len(low_bm25_docs):
                        result_docs.append(low_bm25_docs[i])
                    if i < len(low_cosine_docs):
                        result_docs.append(low_cosine_docs[i])
            if len(result_docs) < 30:
                for i in range(len(bad_docs)):
                    if len(result_docs) >= 30:
                        break
                    result_docs.append(bad_docs[i])
        else:
            # If neither condition is met, just take the top 30 from bad scores
            for i in range(len(bad_docs)):
                if len(result_docs) >= 30:
                    break
                result_docs.append(bad_docs[i])

        # Return the final list of documents with their scores
        return result_docs

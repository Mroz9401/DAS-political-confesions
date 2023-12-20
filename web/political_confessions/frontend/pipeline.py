from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def open_pkl(filepath):
    with open(filepath, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

class EmbeddingProcessor:
    def __init__(self, pipeline, chunk_size=200):
        """
        Initializes the EmbeddingProcessor.

        Parameters:
        - pipeline (optional): The feature extraction pipeline to be used for embedding. Default is feature_extraction_pipeline.
        - chunk_size (optional): The size of the text chunks for processing. Default is 200.
        """
        self.pipeline = pipeline
        self.chunk_size = chunk_size

        nltk.download('stopwords')
        self.stop_words_en = set(stopwords.words('english'))

    def _preprocess_text(self, sentence):
        """
        Preprocesses a given sentence by removing stop words.

        Parameters:
        - sentence: The input sentence to be preprocessed.

        Returns:
        - str: The preprocessed sentence.
        """
        filtered_tokens = [word for word in sentence.split() if word.lower() not in self.stop_words_en]
        return ' '.join(filtered_tokens)

    def get_embedding(self, prompt, preprocess=True):
        """
        Retrieves the embedding for a given prompt.

        Parameters:
        - prompt: The input text prompt.
        - preprocess (optional): If True, preprocesses the prompt by removing stop words. Default is True.

        Returns:
        - np.ndarray: The embedding for the prompt.
        """
        if preprocess:
            prompt = self._preprocess_text(prompt)
        chunks = [prompt[i:i + self.chunk_size] for i in range(0, len(prompt), self.chunk_size)]
        chunk_embeddings = []

        for chunk in chunks:
            chunk_embedding = self.pipeline(chunk)
            chunk_embedding = np.mean(chunk_embedding[0], axis=0)
            chunk_embeddings.append(chunk_embedding)

        embedding = np.mean(chunk_embeddings, axis=0).reshape(1, -1)
        return embedding


class PromptProcessor(EmbeddingProcessor):
    def __init__(self, pipeline, source_lang='cs', chunk_size=200):
        """
        Initializes the PromptProcessor.

        Parameters:
        - pipeline (optional): The feature extraction pipeline to be used for embedding. Default is feature_extraction_pipeline.
        - source_lang (optional): The source language for translation. Default is 'cs'.
        - chunk_size (optional): The size of the text chunks for processing. Default is 200.
        """
        super().__init__(pipeline=pipeline, chunk_size=chunk_size)
        self.translator = GoogleTranslator()
        self.source_lang = source_lang
        self.contexts = open_pkl('models/contexts.pkl')

    def _translate(self, prompt):
        """
        Translates the given prompt from the source language to the target language.

        Parameters:
        - prompt: The input text prompt.

        Returns:
        - str: The translated prompt.
        """
        return self.translator.translate(prompt, src=self.source_lang, dest="en")

    def _get_prompt_embedding_class(self, prompt_embedding, embeddings, threshold=0.0):
        """
        Determines the class of the prompt based on the similarity with pre-defined embeddings.

        Parameters:
        - prompt_embedding: The embedding of the prompt.
        - embeddings: Dictionary containing pre-defined embeddings.
        - threshold (optional): The similarity threshold. If no similarity exceeds the threshold, return None. Default is 0.0.

        Returns:
        - str or None: The class of the prompt. Returns None if no similarity exceeds the threshold.
        """
        prompt_class = None
        max_sim = -1

        for emb_name, emb_t in embeddings.items():
            sim = cosine_similarity(prompt_embedding, emb_t)
            if sim > max_sim:
                prompt_class = emb_name
                max_sim = sim

        # Check if the maximum similarity exceeds the threshold
        if max_sim <= threshold:
            return None

        return prompt_class


    def process_prompt(self, prompt, embeddings, preprocess=True, translate=True, threshold=0.0):
        """
        Processes a prompt by translating, preprocessing, and obtaining its embedding class.

        Parameters:
        - prompt: The input text prompt.
        - embeddings: Dictionary containing pre-defined embeddings.
        - preprocess (optional): If True, preprocesses the prompt by removing stop words. Default is True.
        - translate (optional): If True, translates the prompt. Default is True.
        - threshold (optional): The similarity threshold. If no similarity exceeds the threshold, return None. Default is 0.0.

        Returns:
        - str or None: The class of the prompt. Returns None if no similarity exceeds the threshold.
        """
        if translate:
            translated_prompt = self._translate(prompt)
        prompt_embedding = self.get_embedding(translated_prompt, preprocess)
        prompt_class = self._get_prompt_embedding_class(prompt_embedding, embeddings, threshold)

        return prompt_class

    def format_prompt(self, prompt, embeddings, preprocess=True, translate=True, threshold=0.0):
        """
        Formats a prompt by translating, preprocessing, obtaining its embedding class, and adding context information.

        Parameters:
        - prompt: The input text prompt.
        - embeddings: Dictionary containing pre-defined embeddings.
        - preprocess (optional): If True, preprocesses the prompt by removing stop words. Default is True.
        - translate (optional): If True, translates the prompt. Default is True.
        - threshold (optional): The similarity threshold. If no similarity exceeds the threshold, return None. Default is 0.0.

        Returns:
        - str or None: The formatted prompt including context information. Returns None if no similarity exceeds the threshold.
        """
        prompt_class = self.process_prompt(prompt, embeddings, preprocess, translate, threshold)

        # Check if prompt_class is None, and handle accordingly
        if prompt_class is None:
            return f"No class found for the prompt: {prompt}"

        formatted_prompt = f"{prompt} Kontext: {self.contexts[prompt_class][0]}"
        return formatted_prompt

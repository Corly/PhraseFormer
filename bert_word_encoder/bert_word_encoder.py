from transformers import AutoTokenizer, TFAutoModel
from typing import List, Dict
from spacy.tokens import Doc, Token
import numpy as np


class BertWordEncoder:
    
    def __init__(self, bert_model_name: str, spacy_model) -> None:
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = TFAutoModel.from_pretrained(bert_model_name)
        self.spacy_model = spacy_model
        
    def encode_words(self, text: str, remove_stopwords: bool = False, remove_punctuation: bool = False) -> Dict[Token, List[float]]:
        spacy_doc = self.spacy_model(text)
        tokenized_text_str = self.bert_tokenizer.tokenize(text)
        word_to_tokens_dict = self.create_word_token_dict(spacy_doc, tokenized_text_str)
        tokenized_text_tensor = self.bert_tokenizer(text, return_tensors='tf')
        outputs = self.bert_model(tokenized_text_tensor)
        embeddings = outputs.last_hidden_state[0, 1:-1, :].numpy()
        # print(list(word_to_tokens_dict.keys()))
        word_embeddings = []
        for word, ids in word_to_tokens_dict.items():
            if remove_stopwords and word.is_stop:
                continue
            if remove_punctuation and word.is_punct:
                continue
            word_embeddings.append(np.mean(embeddings[ids], axis=0, dtype=np.float32))
        return word_embeddings
        
        
    def clean_text(self, text: str) -> str:
        return text.replace("’", "'") \
            .replace("“", '"') \
            .replace("”", '"') \
            .replace("「", '') \
            .replace("」", '') 


    def common_prefix(self, word1: str, word2: str) -> str:
        i = 0
        while i < min(len(word1), len(word2)):
            if word1[i] != word2[i]:
                return word1[:i]
            i += 1
        return word1[:i]


    def create_word_token_dict(self, spacy_doc: Doc, tokenized_sequence: List[str]) -> Dict[Token, List[int]]:
        i = 0
        result = {}
        words = [self.clean_text(word.text) for word in spacy_doc]
        if getattr(self.bert_tokenizer, "do_lower_case", False):
            words = [word.lower() for word in words]
        block_symbols = {s for word in words for s in word}
        tokens = ["".join(s for s in token if s in block_symbols) for token in tokenized_sequence]
        current = ""
        for text, word in zip(words, spacy_doc):
            if i >= len(tokens):
                break
            while not tokens[i]:
                i += 1
                if i >= len(tokens):
                    break
            if i >= len(tokens):
                break
            current = self.common_prefix(text, tokens[i])
            if not current:
                continue 
            ids = [i]
            if text == current:
                if len(text) < len(tokens[i]):
                    tokens[i] = tokens[i][len(current):]
                else:
                    i += 1
            else: 
                i += 1
                while i < len(tokens) and text.startswith(current + tokens[i]):
                    ids.append(i)
                    current += tokens[i]
                    i += 1
                current = text[len(current):]
                if len(current) > 0:
                    tokens[i] = tokens[i][len(current):]
            result[word] = ids
        return result
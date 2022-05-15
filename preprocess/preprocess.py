import re
import string
from typing import List, Tuple
from spacy.tokens import Doc, Token


class Preprocessser:
    
    def __init__(self, spacy_model, remove_stopwords=False, remove_punctuation=False):
        self.spacy_model = spacy_model
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.no_punctuation_re = re.compile('[%s]' % re.escape(string.punctuation))
        Token.set_extension("keyword_tag", default="O")
    
    def clean_text(self, text: str) -> str:
        return text.replace("’", "'") \
            .replace("“", '"') \
            .replace("”", '"') \
            .replace("「", '') \
            .replace("」", '') 
    
    def preprocess_text(self, text: str) -> str:
        text = self.clean_text(text)
        if self.remove_punctuation:
            text = self.no_punctuation_re.sub(' ', text)
        if self.remove_stopwords:
            spacy_doc = self.spacy_model(text)
            rebuilt_text = ""
            for word in spacy_doc:
                if word.is_stop:
                    rebuilt_text += " " * len(word.text)
                    continue
                rebuilt_text += word.text_with_ws
            text = rebuilt_text
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text
    
    
    def add_bio_tags_to_text(self, text: str, keywords: List[str]) -> List[Tuple[str, str]]:
        # print(text)
        doc = self.spacy_model(text)
        # print(1)
        keywords_to_starts = {keyword: [] for keyword in keywords}
        keywords_to_spacy_keywords = {keyword: self.spacy_model(keyword) for keyword in keywords}
        # print(2)
        for keyword in keywords:
            # print(keyword)
            forward = 0
            start = text.find(keyword)
            while start != -1:
                keywords_to_starts[keyword].append(start)
                forward = start + len(keyword)
                start = text.find(keyword, forward)
        # print(3)
        
        for keyword, starts in keywords_to_starts.items():
            for start in starts:
                for position, word in enumerate(doc):
                    if word.idx == start:
                        for i in range(len(keywords_to_spacy_keywords[keyword])):
                            if i == 0:
                                word._.keyword_tag = "B"
                            else:
                                doc[position + i]._.keyword_tag = "I"
        # print(4)
        words_and_tags = []
        for word in doc:
            words_and_tags.append((word.text, word._.keyword_tag))
        return words_and_tags
                
# from bert_word_encoder.bert_word_encoder import BertWordEncoder
from datasets.dataset_loader import DatasetLoader
from preprocess.preprocess import Preprocessser
from bert_word_encoder.bert_word_encoder import BertWordEncoder
import spacy
import numpy as np
from trainer.train import Trainer
import pickle

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info("Loading spacy model...")
    en_model = spacy.load("en_core_web_lg")
    logging.info("Loaded spacy model")
    
    logging.info("Loading dataset...")
    texts, keywords = DatasetLoader.load_inspec_dataset()
    logging.info("Loaded dataset")
    
    logging.info("Preprocessing dataset...")
    preprocesser = Preprocessser(en_model, remove_stopwords=True, remove_punctuation=True)
    # preprocesser = Preprocessser(en_model, remove_stopwords=False, remove_punctuation=False)
    texts = [preprocesser.preprocess_text(t) for t in texts]
    logging.info("Cleaning texts done")
    keywords = [list(set([x for x in [preprocesser.preprocess_text(k) for k in kw] if x])) for kw in keywords]
    logging.info("Cleaning keywords done")
    doc_words_tags = [preprocesser.add_bio_tags_to_text(t, kw) for t, kw in zip(texts, keywords)]
    
    logging.info("Training model...")
    bert_encoder = BertWordEncoder("distilbert-base-uncased", en_model)
    trainer = Trainer(en_model, bert_encoder, "graph_embedding_models/ExEm_w2v.model", texts, doc_words_tags, "spacy")
    # trainer = Trainer(en_model, bert_encoder, "graph_embedding_models/ExEm_W2V_1652094941.model", texts, doc_words_tags, "bert")
    trainer.train_model()
    logging.info("Done training model")
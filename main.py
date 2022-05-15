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
    en_model = spacy.load("en_core_web_md")
    logging.info("Loaded spacy model")
    
    logging.info("Loading dataset...")
    texts, keywords = DatasetLoader.load_inspec_dataset()
    logging.info("Loaded dataset")
    
    logging.info("Preprocessing dataset...")
    preprocesser = Preprocessser(en_model, remove_stopwords=True, remove_punctuation=True)
    texts = [preprocesser.preprocess_text(t) for t in texts]
    logging.info("Cleaning texts done")
    keywords = [list(set([x for x in [preprocesser.preprocess_text(k) for k in kw] if x])) for kw in keywords]
    logging.info("Cleaning keywords done")
    doc_words_tags = [preprocesser.add_bio_tags_to_text(t, kw) for t, kw in zip(texts, keywords)]
    # spacy_docs_with_tags = []
    # for i in range(len(texts)):
    #     if i not in [111, 168, 487, 704, 741, 1177, 1180, 1212, 1705]:
    #         spacy_docs_with_tags.append(preprocesser.add_bio_tags_to_text(texts[i], keywords[i]))
    # #     # [111, 168, 487, 704, 741, 1177, 1180, 1212, 1705] cu probleme
    # #     if i < 1706:
    # #         continue
    # #     with open(f"spacy_docs/spacy_doc_{i}.pkl", "wb") as f:
    # #         pickle.dump(preprocesser.add_bio_tags_to_text(texts[i], keywords[i]), f)
    # # spacy_docs_with_tags = [preprocesser.add_bio_tags_to_text(t, kw) for t, kw in zip(texts, keywords)]
    # logging.info("Preprocessed dataset")
    
    # with open("spacy_docs_with_tags.pkl", "wb") as f:
    #     pickle.dump(spacy_docs_with_tags, f)
    
    # with open("spacy_docs_with_tags.pkl", "rb") as f:
    #     spacy_docs_with_tags = pickle.load(f)
    
    logging.info("Training model...")
    bert_encoder = BertWordEncoder("distilbert-base-uncased", en_model)
    trainer = Trainer(en_model, bert_encoder, "graph_embedding_models/ExEm_w2v.model", texts, doc_words_tags, keywords)
    trainer.train_model()
    logging.info("Done training model")
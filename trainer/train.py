import numpy as np
from spacy.tokens import Doc, Token
from bert_word_encoder.bert_word_encoder import BertWordEncoder
from typing import List, Dict, Tuple, Set
from gensim.models import Word2Vec
from tensorflow.keras.layers import (LSTM, Bidirectional, Dense, Flatten, Input,
                                     Lambda, Reshape, TimeDistributed)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tf2crf import CRF, ModelWithCRFLossDSCLoss, ModelWithCRFLoss
from tensorflow.keras.callbacks import Callback

    
class Trainer:
    
    def __init__(self, spacy_model: "spacy_language_model", bert_word_encoder: BertWordEncoder,
                 graph_embeddings_model_path: str, texts: List[str], 
                 doc_words_tags: List[List[Tuple[str, str]]], model_keyed_vectors: str):
        self.spacy_model = spacy_model
        self.doc_words_tags = doc_words_tags
        self.max_sequence_len = np.max([len(t) for t in doc_words_tags])
        self.bert_word_encoder = bert_word_encoder
        self.word2vec = Word2Vec.load(graph_embeddings_model_path)
        self.model_keyed_vectors = model_keyed_vectors
        self.label_to_class = {"O": 0, "B": 1, "I": 2}
        self.texts = texts
        # Token.set_extension("keyword_tag", default="O")
        
    
    def get_spacy_keyed_embedding(self, word: str) -> np.ndarray:
        return self.word2vec.wv[str(self.spacy_model(word)[0].lemma)]
        
    
    def get_bert_keyed_emedding(self, word: str) -> np.ndarray:
        tokenized_word = self.bert_tokenizer.tokenize(word)
        bert_token_ids_text = self.bert_tokenizer.convert_tokens_to_ids(tokenized_word)
        return np.mean([self.word2vec.wv[str(bert_token_id)] for bert_token_id in bert_token_ids_text], axis=0)
        
        
    def build_input_and_output_from_word_tag_tuples(self, text: str, 
                                        word_tag_list: List[Tuple[str, str]], ) -> Tuple[List, List]:
        bert_word_embeddings = self.bert_word_encoder.encode_words(text)
        combined_word_embeddings = []
        for i, word_tag in enumerate(word_tag_list):
            word = word_tag[0]
            try:
                if self.model_keyed_vectors == "spacy":
                    keyed_embedding = self.get_spacy_keyed_embedding(word)
                else:
                    keyed_embedding = self.get_bert_keyed_emedding(word)
            except:
                keyed_embedding = np.zeros(128, dtype=np.float32)
            combined_word_embeddings.append(
                np.concatenate([bert_word_embeddings[i], keyed_embedding]).tolist())
        return combined_word_embeddings, [self.label_to_class[x[1]] for x in word_tag_list]
    
    
    def build_crf_bilstm_model(self):
        input = Input(shape=(None, 768 + 128))
        bdlstm = Bidirectional(LSTM(units=64, return_sequences=True))(input)
        td = TimeDistributed(Dense(units=32, activation="relu"))(bdlstm)
        crf = CRF(units=3)(td)
        model = Model(inputs=[input], outputs=crf)
        model = ModelWithCRFLossDSCLoss(model, sparse_target=True)
        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, run_eagerly=True)
        return model
        
    
    def build_bilstm_model(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=16, return_sequences=True, input_shape=(None, 768 + 128))))
        # model.add(TimeDistributed(Dense(units=8, activation="relu")))
        model.add(TimeDistributed(Dense(units=3, activation="softmax")))
        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy", accuracy_of_B_and_I], run_eagerly=True)
        return model
    
    
    def get_cross_validation_splits(self, k_fold: int = 5) -> List[List[List[int]]]:
        length = len(self.doc_words_tags)
        slices = []
        for i in range(1, k_fold + 1):
            test_slice_start = (i - 1) * (length // k_fold)
            test_slic_end = i * (length // k_fold)
            slices.append([list(range(test_slice_start, test_slic_end)), list(range(0, test_slice_start)) + list(range(test_slic_end, length))])
        return slices
    
    
    def train_model(self, k_fold: int = 5):
        inputs = []
        outputs = []
        for i in range(len(self.doc_words_tags)):
            input_v, output_v = self.build_input_and_output_from_word_tag_tuples(self.texts[i], self.doc_words_tags[i])
            inputs.append(input_v)
            outputs.append(output_v)
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, padding="post", value=[-10] * 896)
        outputs = tf.keras.preprocessing.sequence.pad_sequences(outputs, padding="post", value=0)
        inputs = np.asarray(inputs).astype(np.float32)
        outputs = np.asarray(outputs).astype(np.int32)
        cross_validation_slices = self.get_cross_validation_splits(k_fold)
        best_scores = []
        for slice in cross_validation_slices:
            # model = self.build_bilstm_model()
            model = self.build_crf_bilstm_model()
            test_indices = slice[0]
            train_indices = slice[1]
            train_input = inputs[train_indices]
            train_output = outputs[train_indices]
            test_input = inputs[test_indices]
            test_output = outputs[test_indices]
            f1callback = F1Callback(test_input, test_output)
            history = model.fit(train_input, train_output, epochs=42, validation_data=(test_input, test_output), batch_size=32, verbose=1, callbacks=[f1callback])
            # best_scores.append(max(history.history["val_accuracy_of_B_and_I"]))
        print(np.mean(best_scores))


def accuracy_of_B_and_I(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    predicted = np.argmax(y_pred.numpy(), axis=2)
    y_true = y_true.numpy()
    batches = y_true.shape[0]
    len_all_values = y_true.shape[1]
    count_hit = 0
    count_total = 0
    for batch in range(batches):
        for i in range(len_all_values):
            if y_true[batch][i] != 0:
                if y_true[batch][i] == predicted[batch][i]:
                    count_hit += 1    
                count_total += 1
        
    return count_hit / count_total
        

class F1Callback(Callback):
    
    def __init__(self, val_inputs, val_outputs):
        super().__init__()
        self.val_inputs = val_inputs
        self.val_outputs = val_outputs
    
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        
    def on_epoch_end(self, epoch, logs={}):
        val_targ = self.val_outputs
        val_predict = self.model.predict(self.val_inputs)
        _val_f1 = self.f1_score_of_full_keywords(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        print(" — val_f1: %f" % (_val_f1))
        return
    
    def find_start_end_of_keywords(self, y) -> Set[Tuple[int, int]]:
        index = 0
        start_end_tuples = set()
        start = -1
        end = -1
        while index < len(y):
            if y[index] == 1:
                start = index
                index += 1
                while index < len(y) and y[index] == 2:
                    index += 1
                end = index
                start_end_tuples.add((start, end))
            elif y[index] == 2:
                start = index
                index += 1
                while index < len(y) and y[index] == 2:
                    index += 1
                end = index
                start_end_tuples.add((start, end))
            else:
                index += 1
        return start_end_tuples


    def f1_score_of_full_keywords(self, y_true, y_pred) -> float:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        true_start_end = self.find_start_end_of_keywords(y_true)
        pred_start_end = self.find_start_end_of_keywords(y_pred)
        no_relevant = len(true_start_end)
        no_retrieved = len(pred_start_end)
        count_found = 0
        for start_end in pred_start_end:
            if start_end in true_start_end:
                count_found += 1
        
        if no_retrieved * no_relevant == 0:
            return float('nan')
        precision = count_found / no_retrieved
        recall = count_found / no_relevant
        f1 = 2 * precision * recall / (precision + recall)
        return f1
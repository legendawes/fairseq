
from joblib import Parallel, delayed
import json
import nltk
import numpy as np
import os
import pymorphy2
import torch
import torch.nn as nn

from classifiers.Nets import DenseNet


class MulticlassClassifier:
    def __init__(self, model_path="../models/multiclass", device="cpu"):
        self.__device = device

        # load vocab
        with open(os.path.join(model_path, 'vocab.json')) as f:
            self.__vocab = json.load(f)
        self.__token2id = {tok: i for i, tok in enumerate(self.__vocab)}

        # load model
        with open(os.path.join(model_path, 'config.json')) as f:
            model_config = json.load(f)
        self.__sent_len = model_config['sent_len']
        self.__pad_token = model_config['pad']
        self.__pad_token_id = self.__token2id[self.__pad_token]
        self.__unk_token = model_config['unk']
        self.__unk_token_id = self.__token2id[self.__unk_token]
        self.__classes = model_config['classes']
        self.__class2id = {tok: i for i, tok in enumerate(self.__classes)}
        model_state = torch.load(os.path.join(model_path, 'model.pt'), map_location=self.__device)
        embedding = torch.load(os.path.join(model_path, 'embeddings.pt'), map_location=self.__device)
        self.__model = DenseNet(
            emb_size=model_config['emb_size'],
            pretrained_emb=embedding,
            hidden_size=model_config['hidden'],
            out_size=len(self.__classes),
            dropout=model_config['dropout']
        )
        self.__model.load_state_dict(model_state)

        # for lemmatization
        self.__morph = pymorphy2.MorphAnalyzer()

    def __lemmatize_sentence(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        lemmatized_sentences = []
        for tok in tokens:
            p = self.__morph.parse(tok)[0]
            lemmatized_sentences.append(p.normal_form)
        return lemmatized_sentences

    def __lemmatize(self, sentences, n_jobs=1):
        return Parallel(n_jobs=n_jobs)(delayed(self.__lemmatize_sentence)(x) for x in sentences)

    def __pad_sentence(self, lemmas):
        len_diff = self.__sent_len-len(lemmas)
        if len_diff <= 0:
            return lemmas[:self.__sent_len]
        else:
            return lemmas + [self.__pad]*len_diff

    def __pad(self, sentences):
        return [self.__pad_sentence(sent) for sent in sentences]

    def __encode(self, lemmatized_sentences):
        encoded_sentences = []
        for sentence in lemmatized_sentences:
            encoded_sentences.append([self.__token2id.get(lemma, self.__pad_token_id) for lemma in sentence])
        return encoded_sentences

    @staticmethod
    def __iterate_minibatches(sentences, batch_size=64):
        for start in range(0, len(sentences), batch_size):
            yield sentences[start: start + batch_size]

    def __predict(self, sentences, batch_size):
        final_predictions = []
        for batch in self.__iterate_minibatches(sentences, batch_size):
            if self.__device=="gpu":
                batch_tensor = torch.tensor(batch, dtype=torch.int64).cuda()
            else:
                batch_tensor = torch.tensor(batch, dtype=torch.int64)
            prediction = self.__model(batch_tensor)
            predictions = nn.Sigmoid()(prediction)
            final_predictions.append(predictions.detach().numpy())
        return np.concatenate(final_predictions)

    def predict(self, sentences, classifier_names, batch_size=64, preprocessing_n_jobs=1):
        lemmatized_sentences = self.__lemmatize(sentences, preprocessing_n_jobs)
        lemmatized_sentences = self.__pad(lemmatized_sentences)
        encoded_sentences = self.__encode(lemmatized_sentences)
        prediction = self.__predict(encoded_sentences, batch_size)
        return prediction[:, [self.__class2id[c] for c in classifier_names]]


if __name__ == '__main__':
    list_of_sentences = [
        "Жили-были дед да баба.",
        "Как сообщает ТАСС, сейчас в мире 1% насления болеет коронавирусом.",
        "Наташа Ростова долго танцевала на балу.",
        "Сегодня я ела пельмени."
    ]

    clf = MulticlassClassifier()
    predictions = clf.predict(list_of_sentences, ['fairytale', 'twitter', 'news', 'tolstoy'])
    print(predictions)

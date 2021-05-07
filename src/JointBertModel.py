import json
import logging
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt
import pickle
from itertools import chain
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, TimeDistributed, Dropout
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from plot_keras_history import plot_history
from tqdm import tqdm
from src.preprocessing import Preprocessor
from transformers import TFMobileBertModel, TFPreTrainedModel, TFDistilBertModel
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


class JointBertModel(tf.keras.Model):
    def __init__(self, slots_num, intents_num, model_hub_path, model_name, dropout_prob=0.1):
        super().__init__(name='joint_intent_slot')
        # self.slots_num = slots_num
        # self.intents_num = intents_num
        # self.model_hub_path = model_hub_path
        # self.model_name = model_name
        self.dropout = Dropout(dropout_prob)
        self.bert = hub.KerasLayer(model_hub_path, trainable=True)
        # self.bert = TFPreTrainedModel.from_pretrained('files/mobilebert_multi_cased_L-24_H-128_B-512_A-4_F-4_OPT_1')
        self.intent_classifier = Dense(intents_num, activation='softmax', name='intent_classifier')
        self.slot_tagger = Dense(slots_num, activation='softmax', name='slot_tagger')
        # self.build_model()
        self.compile_model()

    def call(self, inputs, **kwargs):
        # two outputs from BERT
        trained_bert = self.bert(inputs, **kwargs)
        pooled_output = trained_bert["pooled_output"]
        sequence_output = trained_bert["sequence_output"]

        # sequence_output will be used for slot_filling / classification
        sequence_output = self.dropout(sequence_output, training=kwargs.get("training", False))
        slot_logits = self.slot_tagger(sequence_output)

        # pooled_output for intent classification
        pooled_output = self.dropout(pooled_output, training=kwargs.get("training", False))
        intent_logits = self.intent_classifier(pooled_output)

        return {'slot_tagger': slot_logits, 'intent_classifier': intent_logits}

    def compile_model(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08)
        # losses = [tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #           tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)]
        losses = {
            'slot_tagger': 'sparse_categorical_crossentropy',
            'intent_classifier': 'sparse_categorical_crossentropy',
        }
        # loss_weights = {'slot_tagger': 3.0, 'intent_classifier': 1.0}
        # metrics = {'intent_classifier': 'acc'}
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
        # self.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
        self.compile(optimizer=optimizer, loss=losses,  # loss_weights=loss_weights,
                     metrics=metrics)
        # self.model.summary()
        # self.summary()

    # def fit(self, X, Y, validation_data=None, epochs=5, batch_size=32):
    #     # history = self.model.fit(X, Y, validation_data=validation_data, epochs=epochs, batch_size=batch_size)
    #     history = self.fit(X, Y, validation_data=validation_data, epochs=epochs, batch_size=batch_size)
    #     return history

    def prepare_valid_positions(self, in_valid_positions):
        in_valid_positions = np.expand_dims(in_valid_positions, axis=2)
        in_valid_positions = np.tile(in_valid_positions, (1, 1, self.slots_num))
        return in_valid_positions

    def predict_slots_intent(self, x, slots_vectorizer, intent_vectorizer, remove_start_end=True,
                             include_intent_prob=False):
        valid_positions = x[3]
        x = (x[0], x[1], x[2], self.prepare_valid_positions(valid_positions))
        y_slots, y_intent = self.model.predict(x)
        slots = slots_vectorizer.inverse_transform(y_slots, valid_positions)
        if remove_start_end:
            slots = [x[1:-1] for x in slots]
        if not include_intent_prob:
            intents = np.array([intent_vectorizer.inverse_transform([np.argmax(i)])[0] for i in y_intent])
        else:
            intents = np.array(
                [(intent_vectorizer.inverse_transform([np.argmax(i)])[0], round(float(np.max(i)), 4))
                 for i in y_intent])
        return slots, intents

    def save_model(self, model_path):
        model_params = {
            'slots_num': self.slots_num,
            'intents_num': self.intents_num,
            'model_hub_path': self.model_hub_path,
            'model_name': self.model_name
        }
        with open(os.path.join(model_path, 'params.json'), 'w') as json_file:
            json.dump(model_params, json_file)
        self.model.save(os.path.join(model_path, 'joint_bert_model.h5'))

    @staticmethod
    def load_model(load_folder_path):
        with open(os.path.join(load_folder_path, 'params.json'), 'r') as json_file:
            model_params = json.load(json_file)
        model = JointBertModel(**model_params)
        model.model.load_weights(os.path.join(load_folder_path, 'joint_bert_model.h5'))
        return model

    @staticmethod
    def read_goo(dataset_folder_path):
        with open(os.path.join(dataset_folder_path, 'label'), encoding='utf-8') as f:
            labels = f.read().splitlines()
        with open(os.path.join(dataset_folder_path, 'seq.in'), encoding='utf-8') as f:
            text_arr = f.read().splitlines()
        with open(os.path.join(dataset_folder_path, 'seq.out'), encoding='utf-8') as f:
            tags_arr = f.read().splitlines()
        assert len(text_arr) == len(tags_arr) == len(labels)
        return text_arr, tags_arr, labels

    @staticmethod
    def train_model(train_config_path, model_name):
        logging.basicConfig(stream=sys.stdout, format='%(message)s', level=logging.WARNING)
        with open(os.path.join(train_config_path, 'train_config.json'), 'r') as json_file:
            train_config = json.load(json_file)
        data_folder_path = os.path.join(train_config['data_folder_path'], 'train')
        save_folder_path = train_config['save_folder_path']
        epochs = train_config['epochs']
        batch_size = train_config['batch_size']
        if model_name == 'BERT':
            model_hub_path = train_config['bert_hub_path']
        elif model_name == 'ALBERT':
            model_hub_path = train_config['albert_hub_path']
        else:
            model_hub_path = train_config['mobilebert_hub_path']

        logging.log(logging.WARNING, 'Reading data ...')
        text_arr, slots_arr, intents = JointBertModel.read_goo(data_folder_path)
        data = {
            'texts': text_arr,
            'intents': intents,
            'slots': [slots.split() for slots in slots_arr]
        }

        preprocessed_data = Preprocessor(model_name='distilbert-base-multilingual-cased').run_preprocessor(data)

        logging.log(logging.WARNING, 'Loading model ...')
        model = JointBertModel(slots_num=len(preprocessed_data.slot_map),
                               intents_num=len(preprocessed_data.intent_map),
                               model_hub_path=model_hub_path,
                               model_name=model_name)
        joint_model = JointIntentAndSlotFillingModel(model_name='distilbert-base-multilingual-cased',
                                               intent_num_labels=len(preprocessed_data.intent_map),
                                               slot_num_labels=len(preprocessed_data.slot_map),
                                               dropout_prob=0.1)

        logging.log(logging.WARNING, 'Training model ...')
        # X = {
        #     # "input_word_ids": preprocessed_data.encoded_texts["input_ids"],
        #     "input_word_ids": preprocessed_data.encoded_texts["input_word_ids"],
        #     # "input_type_ids": preprocessed_data.encoded_texts["token_type_ids"],
        #     "input_type_ids": preprocessed_data.encoded_texts["input_type_ids"],
        #     # "input_mask": preprocessed_data.encoded_texts["attention_mask"]
        #     "input_mask": preprocessed_data.encoded_texts["input_mask"]
        # }
        X = {"input_ids": preprocessed_data.encoded_texts["input_ids"],
             "attention_mask": preprocessed_data.encoded_texts["attention_mask"]}
        Y = (preprocessed_data.encoded_slots, preprocessed_data.encoded_intents)
        history = {}

        model, history = compile_train_model(model=joint_model, x=X, y=Y,
                                             epochs=2, batch_size=32)

        # X_train, X_val = {key: val[:400] for key, val in X.items()}, {key: val[400:] for key, val in X.items()}
        # Y_train, Y_val = (Y[0][:400], Y[1][:400]), (Y[0][400:], Y[1][400:])
        # hist = model.model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size)
        # hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size)
        # hist = model.fit(X, Y, epochs=epochs, batch_size=batch_size)
        # if history:
        #     history = {key: history[key] + hist.history[key] for key in hist.history}
        # else:
        #     history = hist.history
        plot_history(history)
        plt.show()
        plt.close()

        logging.log(logging.WARNING, 'Saving model ...')
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
            logging.info('Folder `%s` created' % save_folder_path)
        # model.save_model(save_folder_path)
        model.save(save_folder_path)
        # with open(os.path.join(save_folder_path, 'tags_vectorizer.pkl'), 'wb') as handle:
        #     pickle.dump(tags_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(os.path.join(save_folder_path, 'intents_label_encoder.pkl'), 'wb') as handle:
        #     pickle.dump(intents_label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return model

    def evaluate_model(self, train_config_path, model_name):
        def flatten(y):
            return list(chain.from_iterable(y))

        with open(os.path.join(train_config_path, 'train_config.json'), 'r') as json_file:
            train_config = json.load(json_file)
        if model_name == 'BERT':
            model_hub_path = train_config['bert_hub_path']
        elif model_name == 'ALBERT':
            model_hub_path = train_config['albert_hub_path']
        else:
            model_hub_path = train_config['mobilebert_hub_path']
        load_folder_path = train_config['save_folder_path']
        test_folder_path = os.path.join(train_config['data_folder_path'], 'test')

        logging.log(logging.WARNING, 'Loading model ...')
        if not os.path.exists(load_folder_path):
            logging.warning('Folder `%s` not exist' % load_folder_path)
            return

        with open(os.path.join(load_folder_path, 'tags_vectorizer.pkl'), 'rb') as handle:
            tags_vectorizer = pickle.load(handle)
        with open(os.path.join(load_folder_path, 'intents_label_encoder.pkl'), 'rb') as handle:
            intents_label_encoder = pickle.load(handle)

        text_arr, tags_arr, intents = self.read_goo(test_folder_path)
        input_ids, input_mask, segment_ids, valid_positions, sequence_lengths = \
            bert_vectorizer.transform(text_arr)

        predicted_tags, predicted_intents = self.predict_slots_intent(
            [input_ids, input_mask, segment_ids, valid_positions],
            tags_vectorizer, intents_label_encoder, remove_start_end=True)
        gold_tags = [x.split() for x in tags_arr]
        f1_score = metrics.f1_score(flatten(gold_tags), flatten(predicted_tags), average='micro')
        acc = metrics.accuracy_score(intents, predicted_intents)

        logging.log(logging.WARNING, metrics.classification_report(flatten(gold_tags),
                                                                   flatten(predicted_tags), digits=3))
        logging.log(logging.WARNING, 'Slot f1_score = %f' % f1_score)
        logging.log(logging.WARNING, 'Intent accuracy = %f' % acc)
        return f1_score, acc


class JointIntentAndSlotFillingModel(tf.keras.Model):
    def __init__(self, model_name, intent_num_labels, slot_num_labels, dropout_prob=0.1):
        super().__init__(name="joint_intent_slot_filling")
        self.bert = TFDistilBertModel.from_pretrained(model_name, output_hidden_states=True)
        self.dropout = Dropout(dropout_prob)
        self.intent_classifier = Dense(intent_num_labels, name="intent_classifier")
        self.slot_classifier = Dense(slot_num_labels, name="slot_classifier")

    def call(self, inputs, **kwargs):
        trained_bert = self.bert(inputs, **kwargs)
        # pooled_output = trained_bert.pooler_output
        # We only care about DistilBERT's output for the [CLS] token,
        # which is located at index 0 of every encoded sequence.
        # Splicing out the [CLS] tokens gives us 2D data.
        pooled_output = trained_bert.last_hidden_state[:, 0, :]
        sequence_output = trained_bert.last_hidden_state

        # sequence_output will be used for slot_filling / classification
        sequence_output = self.dropout(sequence_output,
                                       training=kwargs.get("training", False))
        slot_logits = self.slot_classifier(sequence_output)

        # pooled_output for intent classification
        pooled_output = self.dropout(pooled_output,
                                     training=kwargs.get("training", False))
        intent_logits = self.intent_classifier(pooled_output)

        return slot_logits, intent_logits


def compile_train_model(model, x, y,
                        epochs, batch_size,
                        learning_rate=3e-5, epsilon=1e-08):
    logging.info("compile_train_model :: running")

    # optimizer -> Adam
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)

    # two outputs, one for slots, another for intents
    losses = [SparseCategoricalCrossentropy(from_logits=True),
              SparseCategoricalCrossentropy(from_logits=True)]

    # metrics -> accuracy -> requirement for the project
    metrics = [SparseCategoricalAccuracy('accuracy')]

    # compile model
    model.compile(optimizer=opt, loss=losses, metrics=metrics)

    history = model.fit(x, y, epochs=epochs,
                        batch_size=batch_size, shuffle=True,
                        verbose=1)

    logging.info("compile_train_model :: complete")

    return model, history

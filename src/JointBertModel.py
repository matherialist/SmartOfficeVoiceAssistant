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
from tensorflow.keras.layers import Input, Dense, Multiply, TimeDistributed
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from plot_keras_history import plot_history
from tqdm import tqdm
from src.AlbertTokenization import FullTokenizer as AlbertFullTokenizer
from src.BertTokenization import FullTokenizer as BertFullTokenizer


class BERTVectorizer:
    def __init__(self, model_name, bert_model_hub_path):
        self.model_name = model_name
        self.bert_model_hub_path = bert_model_hub_path
        self.create_tokenizer_from_hub_module(model_name=model_name)

    def create_tokenizer_from_hub_module(self, model_name):
        """Get the vocab file and casing info from the Hub module."""
        keras_layer = hub.KerasLayer(self.bert_model_hub_path, trainable=False)
        if model_name == 'BERT':
            vocab_file = keras_layer.resolved_object.vocab_file.asset_path.numpy()
            do_lower_case = keras_layer.resolved_object.do_lower_case.numpy()
            self.tokenizer = BertFullTokenizer(vocab_file, do_lower_case)
            self.preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")
        elif model_name == 'ALBERT':
            sp_model_file = keras_layer.resolved_object.sp_model_file.asset_path.numpy()
            self.tokenizer = AlbertFullTokenizer(vocab_file=sp_model_file,
                                                 do_lower_case=True,
                                                 spm_model_file=sp_model_file)
        else:
            vocab_file = keras_layer.resolved_object.vocab_file.asset_path.numpy()
            do_lower_case = keras_layer.resolved_object.do_lower_case.numpy()
            self.tokenizer = BertFullTokenizer(vocab_file, do_lower_case)
            self.preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")

        del keras_layer

    def tokenize(self, text: str):
        words = text.split()  # whitespace tokenizer
        tokens = []
        valid_positions = []
        for i, word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions

    def transform(self, text_arr):
        res = self.preprocessor(text_arr)
        input_ids, input_mask, segment_ids = res['input_word_ids'], res['input_mask'], res['input_type_ids']
        valid_positions = []
        for text in tqdm(text_arr):
            _, valid_pos = self.tokenize(text)
            valid_pos.insert(0, 1)
            valid_pos.append(1)
            valid_positions.append(valid_pos)

        sequence_lengths = np.array([len(i) for i in input_ids])
        # if self.model_name != 'MobileBert':
        #     input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, padding='post')
        #     input_mask = tf.keras.preprocessing.sequence.pad_sequences(input_mask, padding='post')
        #     segment_ids = tf.keras.preprocessing.sequence.pad_sequences(segment_ids, padding='post')
        valid_positions = tf.keras.preprocessing.sequence.pad_sequences(valid_positions, padding='post', maxlen=42)
        print('input_ids: ', input_ids.shape)
        print('input_mask: ', input_mask.shape)
        print('segment_ids: ', segment_ids.shape)
        print('valid_positions: ', valid_positions.shape)
        return input_ids, input_mask, segment_ids, valid_positions, sequence_lengths


class TagsVectorizer:
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def tokenize(self, tags_str_arr):
        return [s.split() for s in tags_str_arr]

    def fit(self, tags_str_arr):
        data = ['<PAD>'] + [item for sublist in self.tokenize(tags_str_arr) for item in sublist]
        self.label_encoder.fit(data)

    def transform(self, tags_str_arr, valid_positions):
        seq_length = valid_positions.shape[1]
        data = self.tokenize(tags_str_arr)
        data = [self.label_encoder.transform(['O'] + x + ['O']).astype(np.int32) for x in data]
        output = np.zeros((len(data), seq_length))
        for i in tqdm(range(len(data))):
            idx = 0
            for j in range(seq_length):
                if valid_positions[i][j] == 1:
                    output[i][j] = data[i][idx]
                    idx += 1
        return output

    def inverse_transform(self, model_output_3d, valid_positions):
        seq_length = valid_positions.shape[1]
        slots = np.argmax(model_output_3d, axis=-1)
        slots = [self.label_encoder.inverse_transform(y) for y in slots]
        output = []
        for i in range(len(slots)):
            y = []
            for j in range(seq_length):
                if valid_positions[i][j] == 1:
                    y.append(str(slots[i][j]))
            output.append(y)
        return output


class JointBertModel:
    def __init__(self, slots_num, intents_num, bert_hub_path, model_name):
        self.slots_num = slots_num
        self.intents_num = intents_num
        self.bert_hub_path = bert_hub_path
        self.model_name = model_name
        self.build_model()
        self.compile_model()

    def build_model(self):
        in_valid_positions = Input(shape=(None, self.slots_num), dtype=tf.float32, name='valid_positions')
        text_input = Input(shape=(), dtype=tf.string)
        encoder_inputs = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")(text_input)
        outputs = hub.KerasLayer(self.bert_hub_path, trainable=True)(encoder_inputs)
        pooled_output = outputs["pooled_output"]  # [batch_size, 512].
        sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 512].

        intents_fc = Dense(self.intents_num, activation='softmax', name='intent_classifier')(pooled_output)
        slots_output = TimeDistributed(Dense(self.slots_num, activation='softmax'))(sequence_output)
        slots_output = Multiply(name='slots_tagger')([slots_output, in_valid_positions])

        self.model = Model(inputs=[text_input, in_valid_positions], outputs=[slots_output, intents_fc])

    def compile_model(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        losses = {
            'slots_tagger': 'sparse_categorical_crossentropy',
            'intent_classifier': 'sparse_categorical_crossentropy',
        }
        loss_weights = {'slots_tagger': 3.0, 'intent_classifier': 1.0}
        metrics = {'intent_classifier': 'acc'}
        self.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
        self.model.summary()

    def fit(self, X, Y, validation_data=None, epochs=5, batch_size=32):
        # X["valid_positions"] = self.prepare_valid_positions(X["valid_positions"])
        # if validation_data is not None:
        #     X_val, Y_val = validation_data
        #     X_val["valid_positions"] = self.prepare_valid_positions(X_val["valid_positions"])
        #     validation_data = (X_val, Y_val)
        history = self.model.fit(X, Y, validation_data=validation_data,
                                 epochs=epochs, batch_size=batch_size)
        return history

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
            'bert_hub_path': self.bert_hub_path,
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
            labels = f.readlines()
        with open(os.path.join(dataset_folder_path, 'seq.in'), encoding='utf-8') as f:
            text_arr = f.readlines()
        with open(os.path.join(dataset_folder_path, 'seq.out'), encoding='utf-8') as f:
            tags_arr = f.readlines()
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
        text_arr, tags_arr, intents = JointBertModel.read_goo(data_folder_path)

        logging.log(logging.WARNING, 'Vectorizing data ...')
        bert_vectorizer = BERTVectorizer(model_name, model_hub_path)
        # input_ids, input_mask, segment_ids, valid_positions, sequence_lengths = bert_vectorizer.transform(text_arr)
        _, _, _, valid_positions, sequence_lengths = bert_vectorizer.transform(text_arr)

        logging.log(logging.WARNING, 'Vectorizing tags ...')
        tags_vectorizer = TagsVectorizer()
        tags_vectorizer.fit(tags_arr)
        tags = tags_vectorizer.transform(tags_arr, valid_positions)
        slots_num = len(tags_vectorizer.label_encoder.classes_)

        logging.log(logging.WARNING, 'Encoding labels ...')
        intents_label_encoder = LabelEncoder()
        intents = intents_label_encoder.fit_transform(intents).astype(np.int32)
        intents_num = len(intents_label_encoder.classes_)

        logging.log(logging.WARNING, 'Loading model ...')
        model = JointBertModel(slots_num, intents_num, model_hub_path, model_name=model_name)

        logging.log(logging.WARNING, 'Training model ...')
        X = (np.array(text_arr), valid_positions)
        Y = (tags, intents)

        history = {}
        print(X)
        print(X[0].shape)
        print(X[1].shape)
        print(Y[0].shape)
        print(Y[1].shape)

        X_train, X_val = (X[0][:400], X[1][:400]), (X[0][400:], X[1][400:])
        Y_train, Y_val = (Y[0][:400], Y[1][:400]), (Y[0][400:], Y[1][400:])
        hist = model.model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size)
        if history:
            history = {key: history[key] + hist.history[key] for key in hist.history}
        else:
            history = hist.history
        plot_history(history)
        plt.show()
        plt.close()

        logging.log(logging.WARNING, 'Saving model ...')
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
            logging.info('Folder `%s` created' % save_folder_path)
        model.save_model(save_folder_path)
        with open(os.path.join(save_folder_path, 'tags_vectorizer.pkl'), 'wb') as handle:
            pickle.dump(tags_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(save_folder_path, 'intents_label_encoder.pkl'), 'wb') as handle:
            pickle.dump(intents_label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
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

        bert_vectorizer = BERTVectorizer(model_name, model_hub_path)

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

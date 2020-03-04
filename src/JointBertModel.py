"""
Original implementation: https://github.com/MahmoudWahdan/dialog-nlu
"""

import json
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, TimeDistributed
from sklearn.preprocessing import LabelEncoder
from src.AlbertTokenization import FullTokenizer


class BERTVectorizer:

    def __init__(self, sess, bert_model_hub_path="https://tfhub.dev/google/albert_base/1"):
        self.sess = sess
        self.bert_model_hub_path = bert_model_hub_path
        self.create_tokenizer_from_hub_module()

    def create_tokenizer_from_hub_module(self):
        """Get the vocab file and casing info from the Hub module."""
        bert_module = hub.Module(self.bert_model_hub_path)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        vocab_file, do_lower_case = self.sess.run(
            [
                tokenization_info["vocab_file"],
                tokenization_info["do_lower_case"],
            ]
        )
        self.tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case, spm_model_file=vocab_file)

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
        input_ids = []
        input_mask = []
        segment_ids = []
        valid_positions = []
        for text in text_arr:
            ids, mask, seg_ids, valid_pos = self.__vectorize(text)
            input_ids.append(ids)
            input_mask.append(mask)
            segment_ids.append(seg_ids)
            valid_positions.append(valid_pos)

        sequence_lengths = np.array([len(i) for i in input_ids])
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, padding='post')
        input_mask = tf.keras.preprocessing.sequence.pad_sequences(input_mask, padding='post')
        segment_ids = tf.keras.preprocessing.sequence.pad_sequences(segment_ids, padding='post')
        valid_positions = tf.keras.preprocessing.sequence.pad_sequences(valid_positions, padding='post')
        return input_ids, input_mask, segment_ids, valid_positions, sequence_lengths

    def __vectorize(self, text: str):
        tokens, valid_positions = self.tokenize(text)
        tokens.insert(0, '[CLS]')
        valid_positions.insert(0, 1)
        tokens.append('[SEP]')
        valid_positions.append(1)

        segment_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        return input_ids, input_mask, segment_ids, valid_positions


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
        for i in range(len(data)):
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


class AlbertLayer(tf.keras.layers.Layer):

    def __init__(self, fine_tune=True, pooling='first',
                 albert_path="https://tfhub.dev/google/albert_base/1", **kwargs, ):
        self.fine_tune = fine_tune
        self.output_size = 768
        self.pooling = pooling
        self.albert_path = albert_path
        if self.pooling not in ['first', 'mean']:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        super(AlbertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.albert = hub.Module(self.albert_path, trainable=self.fine_tune, name=f"{self.name}_module")

        if self.fine_tune:
            # Remove unused layers
            trainable_vars = self.albert.variables
            if self.pooling == 'first':
                trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
                trainable_layers = ["pooler/dense"]

            elif self.pooling == 'mean':
                trainable_vars = [var for var in trainable_vars
                                  if not "/cls/" in var.name and not "/pooler/" in var.name
                                  ]
                trainable_layers = []
            else:
                raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

            # Select how many layers to fine tune
            trainable_layers.append("encoder/transformer/group_0")

            # Update trainable vars to contain only the specified layers
            trainable_vars = [var for var in trainable_vars if any([l in var.name for l in trainable_layers])]

            # Add to trainable weights
            for var in trainable_vars:
                self._trainable_weights.append(var)

            for var in self.albert.variables:
                if var not in self._trainable_weights:
                    self._non_trainable_weights.append(var)

        super(AlbertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [tf.keras.backend.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids, valid_positions = inputs
        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        result = self.albert(inputs=bert_inputs, signature='tokens', as_dict=True)
        return result['pooled_output'], result['sequence_output']

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_size)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'fine_tune': self.fine_tune,
            # 'trainable': self.trainable,
            'output_size': self.output_size,
            'pooling': self.pooling,
            'albert_path': self.albert_path,
        })
        return config


class JointBertModel:

    def __init__(self, slots_num, intents_num, bert_hub_path, sess, num_bert_fine_tune_layers=1):
        self.slots_num = slots_num
        self.intents_num = intents_num
        self.bert_hub_path = bert_hub_path
        self.num_bert_fine_tune_layers = num_bert_fine_tune_layers

        self.model_params = {
            'slots_num': slots_num,
            'intents_num': intents_num,
            'bert_hub_path': bert_hub_path,
            'num_bert_fine_tune_layers': num_bert_fine_tune_layers,
        }

        self.build_model()
        self.compile_model()
        self.initialize_vars(sess)

    def compile_model(self):
        optimizer = tf.keras.optimizers.Adam(lr=5e-5)

        losses = {
            'slots_tagger': 'sparse_categorical_crossentropy',
            'intent_classifier': 'sparse_categorical_crossentropy',
        }
        loss_weights = {'slots_tagger': 3.0, 'intent_classifier': 1.0}
        metrics = {'intent_classifier': 'acc'}
        self.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
        self.model.summary()

    def build_model(self):
        in_id = Input(shape=(None,), name='input_ids')
        in_mask = Input(shape=(None,), name='input_masks')
        in_segment = Input(shape=(None,), name='segment_ids')
        in_valid_positions = Input(shape=(None, self.slots_num), name='valid_positions')
        bert_inputs = [in_id, in_mask, in_segment, in_valid_positions]

        bert_pooled_output, bert_sequence_output = AlbertLayer(
            fine_tune=True if self.num_bert_fine_tune_layers > 0 else False,
            albert_path=self.bert_hub_path,
            pooling='mean', name='AlbertLayer')(bert_inputs)

        intents_fc = Dense(self.intents_num, activation='softmax', name='intent_classifier')(bert_pooled_output)

        slots_output = TimeDistributed(Dense(self.slots_num, activation='softmax'))(bert_sequence_output)
        slots_output = Multiply(name='slots_tagger')([slots_output, in_valid_positions])

        self.model = Model(inputs=bert_inputs, outputs=[slots_output, intents_fc])

    def fit(self, X, Y, validation_data=None, epochs=5, batch_size=32):
        history = self.model.fit(X, Y, validation_data=validation_data, epochs=epochs, batch_size=batch_size)
        return history

    def prepare_valid_positions(self, in_valid_positions):
        in_valid_positions = np.expand_dims(in_valid_positions, axis=2)
        in_valid_positions = np.tile(in_valid_positions, (1, 1, self.slots_num))
        return in_valid_positions

    def initialize_vars(self, sess):
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())
        tf.compat.v1.keras.backend.set_session(sess)

    def predict(self, x):
        return self.model.predict(x)

    def predict_slots_intent(self, x, slots_vectorizer, intent_vectorizer, remove_start_end=True,
                             include_intent_prob=False):
        valid_positions = x[3]
        x = (x[0], x[1], x[2], self.prepare_valid_positions(valid_positions))
        y_slots, y_intent = self.predict(x)
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

    def save(self, model_path):
        with open(os.path.join(model_path, 'params.json'), 'w') as json_file:
            json.dump(self.model_params, json_file)
        self.model.save(os.path.join(model_path, 'joint_bert_model.h5'))

    @staticmethod
    def load(load_folder_path, sess):
        with open(os.path.join(load_folder_path, 'params.json'), 'r') as json_file:
            model_params = json.load(json_file)

        slots_num = model_params['slots_num']
        intents_num = model_params['intents_num']
        bert_hub_path = model_params['bert_hub_path']
        num_bert_fine_tune_layers = model_params['num_bert_fine_tune_layers']

        new_model = JointBertModel(slots_num, intents_num, bert_hub_path, sess, num_bert_fine_tune_layers)
        new_model.model.load_weights(os.path.join(load_folder_path, 'joint_bert_model.h5'))
        return new_model

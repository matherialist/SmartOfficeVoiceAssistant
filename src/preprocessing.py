import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import logging
from transformers import AutoTokenizer


class Preprocessor:
    """
    Preprocesses data for training
    data : list of data read from the training file
    model_name : pre-trained bert model to load
    """
    def __init__(self, model_name='bert-base-multilingual-cased'):
        self.all_texts = None
        self.all_intents = None
        self.all_slots = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")

        # computed properties
        self.intent_names = None
        self.slot_names = None

        self.encoded_texts = None
        self.encoded_intents = None
        self.encoded_slots = None

        # index -> name mapping
        self.intent_map = None
        self.slot_map = None

    def encode_texts(self):
        """
        Encodes texts to tensorflow tensors using AutoTokenizer,
        each instance will be a dictionary containing tf tensors
            input_ids
            token_type_ids
            attention_masks
        """
        # self.encoded_texts = self.preprocessor(self.all_texts)
        self.encoded_texts = self.tokenizer(self.all_texts,
                                            padding=True,
                                            truncation=True,
                                            return_tensors="tf")

    # index -> intent mapping
    def create_intent_map(self):
        # find unique intents or intent_names
        self.intent_names = list(set(self.all_intents))

        # create a index -> intent mapping
        self.intent_map = dict()
        for idx, intent in enumerate(self.intent_names):
            self.intent_map[intent] = idx

    # creates a tf tensor containing encoded intents
    def encode_intents(self):
        encoded = []
        for intent in self.all_intents:
            encoded.append(self.intent_map[intent])
        # convert to tf tensor
        self.encoded_intents = tf.convert_to_tensor(encoded, dtype="int32")

    def create_slot_map(self, ):
        # unique slots
        self.slot_names = list()
        for slots in self.all_slots:
            for slot in slots:
                if slot != 'O':
                    slot = slot[2:]
                if slot not in self.slot_names:
                    self.slot_names.append(slot)

        # the tokenizer adds special chars for padding add a special slot_name, <PAD> for them
        self.slot_names.insert(0, "<PAD>")

        # mapping
        self.slot_map = dict()
        for idx, slot_name in enumerate(self.slot_names):
            self.slot_map[slot_name] = idx

    def encode_slots(self):
        # find the max encoded test length
        # tokenizer pads all texts to same length anyway so
        # just get the length of the first one's input_ids
        max_len = len(self.encoded_texts["input_ids"][0])
        # max_len = len(self.encoded_texts["input_word_ids"][0])

        self.encoded_slots = np.zeros(shape=(len(self.all_texts), max_len), dtype=np.int32)

        for idx, text in enumerate(self.all_texts):
            enc = []  # for this idx, to be added at the end to encoded_slots

            # slot names for this idx
            slot_names = self.all_slots[idx]

            # raw word tokens
            # not using bert for this block because bert uses a wordpiece tokenizer which will make
            # the slot label to word mapping difficult
            raw_tokens = text.split()

            # words or slot_values associated with a certain slot_name are contained in the values of the
            # dict slots_names now this becomes a two way lookup
            # first we check if a word belongs to any slot label or not and then we add the value from
            # slot map to encoded for that word
            for i, rt in enumerate(raw_tokens):
                # use bert tokenizer to get wordpiece tokens
                bert_tokens = self.tokenizer.tokenize(rt)

                # find the slot name for a token
                rt_slot_name = slot_names[i]
                if rt_slot_name is not None:
                    # fill with the slot_map value for all bert tokens for rt
                    if rt_slot_name != 'O':
                        rt_slot_name = rt_slot_name[2:]
                    enc.append(self.slot_map[rt_slot_name])
                    enc.extend([self.slot_map[rt_slot_name]] * (len(bert_tokens) - 1))
                else:
                    # rt is not associated with any slot name add a 0
                    enc.append(0)

            # now add to encoded_slots
            # ignore the first and the last elements in encoded text as they're special chars
            self.encoded_slots[idx, 1:len(enc) + 1] = enc

    # call this method to do all the work
    def run_preprocessor(self, data):
        self.all_texts = data['texts']
        self.all_intents = data['intents']
        self.all_slots = data['slots']
        logging.warning("preprocessing :: running")
        logging.info("encode_texts")
        self.encode_texts()
        self.create_intent_map()
        logging.info("encode_intents")
        self.encode_intents()
        self.create_slot_map()
        logging.info("encode_slots")
        self.encode_slots()
        logging.warning("preprocessing :: complete")
        return self

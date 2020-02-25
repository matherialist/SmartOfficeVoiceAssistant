import os
import pickle
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from src.JointBertModel import JointBertModel, BERTVectorizer, TagsVectorizer


class ActionClassifier:

    def __init__(self, load_folder_path):
        self.sess = sess = tf.compat.v1.Session()
        self.bert_vectorizer = BERTVectorizer(sess)
        self.tags_vectorizer = TagsVectorizer()
        self.intents_label_encoder = LabelEncoder()
        with open(os.path.join(load_folder_path, 'tags_vectorizer.pkl'), 'rb') as handle:
            self.tags_vectorizer = pickle.load(handle)
        with open(os.path.join(load_folder_path, 'intents_label_encoder.pkl'), 'rb') as handle:
            self.intents_label_encoder = pickle.load(handle)
        self.model = JointBertModel.load(load_folder_path, sess)

    def run(self):
        text = None
        while text != "q":
            text = input("New message: ")
            if text != "q":
                result = self.__predict(text)
                print('Intent: ', result['intent']['name'])
                print('Confidence: ', result['intent']['confidence'])
                if len(result['slots']) > 0:
                    print('Slots:')
                    for slot in result['slots']:
                        print(' ', slot['slot'], '=', slot['value'])
                else:
                    print('Slots:\n', ' None')

    def __predict(self, utterance):
        tokens = utterance.split()
        input_ids, input_mask, segment_ids, valid_positions, data_sequence_lengths = \
            self.bert_vectorizer.transform([utterance])
        predicted_tags, predicted_intents = self.model.predict_slots_intent(
            [input_ids, input_mask, segment_ids, valid_positions],
            self.tags_vectorizer, self.intents_label_encoder, remove_start_end=True,
            include_intent_prob=True)
        slots = self.__fill_slots(predicted_tags[0])
        slots = [{"slot": slot, "start": start, "end": end, "value": ' '.join(tokens[start:end + 1])}
                 for slot, start, end in slots]
        predictions = {
            "intent": {
                "name": predicted_intents[0][0].strip(),
                "confidence": predicted_intents[0][1]
            },
            "slots": slots
        }
        return predictions

    def __fill_slots(self, slots_arr, no_class_tag='O', begin_prefix='B-', in_prefix='I-'):
        previous = None
        slots = []
        start = -1
        end = -1
        for i, slot in enumerate(slots_arr):
            if slot == no_class_tag:
                current = None
                if previous is not None:
                    slots.append(self.__form_slot(previous, start, end))
            if slot.startswith(begin_prefix):
                current = slot[len(begin_prefix):]
                if previous is not None:
                    slots.append(self.__form_slot(previous, start, end))
                start = i
            elif slot.startswith(in_prefix):
                current = slot[len(in_prefix):]
                if current != previous:
                    if previous is not None:
                        slots.append(self.__form_slot(previous, start, end))
                    for j, sl in enumerate(slots):
                        if current == sl[0]:
                            sl = list(sl)
                            sl[2] = i
                            slots[j] = tuple(sl)
                    current = None
                else:
                    end = i
            previous = current
        if previous is not None:
            slots.append(self.__form_slot(previous, start, end))
        return slots

    def __form_slot(self, name, start, end):
        if end < start:
            end = start
        return (name, start, end)

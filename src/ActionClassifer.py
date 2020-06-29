import os
import pickle
import random
from sklearn.preprocessing import LabelEncoder
from src.JointBertModel import JointBertModel, BERTVectorizer, TagsVectorizer


class ActionClassifier:
    def __init__(self, load_folder_path, model_hub_path, is_bert):
        self.bert_vectorizer = BERTVectorizer(is_bert=is_bert, bert_model_hub_path=model_hub_path)
        self.tags_vectorizer = TagsVectorizer()
        self.intents_label_encoder = LabelEncoder()
        with open(os.path.join(load_folder_path, 'tags_vectorizer.pkl'), 'rb') as handle:
            self.tags_vectorizer = pickle.load(handle)
        with open(os.path.join(load_folder_path, 'intents_label_encoder.pkl'), 'rb') as handle:
            self.intents_label_encoder = pickle.load(handle)
        self.model = JointBertModel.load_model(load_folder_path)

    def make_prediction(self, utterances):
        intent_slots = self._predict(utterances[0])
        intent_slots2 = self._predict(utterances[1])
        lang = 'en'
        if intent_slots['intent']['confidence'] < intent_slots2['intent']['confidence']\
                and intent_slots2['intent']['name'] != 'no_intent' or intent_slots['intent']['name'] == 'no_intent':
            intent_slots = intent_slots2
            lang = 'ru'
        if intent_slots['intent']['name'] == 'no_intent':
            command = None
        else:
            command = self._get_command(intent_slots)
        if lang == 'en':
            response = utterances[0]
        else:
            response = utterances[1]
        return command, response

    def get_phrase(self, command):
        if command is None:
            return "i don't understand you"
        device, action, parameter = command['device'], command['action'], command['parameter']
        if action in ['switch_on', 'switch_off']:
            action = action.replace('_', ' ')
        if device == 'air':
            phrase = 'the air '
            if parameter == 'temperature':
                phrase += 'temperature is {} degrees'.format(command['value'][0])
            elif parameter == 'humidity':
                phrase += 'humidity is {} percent'.format(command['value'][1])
            elif parameter == 'CO2':
                phrase += 'CO2 concentration is {} percent'.format(command['value'][2])
            elif parameter == 'all':
                phrase += 'temperature is {} degrees, humidity is {} percent, CO2 concentration is {} percent'\
                    .format(command['value'][0], command['value'][1], command['value'][2])

        elif device == 'timer':
            phrase = 'okay, i will {} the {}'.format(action, device)
        elif action in ['switch on', 'switch off', 'open', 'close', 'mute', 'unmute']:
            phrase = 'okay, i will {} the {}'.format(action, device)
        else:
            phrase = 'okay, i will {} the {} {}'.format(action, device, parameter)
        return phrase

    def _predict(self, utterance):
        tokens = utterance.split()
        input_ids, input_mask, segment_ids, valid_positions, data_sequence_lengths = \
            self.bert_vectorizer.transform([utterance])
        predicted_tags, predicted_intents = self.model.predict_slots_intent(
            [input_ids, input_mask, segment_ids, valid_positions],
            self.tags_vectorizer, self.intents_label_encoder, remove_start_end=True,
            include_intent_prob=True)
        slots = self._fill_slots(predicted_tags[0])
        slots = [{'name': name, 'value': ' '.join([tokens[i] for i in slots[name]])} for name in slots.keys()]
        predictions = {
            'intent': {
                'name': predicted_intents[0][0].strip(),
                'confidence': predicted_intents[0][1]
            },
            'slots': slots
        }
        return predictions

    def _fill_slots(self, slots_arr, no_class_tag='O', begin_prefix='B-', in_prefix='I-'):
        slots = {}
        for i, slot in enumerate(slots_arr):
            if slot == no_class_tag:
                continue
            if slot.startswith(begin_prefix):
                name = slot[len(begin_prefix):]
                slots[name] = [i]
            elif slot.startswith(in_prefix):
                name = slot[len(in_prefix):]
                if name in slots.keys():
                    slots[name].append(i)
                else:
                    slots[name] = [i]
        return slots

    def _get_command(self, intent_slots):
        command = {'device': None, 'action': None, 'parameter': None, 'value': None}
        intent = intent_slots['intent']['name']
        slots = intent_slots['slots']
        command['device'] = intent
        # light
        if intent == 'light':
            for slot in slots:
                if slot['name'] in ['action.switch_on', 'action.switch_off', 'action.set']:
                    command['action'] = slot['name'].split('.')[1]
                elif slot['name'] in ['brightness.increase', 'brightness.decrease']:
                    command['action'] = slot['name'].split('.')[1]
                    command['parameter'] = slot['name'].split('.')[0]
                elif slot['name'] == 'brightness.value':
                    command['parameter'] = slot['name'].split('.')[0]
                    command['value'] = slot['value'].replace('%', '')
                elif slot['name'] == 'color':
                    command['parameter'] = 'color'
                    command['value'] = slot['value']
            if command['action'] in ['increase', 'decrease'] and command['value'] is None:
                command['value'] = 'default'

        # implicit_light
        if intent == 'implicit_light':
            command['device'] = 'light'
            for slot in slots:
                if slot['name'] in ['increase', 'decrease']:
                    command['action'] = slot['name']
                    command['parameter'] = 'brightness'
                    command['value'] = 'default'

        # conditioner
        if intent == 'conditioner':
            for slot in slots:
                if slot['name'] in ['action.switch_on', 'action.switch_off']:
                    command['action'] = slot['name'].split('.')[1]
                elif slot['name'] == 'action.set':
                    command['action'] = 'set'
                    command['parameter'] = 'temperature'
                elif slot['name'] in ['action.increase_temp', 'action.decrease_temp']:
                    command['action'] = slot['name'].split('.')[1].split('_')[0]
                    command['parameter'] = 'temperature'
                elif slot['name'] == 'temperature':
                    command['parameter'] = 'temperature'
                    command['value'] = slot['value']
            if command['action'] in ['increase', 'decrease'] and command['value'] is None:
                command['value'] = 'default'

        # implicit_conditioner
        if intent == 'implicit_conditioner':
            command['device'] = 'conditioner'
            for slot in slots:
                if slot['name'] in ['increase', 'decrease']:
                    command['action'] = slot['name']
                    command['parameter'] = 'temperature'
                    command['value'] = 'default'

        # curtains
        if intent == 'curtains':
            for slot in slots:
                if slot['name'] in ['action.open', 'action.close']:
                    command['action'] = slot['name'].split('.')[1]

        # tv
        if intent == 'tv':
            for slot in slots:
                if slot['name'] in ['action.switch_on', 'action.switch_off', 'action.set']:
                    command['action'] = slot['name'].split('.')[1]
                elif slot['name'] in ['action.mute', 'action.unmute']:
                    command['action'] = slot['name'].split('.')[1]
                elif slot['name'] in ['sound.increase', 'sound.decrease']:
                    command['action'] = slot['name'].split('.')[1]
                    command['parameter'] = slot['name'].split('.')[0]
                elif slot['name'] == 'sound.value':
                    command['parameter'] = 'sound'
                    command['value'] = slot['value'].replace('%', '')
            if command['action'] in ['increase', 'decrease'] and command['value'] is None:
                command['value'] = 'default'

        # air
        if intent == 'air':
            for slot in slots:
                if slot['name'] in ['parameter.temperature', 'parameter.humidity', 'parameter.CO2', 'parameter.all']:
                    command['action'] = 'get_info'
                    command['parameter'] = slot['name'].split('.')[1]

        # timer
        if intent == 'timer':
            time = {}
            for slot in slots:
                if slot['name'] == 'action.set':
                    command['action'] = 'set'
                elif slot['name'] == 'minutes':
                    time['minutes'] = slot['value']
                elif slot['name'] == 'seconds':
                    time['seconds'] = slot['value']
            if time:
                command['value'] = time
            else:
                command = None

        # reminder
        if intent == 'reminder':
            time = {}
            for slot in slots:
                if slot['name'] == 'action.set':
                    command['action'] = 'set'
                    command['parameter'] = 'time'
                elif slot['name'] == 'day':
                    time['day'] = slot['value']
                elif slot['name'] == 'hours':
                    time['hours'] = slot['value']
                elif slot['name'] == 'minutes':
                    time['minutes'] = slot['value']
                elif slot['name'] == 'seconds':
                    time['seconds'] = slot['value']
                elif slot['name'] == 'message':
                    command['value']['message'] = slot['value']
            if time:
                command['value']['time'] = time
            else:
                command = None

        # Change string numbers to integer
        str_numbers = {'half': 50, 'third': 33, 'quarter': 25, 'fourth': 25, 'fifth': 20,
                       'one tenth': 10, 'one second': 50, 'one third': 33, 'one fourth': 25,
                       'one fifth': 20, 'two thirds': 66, 'two fifths': 40, 'three quarters': 75,
                       'three fifths': 60, 'four fifths': 80}

        if command['value'] and command['parameter'] != 'color' and command['device'] != 'timer'\
                and command['device'] != 'air':
            if command['value'] in str_numbers.keys():
                command['value'] = str_numbers[command['value']]
            # elif command['value'] not in ['maximum', 'minimum']:
            #     command['value'] = 'default'

        if command['action'] is None:
            command = None

        return command

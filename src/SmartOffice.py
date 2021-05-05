import random
from src.ActionClassifer import ActionClassifier
from src.VoiceAssistant import VoiceAssistant


class WebSOVA:
    def __init__(self):
        self.light = {'active': False, 'color': 'white', 'brightness': 50}
        self.conditioner = {'active': False, 'temperature': 20}
        self.tv = {'active': False, 'muted': False, 'volume': 50}
        self.curtains = {'open': False}
        self.air = {'temperature': random.randint(10, 25), 'humidity': random.randint(0, 100),
                    'CO2': random.randint(10, 25)}

    def set_parameters(self, command):
        if command['device'] == 'light':
            if command['action'] == 'switch_on':
                self.light['active'] = True
                self.light['brightness'] = 50
            elif command['action'] == 'switch_off':
                self.light['active'] = False
                self.light['brightness'] = 0
            elif command['action'] == 'set':
                if command['parameter'] == 'brightness':
                    if command['value'] == 'minimum':
                        self.light['brightness'] = 0
                    elif command['value'] == 'maximum':
                        self.light['brightness'] = 100
                    elif 0 <= int(command['value']) <= 100:
                        self.light['brightness'] = command['value']
                elif command['parameter'] == 'color':
                    self.light['color'] = command['value']
            elif command['action'] == 'increase':
                if command['value'] == 'default':
                    add = 10
                else:
                    add = int(command['value'])
                new_val = self.light['brightness'] + add
                if 0 <= new_val <= 100:
                    self.light['brightness'] = new_val
            elif command['action'] == 'decrease':
                if command['value'] == 'default':
                    add = 10
                else:
                    add = int(command['value'])
                new_val = self.light['brightness'] - add
                if 0 <= new_val <= 100:
                    self.light['brightness'] = new_val
            return {'device': command['device'], 'parameters': self.light}

        elif command['device'] == 'conditioner':
            if command['action'] == 'switch_on':
                self.conditioner['active'] = True
            elif command['action'] == 'switch_off':
                self.conditioner['active'] = False
            elif command['action'] == 'set':
                if command['parameter'] == 'temperature':
                    if command['value'] == 'minimum':
                        self.conditioner['temperature'] = 15
                    elif command['value'] == 'maximum':
                        self.conditioner['temperature'] = 30
                    elif 15 <= int(command['value']) <= 30:
                        self.conditioner['temperature'] = command['value']
            elif command['action'] == 'increase':
                if command['value'] == 'default':
                    add = 2
                else:
                    add = int(command['value'])
                new_val = self.conditioner['temperature'] + add
                if 15 <= new_val <= 30:
                    self.conditioner['temperature'] = new_val
            elif command['action'] == 'decrease':
                if command['value'] == 'default':
                    add = 2
                else:
                    add = int(command['value'])
                new_val = self.conditioner['temperature'] - add
                if 15 <= new_val <= 30:
                    self.conditioner['temperature'] = new_val
            return {'device': command['device'], 'parameters': self.conditioner}

        elif command['device'] == 'tv':
            if command['action'] == 'switch_on':
                self.tv['active'] = True
            elif command['action'] == 'switch_off':
                self.tv['active'] = False
            if command['action'] == 'mute':
                self.tv['muted'] = True
            elif command['action'] == 'unmute':
                self.tv['muted'] = False
            elif command['action'] == 'set':
                if command['parameter'] == 'sound':
                    if command['value'] == 'minimum':
                        self.tv['volume'] = 0
                    elif command['value'] == 'maximum':
                        self.tv['volume'] = 100
                    elif 0 <= int(command['value']) <= 100:
                        self.tv['volume'] = command['value']
            elif command['action'] == 'increase':
                if command['value'] == 'default':
                    add = 10
                else:
                    add = int(command['value'])
                new_val = self.tv['volume'] + add
                if 0 <= new_val <= 100:
                    self.tv['volume'] = new_val
            elif command['action'] == 'decrease':
                if command['value'] == 'default':
                    add = 10
                else:
                    add = int(command['value'])
                new_val = self.tv['volume'] - add
                if 0 <= new_val <= 100:
                    self.tv['volume'] = new_val
            return {'device': command['device'], 'parameters': self.tv}

        elif command['device'] == 'curtains':
            if command['action'] == 'open':
                self.curtains['open'] = True
            elif command['action'] == 'close':
                self.curtains['open'] = False
            return {'device': command['device'], 'parameters': self.curtains}

        elif command['device'] == 'air':
            if command['parameter'] == 'temperature':
                self.air['temperature'] = random.randint(10, 25)
            elif command['parameter'] == 'humidity':
                self.air['humidity'] = random.randint(0, 100)
            elif command['parameter'] == 'CO2':
                self.air['CO2'] = random.randint(10, 25)
            elif command['parameter'] == 'all':
                self.air['temperature'] = random.randint(10, 25)
                self.air['humidity'] = random.randint(0, 100)
                self.air['CO2'] = random.randint(10, 25)
            return {'device': command['device'], 'parameters': self.air}


class SmartOffice:
    def __init__(self, load_folder_path):
        self.action_classifier = ActionClassifier(
            load_folder_path,
            model_hub_path="https://tfhub.dev/tensorflow/mobilebert_multi_cased_L-24_H-128_B-512_A-4_F-4_OPT/1",
            model_name='MobileBert')
        self.voice_assistant = VoiceAssistant()
        self.sova = WebSOVA()

    def run(self):
        text_en, text_ru = self.voice_assistant.recognise_audio()
        command, response = self.action_classifier.make_prediction([text_en, text_ru])
        print(command)
        command = self.sova.set_parameters(command)
        return {"command": command, "response": response}

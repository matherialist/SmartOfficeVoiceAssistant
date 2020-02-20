class ActionClassifer:
    def __init__(self, configPath):
        pass

    def run(self):
        myModel = Model()
        text = None
        while text != "q":
            text = input("New message: ")
            if text != "q":
                result = myModel.classify(text)
                print(result)
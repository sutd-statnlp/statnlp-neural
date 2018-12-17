from Instance import Instance

class LinearInstance(Instance):

    def __init__(self, instance_id, weight, input = None, output = None):
        '''

        :param instance_id:
        :param weight: 1.0
        :param input: a list of input list [[], []]
        :param output: a list of label []
        '''
        super.__init__(LinearInstance, instance_id, weight = 1.0, input = None, output = None)


    def size(self):
        return len(input)

    def duplicate(self):
        dup = LinearInstance(self.instance_id, self.weight, self.input, self.output)
        return dup

    def removeOutput(self):
        self.output = None

    def removePrediction(self):
        self.prediction = None

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output

    def get_prediction(self):
        return self.prediction

    def set_prediction(self, prediction):
        self.prediction = prediction

    def has_output(self):
        return self.output != None

    def has_prediction(self):
        return self.prediction != None



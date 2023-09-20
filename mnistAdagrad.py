import numpy as np

train_file = open("mnist_train.csv", 'r')
test_file = open("mnist_test.csv", 'r')
train_list = train_file.readlines()
test_list = test_file.readlines()

train_file.close()
test_file.close()


class ReseauNeurone:
    def __init__(self, sizes=[784, 200, 10], epochs=10, lr=0.1):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr
        self.eps = 1e-8


        input_layer = sizes[0]
        hidden_1 = sizes[1]
        output_layer = sizes[2]

        self.v = {
            "W1": np.zeros((200, 784)),
            "W2": np.zeros((10, 200))
        }
        self.params = {
            "W1": np.random.randn(hidden_1, input_layer) * np.sqrt(1 / hidden_1),
            "W2": np.random.randn(output_layer, hidden_1) * np.sqrt(1 / output_layer),
        }

    def sigmoid(self, x, derivation=False):
        # the flag is used in backwards
        if derivation:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        else:
            return 1 / (np.exp(-x) + 1)

    def softmax(self, x, derivation=False):
        exps = np.exp(x - x.max())
        if derivation:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def forward_pass(self, x_train):
        # using a copy is to keep list of weights for backpropagation
        params = self.params
        params["A0"] = x_train
        # from input_layer to hidden_1
        params["Z1"] = np.dot(params["W1"], params["A0"])
        params["A1"] = self.sigmoid(params["Z1"])
        # from hidden_1 to hidden_
        params["Z2"] = np.dot(params["W2"], params["A1"])
        params["A2"] = self.softmax(params["Z2"])

        return params['Z2']

    def backward_pass(self, y_train, output):
        params = self.params
        change_w = {}
        # calculate w3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params["Z2"])
        change_w['W2'] = np.outer(error, params["A1"])
        # calculate w2 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params["Z1"], derivation=True)
        change_w['W1'] = np.outer(error, params["A0"])

        self.v["W2"] = self.v["W2"] + (change_w["W2"]**2)
        self.v["W1"] = self.v["W1"] + (change_w["W1"]**2)

        return self.v, change_w

    def update_weights(self, v: dict, change_w):
        for key, val in v.items():
            self.params[key] -= (self.lr/np.sqrt(val+self.eps)) * change_w[key]

    def compute_accuracy(self, test_data):
        predictions = []
        for x in test_data:
            values = x.split(',')
            inputs = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(10) + 0.01
            targets[int(values[0])] = 0.99
            output = self.forward_pass(inputs)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(targets))
        return np.mean(predictions)

    def train(self, train_list, test_list):

        for i in range(self.epochs):
            for x in train_list:
                values = x.split(',')

                inputs = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01

                targets = np.zeros(10) + 0.01
                targets[int(values[0])] = 0.99
                output = self.forward_pass(inputs)
                v, change_w = self.backward_pass(targets, output)
                self.update_weights(v, change_w)
            accuracy = self.compute_accuracy(test_list)
            print("epoch:", i + 1, "accuracy", accuracy, "")


reseau = ReseauNeurone(sizes=[784, 200, 10], epochs=10, lr=0.1)
reseau.train(train_list, test_list)

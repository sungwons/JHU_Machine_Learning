class Winnow:

    def __init__(self, alpha, theta, num_of_weights, weight_initial_val):
        self.alpha = alpha
        self.theta = theta
        self.weights = []
        for idx in range(num_of_weights):
            self.weights.append(float(weight_initial_val))

    def learn(self, data_instance, label):
        prediction = self.predict(data_instance)
        if prediction == 1 and label == 0:
            self.demote_weights(data_instance)
        elif prediction == 0 and label == 1:
            self.promote_weights(data_instance)

    def promote_weights(self, data_instance):
        for idx, val in enumerate(data_instance):
            if val == 1:
                self.weights[idx] = self.weights[idx] * self.alpha

    def demote_weights(self, data_instance):
        for idx, val in enumerate(data_instance):
            if val == 1:
                self.weights[idx] = self.weights[idx] / self.alpha

    def predict(self, data_instance):
        weight_instance_sum = 0
        for idx, val in enumerate(self.weights):
            weight_instance_sum += self.weights[idx] * data_instance[idx]
        prediction = 1 if weight_instance_sum > self.theta else 0
        return prediction

    def print_model(self):
        model = "Alpha: %f, theta: %f\n" % (self.alpha, self.theta)
        for weight_index in xrange(len(self.weights)):
            model += "Weight_{} = {}".format(weight_index, self.weights[weight_index])
            if weight_index != (len(self.weights) - 1):
                model += ", "
        return model

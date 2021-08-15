import numpy as np

def linear(x, derivative=False):
    return np.ones(len(x)) if derivative else x

def square(x, derivative=False):
    return 2*x if derivative else np.square(x)

def relu(x, derivative=False):
    if derivative:
        return (x>0)*1.0
    else:
        return (x>0)*x

def prod(a):
    for i in range(len(a)):
        if i == 0:
            r = a[i]
        else:
            if (type(a[i]) == np.ndarray) & (type(r) == np.ndarray):
                r = r.dot(a[i])
            else:
                r = r*a[i]
    return r

class NeuralNetworkModel:
    def __init__(self,feature_size):
        self.L = 0
        
        self.n_nodes = [feature_size]
        self.weights = []
        self.biases = []
        self.activations = []
        
    def AddLayer(self,n_node,activation):
        
        self.L += 1
        self.n_nodes.append(n_node)
        self.weights.append(np.random.normal(size = [self.n_nodes[-2],self.n_nodes[-1]]))
        self.biases.append(np.random.normal(size = [1,self.n_nodes[-1]]))
        self.activations.append(activation)
        
    def Predict(self,X):
        outputs = []
        for i in range(X.shape[0]):
            output = X[i]
            for l in range(self.L):
                output = output.dot(self.weights[l])+self.biases[l]
                output = self.activations[l](output)
            outputs.append(output)
        outputs = np.array(outputs)
        return outputs
    
    def Train(self,X,y,n_epochs=100,lr=.0001):
        for i in range(n_epochs):
            self.Backward(X,y,lr)

    def Backward(self,X,y,lr):
        for i in range(X.shape[0]):
            node_outputs = [X[i]]
            node_outputs_activated = [X[i]]
            for l in range(self.L):
                node_outputs.append(node_outputs_activated[-1].dot(self.weights[l])+self.biases[l].flatten())
                node_outputs_activated.append(self.activations[l](node_outputs[-1]))
                
            e_derivative = 2*(node_outputs[-1]-y[i])
            function_derivatives = []
            
            for l in range(self.L):
                function_derivative = self.activations[l](node_outputs[l].dot(self.weights[l]),derivative=True)
                function_derivatives.append(function_derivative)
            
            all_gradients = []
            all_gradients_bias = []
            for l in range(self.L):
                layer_gradients = []
                layer_gradients_bias = []
                for a in range(node_outputs[l].shape[0]):
                    gradients = [(e_derivative*function_derivatives[-1])[0]]
                    gradients_bias = [(e_derivative*function_derivatives[-1])[0]]
                    for l2 in range(l,self.L):
                        if l == l2:
                            gradients.append(node_outputs_activated[l2][a])
                        else:
                            gradients.append(self.weights[l2])
                            gradients_bias.append(self.weights[l2])
                    layer_gradients.append(prod(gradients).T)
                    layer_gradients_bias.append(prod(gradients_bias).T)
                    
                all_gradients.append(layer_gradients)
                all_gradients_bias.append(layer_gradients_bias)
            
            for l in range(len(self.weights)):
                self.weights[l] -= lr*all_gradients[l][0]
                self.biases[l] -= lr*all_gradients_bias[l][0]

    
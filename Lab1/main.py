import numpy as np
import warnings
import matplotlib.pyplot as plt 
def generate_linear(n = 100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs, labels = [], []

    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n, 1)
def generate_XOR_easy(n = 11):
    inputs, labels = [], []

    for i in range(n):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


class layers:
    def __init__(self,input_dim,output_dim, act_fcn_type="sigmoid"):
        self.weight_matrix = np.random.randn(input_dim,output_dim)
        self.act_fcn_type = act_fcn_type
    def forward_propagate(self,input):
        self.stored_input = input
        if(self.act_fcn_type == "sigmoid"):
            self.output = self.sigmoid(input.dot(self.weight_matrix))
        elif(self.act_fcn_type == 'relu'):
            self.output = self.relu(input.dot(self.weight_matrix))
        return self.output 
  
    def backward_propagate(self,loss):
        #the gradient shape is (n , layer dim)(layer dim ,layer dim)
        if(self.act_fcn_type == 'sigmoid'):
            self.gradient_matrix = loss * self.derivative_sigmoid(self.output)
        elif(self.act_fcn_type == 'relu'):
            self.gradient_matrix = loss * self.derivative_relu(self.output)
        #print(f"The gradient's shape is{self.gradient_matrix.shape} and the weight's shape is {self.weight_matrix.shape}")
        #print(self.gradient_matrix)
        return self.gradient_matrix.dot(self.weight_matrix.T)

    def update(self,learning_rate):
        # print("before update")
        # print(self.weight_matrix)
        self.weight_matrix -= learning_rate * self.stored_input.T.dot(self.gradient_matrix)
        # print("after update")
        # print(self.weight_matrix)
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def derivative_sigmoid(self,y):
        # First derivative of Sigmoid function.
        return y * (1 - y)
    
    def relu(self,x):
        return np.maximum(0,x)
    
    def derivative_relu(self,y):
        return np.where(y >= 0,1,0)


class NN:
    #Y = weight*X + b
    def __init__(self,input_dim, hidden_layer_dim, output_dim, activation_type, learning_rate,optimizer = "gd"):
        """         
        param input_dim : number of train data
        param hidden_layer_dim : number of hidden_layer units 
        """
        """ 
        input->layer1:[hidden_layer1_dim,input_dim][input_dim,n].T->[4,100]
        layer1->layer2:[hidden_layer_dim2,hidden_layer1_dim][hidden_layer1_dim,n]
        layer2->output_layer:[output_dim,hidden_layer2_dim][hidden_layer_dim2,n] 
        """
        self.input_layer = layers(input_dim,hidden_layer_dim,activation_type)
        self.hidden_layer1 = layers(hidden_layer_dim,hidden_layer_dim,activation_type)
        self.output_layer = layers(hidden_layer_dim,output_dim,activation_type)
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.number_of_data = 100
        self.epoch ,self.loss = [],[]

    def forward_propagate(self , input):
        """ ai : a presents the layer i
            zi : z presents the activation of ai """
        z1 = self.input_layer.forward_propagate(input)
        z2 = self.hidden_layer1.forward_propagate(z1)
        out = self.output_layer.forward_propagate(z2)
        return out
    
    def backward_propagate(self,loss):
        """ loss function : (y-ground truth)**2 """
        dz3 = self.output_layer.backward_propagate(loss)
        dz2 = self.hidden_layer1.backward_propagate(dz3)
        dz1 = self.input_layer.backward_propagate(dz2)
    def update(self):
        learning_rate = self.learning_rate
        self.input_layer.update(learning_rate)
        self.hidden_layer1.update(learning_rate)
        self.output_layer.update(learning_rate)

    def train(self,epoch,inputs,labels):
   
        for i in range(epoch):
            predictY = self.forward_propagate(inputs)
            loss = np.mean((predictY-labels)**2)
            self.backward_propagate(2*(predictY-labels)/self.number_of_data)
            self.update()
            acc = np.mean(labels == np.where(predictY > 0.5, 1, 0))
           
            if(i%500 == 0):
                self.epoch.append(i)
                self.loss.append(loss)
                print("epoch {:5d} loss:{:.6f} acc:{:.6f}".format(i,loss,acc))

    def predict(self,inputs):
        return np.where(self.forward_propagate(inputs) > 0.5, 1, 0)
    def show_learing_curve(self):
        
        plt.plot(self.epoch, self.loss, label='Training Loss')
        plt.title('Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.show()

    def show_result(self,x, y, pred_y):
        
        plt.subplot(1,2,1)
        plt.title('Grounnd truth',fontsize=18)
        for i in range(x.shape[0]):
            if y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')
        plt.subplot(1,2,2)
        plt.title('Predict result', fontsize=18)
        for i in range(x.shape[0]):
            if pred_y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')
        plt.show()




if __name__ == "__main__":
    inputs , labels = generate_linear()
    model = NN(2,10,1,"sigmid",0.6)
    # model.input_layer.property()
    model.train(100000,inputs,labels)
    predict_y = model.predict(inputs)
    model.show_result(inputs,labels,predict_y)
    model.show_learing_curve()
   






    


   


        













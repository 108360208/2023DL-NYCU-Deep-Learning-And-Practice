import numpy as np
import matplotlib.pyplot as plt 
import time
#data loader 
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
    def __init__(self,input_dim,output_dim, act_fcn_type="sigmoid",optimizer='gd'):
        self.weight_matrix = np.random.randn(input_dim,output_dim)
        self.act_fcn_type = act_fcn_type
        self.optimizer = optimizer
    def forward_propagate(self,input,batch_mode="none"):
        self.stored_input = input
        if(self.act_fcn_type == "sigmoid"): 
            self.output = self.sigmoid(input.dot(self.weight_matrix))
        elif(self.act_fcn_type == 'relu'):
            self.derivate_relu_input = input.dot(self.weight_matrix)
            self.output = self.relu(input.dot(self.weight_matrix)) 
        elif(self.act_fcn_type == 'leeky_relu'):
            self.derivate_relu_input = input.dot(self.weight_matrix)
            self.output = self.leaky_relu(input.dot(self.weight_matrix))
        elif(self.act_fcn_type == "tanh"):
            self.derivate_tanh_input = input.dot(self.weight_matrix)
            self.output = self.tanh(input.dot(self.weight_matrix))
        elif(self.act_fcn_type == 'none'):
            self.output = input.dot(self.weight_matrix)
        return self.output 
  
    def backward_propagate(self,loss):
        #the gradient shape is (n , layer dim)(layer dim ,layer dim)
        if(self.act_fcn_type == 'sigmoid'):
            self.gradient_matrix = loss * self.derivative_sigmoid(self.output)
        elif(self.act_fcn_type == 'relu'):
            self.gradient_matrix = loss * self.derivative_relu(self.derivate_relu_input)
        elif(self.act_fcn_type == 'leeky_relu'):
            self.gradient_matrix = loss * self.derivative_leaky_relu(self.derivate_relu_input)
        elif(self.act_fcn_type == 'tanh'):
            self.gradient_matrix = loss * self.derivative_tanh(self.derivate_tanh_input)
        elif(self.act_fcn_type == 'none'):
            self.gradient_matrix = loss
        #print(f"The gradient's shape is{self.gradient_matrix.shape} and the weight's shape is {self.weight_matrix.shape}")
        #print(self.gradient_matrix)
        return self.gradient_matrix.dot(self.weight_matrix.T)

    def update(self,learning_rate):
        self.weight_matrix -= learning_rate * self.stored_input.T.dot(self.gradient_matrix) 

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def derivative_sigmoid(self,y):
        # First derivative of Sigmoid function.
        return y * (1 - y)
    
    def relu(self,x):
        return np.maximum(0,x)
    
    def derivative_relu(self,y):
        return np.where(y > 0,1,0)
    
    def leaky_relu(self,x, alpha=0.001):
        return np.maximum(alpha * x, x)

    def derivative_leaky_relu(self,x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    def tanh(self,x):
        return np.tanh(x)

    def derivative_tanh(self,x):
        return 1 - np.tanh(x)**2

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
        self.neural_network =[]
        self.neural_network.append(layers(input_dim,hidden_layer_dim,activation_type,optimizer))
        self.neural_network.append(layers(hidden_layer_dim,hidden_layer_dim,activation_type,optimizer))
        self.neural_network.append(layers(hidden_layer_dim,output_dim,"sigmoid",optimizer))
        # if(activation_type == "none"):
        #     self.output_layer = layers(hidden_layer_dim,output_dim,"sigmoid")
        # else:
        #     self.output_layer = layers(hidden_layer_dim,output_dim,activation_type)
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.epoch ,self.loss = [],[]

    def forward_propagate(self , input):
        """ ai : a presents the layer i
            zi : z presents the activation of ai """
        for layer in self.neural_network:
            input = layer.forward_propagate(input)
        return input
    
    def backward_propagate(self,loss):
        """ loss function : (y-ground truth)**2 """
        for layer in self.neural_network[::-1]:
            loss = layer.backward_propagate(loss)
 
    def update(self):
        learning_rate = self.learning_rate
        for layer in self.neural_network:
            layer.update(learning_rate)
 
    def train(self,epoch,inputs,labels):
        if(self.optimizer == 'msgd'):   
            batch_size = int(len(inputs)/5) 
            for i in range(epoch+1):   
                # random samples
                total_loss = 0
                rand_i = [np.random.randint(len(inputs)) for j in range(len(inputs))]  
                inputs = inputs[rand_i]
                labels = labels[rand_i]
                for j in range(0,len(inputs),batch_size):         
                    predict_each_batch = self.forward_propagate(inputs[j:j+batch_size])            
                    self.backward_propagate(2*(predict_each_batch-labels[j:j+batch_size])/batch_size)          
                    self.update()  
                    total_loss += np.mean((predict_each_batch - labels[j:j+batch_size])**2)
                if(i%1 == 0):
                    predictY = self.forward_propagate(inputs)  
                    loss = (total_loss)/5       
                    self.epoch.append(i)
                    self.loss.append(loss)
                print("epoch {:5d} loss:{:.6f}".format(i,loss))

        elif(self.optimizer == 'sgd'):    
            for i in range(epoch+1):
                # random samples
                total_loss = 0
                rand_i = [np.random.randint(len(inputs)) for j in range(len(inputs))]  
                inputs = inputs[rand_i]
                labels = labels[rand_i]        
                for j in range(0,len(inputs)): 
                    predict_each_batch = self.forward_propagate(inputs[list([j])])            
                    self.backward_propagate(2*(predict_each_batch-labels[j]))          
                    self.update()  
                    total_loss += (predict_each_batch - labels[j])**2
                if(i%1 == 0):
                    predictY = self.forward_propagate(inputs)  
                    loss = float(total_loss[0][0])/len(inputs)                  
                    self.epoch.append(i)
                    self.loss.append(loss)
                print("epoch {:5d} loss:{:6f} ".format(i,loss))
        else :    
            for i in range(epoch+1):
                predictY = self.forward_propagate(inputs)              
                self.backward_propagate(2*(predictY-labels)/len(inputs))
                self.update()            
                if(i%1 == 0):
                    loss = np.mean((predictY-labels)**2)
                    self.epoch.append(i)
                    self.loss.append(loss)
            #print("epoch {:5d} loss:{:.6f} acc:{:.6f}".format(i,loss,acc))
                print("epoch {:5d} loss : {:.10f}".format(i,loss))

    def predict(self,inputs):
        print(self.forward_propagate(inputs))
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

def show_diffent_learning_curve(epoch_list,loss_list,type_list):
    plt.figure()
    for epoch , loss ,type in zip(epoch_list,loss_list,type_list):
        plt.plot(epoch, loss ,label = type)
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
def show_acc_curve(acc_list, type_list ,demo_type):
    plt.figure()
    x = range(len(type_list))
    plt.bar(x, acc_list, tick_label=type_list)
    plt.title('Accuracy Curve')
    plt.xlabel(demo_type)
    plt.ylabel('Accuracy')
    plt.show()
def demo_diffent_case(demo_type,demo_diffent_type,data_type):
    if(data_type == "liner"):
        inputs , labels = generate_linear()
    else:
        inputs , labels = generate_XOR_easy()
    if(demo_type == "optimizer"):
        optimizer_list = demo_diffent_type
        models = []
        #increase different type optimzer model
        for diff_optimizer_model in optimizer_list:
            models.append(NN(2,4,1,"sigmoid",0.5,diff_optimizer_model))

    elif(demo_type == "activation"):
        activation_list = demo_diffent_type
        models = []
        for diff_activation in activation_list:
            models.append(NN(2,4,1,diff_activation,0.2,"gd"))
    
    elif(demo_type == "units"):
        units_list = demo_diffent_type
        models = []
        for diff_units in units_list:
            models.append(NN(2,diff_units,1,"sigmoid",0.5,"gd"))
    elif(demo_type == "learning_rate"):
        learing_rate_list = demo_diffent_type
        models = []
        for diff_learning in learing_rate_list:
            models.append(NN(2,4,1,"sigmoid",diff_learning,"gd"))
    #model train 
    epoch_list = []
    loss_list = []
    for model in models:
        model.train(5000,inputs,labels)
        epoch_list.append(model.epoch)
        loss_list.append(model.loss)
    #model predict 
    acc_list = []
    for model in models:
        predict_y = model.predict(inputs)
        acc = np.mean(labels == np.where(predict_y > 0.5, 1, 0))
        acc_list.append(acc)
      
    #print differnt curve
    show_diffent_learning_curve(epoch_list, loss_list,demo_diffent_type)
    show_acc_curve(acc_list,demo_diffent_type,demo_type)

if __name__ == "__main__":
    inputs , labels = generate_XOR_easy()
    #demo_diffent_case("learning_rate",[0.01,1,100],"liner")
    #demo_diffent_case("units",[2,8,32],"liner")
    #demo_diffent_case("optimizer",["gd","sgd","msgd"],"liner")
    #demo_diffent_case("activation",["sigmoid","tanh","relu","leeky_relu"],"liner")

    #model set 
    model = NN(2,4,1,"sigmoid",0.5,"gd")
    #model train 
    model.train(30000 ,inputs,labels)
    #print model's prediction 
    predict_y = model.predict(inputs)
    model.show_result(inputs,labels,predict_y)
    model.show_learing_curve()
    predict_y = model.forward_propagate(inputs)
    for i in range(0,len(inputs)):
        print("Iter{:2d}|   Ground truth {}|  prediction {:.6f}".format(i,labels[i][0],predict_y[i][0]))
    
    acc = np.mean(labels == np.where(predict_y > 0.5, 1, 0))
    print(f"Accuracy:{acc}")
    
   






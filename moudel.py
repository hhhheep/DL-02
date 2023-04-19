from abc import ABCMeta, abstractmethod
from nn import *
import pickle

class Net(metaclass=ABCMeta):

    # Neural network super class

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dout):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass




class LeNet5(Net):
    # LeNet5

    def __init__(self):
        self.conv1 = Conv2d(3, 6, 5)
        self.Sigm1 = Sigmoid()
        self.pool1 = MaxPool(2,2)
        self.conv2 = Conv2d(6, 16, 5)
        self.Sigm2 = Sigmoid()
        self.pool2 = MaxPool(2,2)
        self.FC1 = Linear_Layer(16*61*61, 120)
        self.ReLU3 = ReLU()
        self.FC2 = Linear_Layer(120, 84)
        self.Sigm4 = Sigmoid()
        self.FC3 = Linear_Layer(84, 50)
        self.Softmax = Softmax()

        self.p2_shape = None

    def forward(self, X,pred = False):
        h1 = self.conv1.forward(X,pred)
        a1 = self.Sigm1.forward(h1,pred)
        p1 = self.pool1.forward(a1,pred)
        h2 = self.conv2.forward(p1,pred)
        a2 = self.Sigm2.forward(h2,pred)
        p2 = self.pool2.forward(a2,pred)

        if pred:
            pass
        else:
            self.p2_shape = p2.shape

        fl = p2.reshape(X.shape[0],-1) # Flatten
        h3 = self.FC1.forward(fl,pred)
        a3 = self.ReLU3.forward(h3,pred)
        h4 = self.FC2.forward(a3,pred)
        a5 = self.Sigm4.forward(h4,pred)
        h5 = self.FC3.forward(a5,pred)
        a5 = self.Softmax.forward(h5,pred)
        return a5

    def backward(self, dout):
        #dout = self.Softmax.backward(dout)
        dout = self.FC3.backward(dout)
        dout = self.Sigm4.backward(dout)
        dout = self.FC2.backward(dout)
        dout = self.ReLU3.backward(dout)
        dout = self.FC1.backward(dout)
        dout = dout.reshape(self.p2_shape) # reshape
        dout = self.pool2.backward(dout)
        dout = self.Sigm2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = self.pool1.backward(dout)
        dout = self.Sigm1.backward(dout)
        dout = self.conv1.backward(dout)

    def get_params(self):
        return [self.conv1.w, self.conv1.b, self.conv2.w, self.conv2.b, self.FC1.w, self.FC1.b, self.FC2.w, self.FC2.b, self.FC3.w, self.FC3.b]

    def set_params(self, params):
        [self.conv1.w, self.conv1.b, self.conv2.w, self.conv2.b, self.FC1.w, self.FC1.b, self.FC2.w, self.FC2.b, self.FC3.w, self.FC3.b] = params

class naive_LeNet5(Net):
    # LeNet5

    def __init__(self):
        self.conv1 = Conv2d(3, 6, 3)
        self.conva = Conv2d(6, 6, 3)
        self.Sigm1 = X_sigmoid()
        self.pool1 = MaxPool(2, 2)
        self.conv2 = Conv2d(6, 16, 3)
        self.Sigm2 = X_sigmoid()
        self.pool2 = MaxPool(2, 2)
        self.FC1 = Linear_Layer(16 * 62 * 62, 120)
        self.ReLU3 = X_sigmoid()
        self.FC2 = Linear_Layer(120, 84)
        self.Sigm4 = X_sigmoid()
        self.FC3 = Linear_Layer(84, 50)
        self.Softmax = Softmax()

        self.p2_shape = None

    def forward(self, X, pred=False):
        # print("forward")
        h1 = self.conv1.forward(X, pred)
        # print("conv1:",h1.size())
        ha = self.conva.forward(h1,pred)
        # print("conva:",ha.size())
        a1 = self.Sigm1.forward(ha, pred)
        # print("Sigm1:",a1.size())
        p1 = self.pool1.forward(a1, pred)
        # print("pool1:",p1.size())
        h2 = self.conv2.forward(p1, pred)
        # print(h2.size())
        a2 = self.Sigm2.forward(h2, pred)
        # print(a2.size())
        p2 = self.pool2.forward(a2, pred)
        # print(p2.size())
        if pred:
            pass
        else:
            self.p2_shape = p2.shape

        fl = p2.reshape(X.shape[0], -1)  # Flatten
        # print(fl.size())
        h3 = self.FC1.forward(fl, pred)
        # print(h3.size())
        a3 = self.ReLU3.forward(h3, pred)
        # print(a3.size())
        h4 = self.FC2.forward(a3, pred)
        # print(h4.size())
        a5 = self.Sigm4.forward(h4, pred)
        # print(a5.size())
        h5 = self.FC3.forward(a5, pred)
        # print(h5.size())
        a5 = self.Softmax.forward(h5, pred)
        return a5

    def backward(self, dout):
        # print("backward")
        # dout = self.Softmax.backward(dout)
        # print(dout.size())
        dout = self.FC3.backward(dout)
        # print(dout.size())
        dout = self.Sigm4.backward(dout)
        # print(dout.size())
        dout = self.FC2.backward(dout)
        # print(dout.size())
        dout = self.ReLU3.backward(dout)
        # print(dout.size())
        dout = self.FC1.backward(dout)
        # print(dout.size())
        dout = dout.view(self.p2_shape)  # reshape
        # print(dout.size())
        dout = self.pool2.backward(dout)
        # print("pool2:",dout.size())
        dout = self.Sigm2.backward(dout)
        # print("sigm2:",dout.size())
        dout = self.conv2.backward(dout)
        # print("conv2:", dout.size())
        dout = self.pool1.backward(dout)
        # print("pool1:", dout.size())
        dout = self.Sigm1.backward(dout)
        # print("Sigm1:", dout.size())
        dout = self.conva.backward(dout)
        dout = self.conv1.backward(dout)

    def get_params(self):
        return [self.conv1.w, self.conv1.b, self.conv2.w, self.conv2.b, self.conva.w, self.conva.b ,
                self.FC1.w, self.FC1.b, self.FC2.w, self.FC2.b, self.FC3.w, self.FC3.b]

    def set_params(self, params):
        [self.conv1.w, self.conv1.b, self.conv2.w, self.conv2.b, self.conva.w, self.conva.b ,
                self.FC1.w, self.FC1.b, self.FC2.w, self.FC2.b, self.FC3.w, self.FC3.b] = params


class TwoLayerNet(Net):

    #Simple 2 layer NN

    def __init__(self, N, D_in, H, D_out, weights=''):
        self.FC1 = Linear_Layer(D_in, H)
        self.ReLU1 = Sigmoid()
        self.FC2 = Linear_Layer(H, D_out)
        self.Softmax = Softmax()
        if weights == '':
            pass
        else:
            with open(weights,'rb') as f:
                params = pickle.load(f)
                self.set_params(params)

    def forward(self, X,pred = False):


        if pred:
            pass
        else:
            self.f1 = X.size()

        fl = X.view(X.size()[0], torch.prod(torch.tensor(X.size())[1:]))
        h1 = self.FC1.forward(fl,pred)
        a1 = self.ReLU1.forward(h1,pred)
        h2 = self.FC2.forward(a1,pred)
        s1 = self.Softmax.forward(h2,pred)
        return s1

    def backward(self, dout):
        dout = self.FC2.backward(dout)
        dout = self.ReLU1.backward(dout)
        dout = self.FC1.backward(dout)
        dout = dout.view(self.f1)
    def get_params(self):
        return [self.FC1.w, self.FC1.b, self.FC2.w, self.FC2.b]

    def set_params(self, params):
        [self.FC1.w, self.FC1.b, self.FC2.w, self.FC2.b] = params

def NLLLoss(Y_pred, Y_true):
    """
    Negative log likelihood loss
    """

    loss = 0.0
    N = Y_pred.shape[0]
    M = torch.sum(Y_pred*Y_true, axis=1)
    for e in M:
        #print(e)
        if e == 0:
            loss += 500
        else:
            loss += -torch.log(e)
    return loss/N

class CrossEntropyLoss():
    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):

        Y_pred = Y_pred.to(torch.float64).to(device)
        Y_true = Y_true.to(torch.float64).to(device)

        N = Y_pred.shape[0]
        softmax = Softmax()
        prob = softmax.forward(Y_pred)
        loss = NLLLoss(prob, Y_true)
        Y_serial = torch.argmax(Y_true, axis=1)#這邊的進去的 Y 必須爲 （10，1）不能為（10，）
        dout = prob.clone()
        dout[torch.arange(N), Y_serial] -= 1
        return loss.to(torch.float16), dout.to(torch.float16)


class SGD():
    def __init__(self, params, lr=0.001, reg=0):
        self.parameters = params
        self.lr = lr
        self.reg = reg

    def step(self):
        for param in self.parameters:
            param['val'] -= (self.lr*param['grad'] + self.reg*param['val'])

# model = ThreeLayerNet(batch_size, D_in, H1, H2, D_out)
#
#
# losses = []
# #optim = optimizer.SGD(model.get_params(), lr=0.0001, reg=0)
# optim = SGDMomentum(model.get_params(), lr=0.0001, momentum=0.80, reg=0.00003)
# criterion = CrossEntropyLoss()
#
# # TRAIN
# ITER = 25000
# for i in range(ITER):
# 	# get batch, make onehot
# 	X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)
# 	Y_batch = MakeOneHot(Y_batch, D_out)
#
# 	# forward, loss, backward, step
# 	Y_pred = model.forward(X_batch)
# 	loss, dout = criterion.get(Y_pred, Y_batch)
# 	model.backward(dout)
# 	optim.step()
#
# 	if i % 100 == 0:
# 		print("%s%% iter: %s, loss: %s" % (100*i/ITER,i, loss))
# 		losses.append(loss)



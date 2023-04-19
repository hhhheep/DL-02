from torch import cuda,device
import torch
import numpy as np


device = device('cuda:1' if cuda.is_available() else "cpu")## .to(device)

##

#### 我們假設 輸入的圖像是以 batch 為形式輸入的
#### (batch_size, input_channels, input_height, input_width)

#### 我希望 僅使用 torch中的tensor 來爲自己的code加速,所以在輸入時 x 應該轉換成tensor並移動到 CUDA中
#使用的torch中的function 都可以在 numpy中找到平替

#參考的 github : https://github.com/toxtli/lenet-5-mnist-from-scratch-numpy/blob/master/app.py#L233
##

class Linear_Layer():

    def __init__(self,input_size,output_size,lr = 0.0001):
        self.lr = lr
        self.cache = None
        self.w = {"val" : np.random.normal(0.0,np.sqrt(2/input_size),size = (input_size,output_size)),
                  "grad" : 0}
        self.b = {"val" : np.random.rand(output_size),"grad" : 0}

        self.w["val"] = torch.from_numpy(self.w["val"]).to(torch.float16).to(device)
        self.b["val"] = torch.from_numpy(self.b["val"]).to(torch.float16).to(device)

    def forward(self,x,pred = False):
        x = torch.as_tensor(x).to(torch.float16).to(device)
        out = torch.matmul(x,self.w["val"]) + self.b["val"]
        if pred:
            pass
        else:
            self.cache = x

        return out

    def backward(self,dout):

        x = self.cache
        dx = torch.matmul(dout, self.w['val'].T).view(x.size())        #如果 w 維度是 [1,27] 那麽 他期望的dout為 [batch_size,1]
        self.w['grad'] = torch.matmul(x.resize(x.size()[0],torch.prod(torch.tensor(x.size())[1:])).T,dout)
        self.b['grad'] = torch.sum(dout, dim = 0)

        return dx

    def update(self):

        self.W['val'] -= self.lr * self.W['grad']
        self.b['val'] -= self.lr * self.b['grad']


class Conv2d():

    def __init__(self,in_c,out_c,kernel_size,stride = 1,padding = 0):
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.S = stride
        self.pad = padding

        self.w = {"val": np.random.normal(0.0, np.sqrt(2 / in_c), size=(out_c,in_c,kernel_size,kernel_size)),"grad" : 0}
        self.b = {"val": np.random.rand(out_c),"grad" : 0}
        self.w["val"] = torch.from_numpy(self.w["val"]).to(torch.float16).to(device)
        self.b["val"] = torch.from_numpy(self.b["val"]).to(torch.float16).to(device)

        self.cache = None

    def forward(self,x,pred = False):
        x = x.to(torch.float16).to(device)
        x = torch.nn.functional.pad(x, [self.pad,self.pad,self.pad,self.pad], mode='constant', value=0)
        (N, Cin, H, W) = x.size()

        H_ = H - self.kernel_size + 1
        W_ = W - self.kernel_size + 1

        # Y = torch.zeros((N, self.out_c, H_, W_))


        # for n in range(N):
        #     for c in range(self.out_c):
        #         for h in range(H_):
        #             for w in range(W_):
        #                 Y[n, c, h, w] = torch.sum(x[n, :, h:h + self.kernal_size, w:w + self.kernal_size] * self.w['val'][c, :, :, :]) + \
        #                                 self.b['val'][c]

        # 將 需要進行滑塊操作的 矩陣轉換成 和 kernel 一樣大小的 數個矩陣快 這部是爲了 加速conV的運行，在numpy中也有類似的操作 function叫 numpy.lib.stride_tricks.as_strided
        #效果是相同的

        x_unflod = x.unfold(2,self.kernel_size,self.S)
        x_unflod = x_unflod.unfold(3,self.kernel_size,self.S)

        # torch.einsum 簡化 for去計算w 這樣就能一部到位的計算出結果

        Y = torch.einsum("nchwkj,dckj->ndhw", x_unflod, self.w["val"]) #w 大小是 (filter，in_chennel,kernel_h,kernel_w)

        # 'n'：批样本数；
        # 'c'：输入通道数；
        # 'h'：输出特征图的高度；
        # 'w'：输出特征图的宽度；
        # 'd'：卷积核的个数（即输出通道数）；
        # 'k'：卷积核的高度（即卷积核的行数）；
        # 'j'：卷积核的宽度（即卷积核的列数）。

        if pred:
            pass
        else:
            self.cache = x

        return Y

    def backward(self,dout):

        dout = torch.as_tensor(dout).to(torch.float16).to(device)
        # dout (N,Cout,H_,W_)
        # W (Cout, Cin, F, F)
        X = self.cache.to(torch.float16).to(device)
        (N, Cin, H, W) = X.shape
        H_ = H - self.kernel_size + 1
        W_ = W - self.kernel_size + 1
        W_rot = torch.rot90(torch.rot90(self.w['val']))

        dX = torch.zeros(X.shape).to(torch.float16).to(device)
        dW = torch.zeros(self.w['val'].size()).to(torch.float16).to(device)
        db = torch.zeros(self.b['val'].size()).to(torch.float16).to(device)

        # dW
        for co in range(self.in_c):
            for ci in range(Cin):
                for h in range(self.kernel_size):
                    for w in range(self.kernel_size):
                        dW[co, ci, h, w] = torch.sum(X[:, ci, h:h + H_, w:w + W_] * dout[:, co, :, :])

        # db
        for co in range(self.out_c):
            db[co] = torch.sum(dout[:, co, :, :])

        self.w['grad'] = dW.to(torch.float16).to(device)
        self.b['grad'] = db.to(torch.float16).to(device)

        dout_pad = torch.nn.functional.pad(dout, [self.kernel_size,self.kernel_size,self.kernel_size,self.kernel_size], 'constant',value=0) # torch的pad和numpy的有點不一樣 四個格子 好像代表上下左右四個方向


        ### 這部分我也想進行修改，但實力有限 真的不知道 怎麽將滑塊取出來后 進行一部到位的sum
        # for n in range(N):
        for ci in range(Cin):
            for h in range(H):
                for w in range(W):
                    # print("self.F.shape: %s", self.F)
                    # print("%s, W_rot[:,ci,:,:].shape: %s, dout_pad[n,:,h:h+self.F,w:w+self.F].shape: %s" % ((n,ci,h,w),W_rot[:,ci,:,:].shape, dout_pad[n,:,h:h+self.F,w:w+self.F].shape))
                    dX[:, ci, h, w] = torch.sum(W_rot[:, ci, :, :] * dout_pad[:, :, h:h + self.kernel_size, w:w + self.kernel_size])

        # x_unflod = dout_pad.unfold(2, self.kernal_size, self.S)
        # x_unflod = x_unflod.unfold(3, self.kernal_size, self.S)
        #
        #
        # (n,c,h,w,i,j)= x_unflod.shape
        # x_unflod = x_unflod.reshape((n,c,h*w,i,j))
        #
        # dX = torch.zeros((n,c,h*w))

        # for n in range(n):
        #     for ci in range(c):
        #         for i in range(h*w):
        #             dX[n, ci, i] = torch.sum(W_rot[:, ci, :, :] * x_unflod[n, :,i])

        # dX = torch.einsum("n.hwkj,.c..->nchw", x_unflod, W_rot)  # w 大小是 (filter，in_chennel,kernel_h,kernel_w)


        return dX



class Sigmoid():
    """
    Sigmoid activation layer
    """
    def __init__(self):
        self.cache = None

    def forward(self, X, pred=False):
        if pred:
            return 1 / (1 + torch.exp(-X))
        else:
            self.cache = torch.as_tensor(X).to(torch.float16).to(device)
            return 1 / (1 + torch.exp(-self.cache))

    def backward(self, dout):
        X = self.cache
        X = self.forward(X,pred=True)
        dX = dout*X*(1-X)
        return dX



class MaxPool():
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.S = stride
        self.cache = None

    def forward(self, X,pred = False):

        X = torch.as_tensor(X).to(torch.float16).to(device)
        # X: (N, Cin, H, W): maxpool along 3rd, 4th dim
        (N,Cin,H,W) = X.shape
        # F = self.kernel_size
        W_ = int(float(W)/self.kernel_size)
        H_ = int(float(H)/self.kernel_size)



        # for n in range(N):
        #     for cin in range(Cin):
        #         for w_ in range(W_):
        #             for h_ in range(H_):
        #                 Y[n,cin,w_,h_] = np.max(X[n,cin,F*w_:F*(w_+1),F*h_:F*(h_+1)])
        #                 i,j = np.unravel_index(X[n,cin,F*w_:F*(w_+1),F*h_:F*(h_+1)].argmax(), (F,F))
        #                 M[n,cin,F*w_+i,F*h_+j] = 1

        patches = X.unfold(2, self.kernel_size, self.S).unfold(3, self.kernel_size, self.S)
        patches = patches.reshape(N, Cin, -1, 1,self.kernel_size * self.kernel_size)

        M = torch.zeros(patches.shape).to(torch.float16).to(device)  # mask

        # Find max values and indices of max values
        max_values, indices = patches.max(dim=4, keepdim=True)

        # Use fold to reshape the output tensor
        Y = max_values.reshape(N, Cin, H_, W_)
        # Y = Y.transpose(2, 3).transpose(1, 2)

        # Save the mask for backpropagation
        if pred:
            pass
        else:
            self.cache = M.scatter_(4, indices, 1).view(X.shape).to(torch.float16).to(device)

        # self.cache = M

        return Y

    def backward(self, dout):

        M = self.cache
        (N, Cin, H, W) = M.shape
        dout = torch.as_tensor(dout).to(torch.float16).to(device)
        # print("dout.shape: %s, M.shape: %s" % (dout.shape, M.shape))
        dX = torch.zeros(M.shape).to(torch.float16).to(device)
        for n in range(N):
            for c in range(Cin):
                # print("(n,c): (%s,%s)" % (n,c))
                dX[n, c, :, :] = dout[n, c, :, :].repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

        return dX*M



class Softmax():
    """
    Softmax activation layer
    """
    def __init__(self):
        #print("Build Softmax")
        self.cache = None

    def forward(self, X,pred = False):

        X = torch.as_tensor(X).to(torch.float16).to(device)
        #print("Softmax: _forward")
        meanes = torch.mean(X, axis=1)
        meanes = meanes.reshape(meanes.shape[0], 1)
        Y = torch.exp(X - meanes)
        Y[torch.isinf(Y)] = 1000
        Z = Y / torch.sum(Y, axis=1).reshape(Y.shape[0], 1)

        Y = Y.to(torch.float16).to(device)
        Z = Z.to(torch.float16).to(device)

        if pred:
            pass
        else:
            self.cache = (X, Y, Z)

        return Z # distribution

    def backward(self, dout):
        X, Y, Z = self.cache
        dZ = torch.zeros(X.shape).to(torch.float16).to(device)
        dY = torch.zeros(X.shape).to(torch.float16).to(device)
        # dX = torch.zeros(X.shape)

        N = X.shape[0]
        for n in range(N):
            i = torch.argmax(Z[n])
            dZ[n,:] = torch.diag(Z[n]) - torch.outer(Z[n],Z[n])
            M = torch.zeros((N,N)).to(torch.float16).to(device)
            M[:,i] = 1
            dY[n,:] = torch.eye(N) - M
        dX = torch.dot(dout,dZ)
        dX = torch.dot(dX,dY)
        return dX


class ReLU():
    """
    ReLU activation layer
    """
    def __init__(self):
        #print("Build ReLU")
        self.cache = None

    def forward(self, X,pred = False):
        #print("ReLU: _forward")
        X = torch.as_tensor(X).to(torch.float16).to(device)
        out = torch.maximum(torch.tensor(0, dtype=torch.float16, device=device), X)

        if pred:
            pass
        else:
            self.cache = X

        return out

    def backward(self, dout):
        #print("ReLU: _backward")
        X = self.cache
        dX = torch.as_tensor(dout).to(torch.float16).to(device)
        dX[X <= 0] = 0
        return dX


class X_sigmoid():
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, pred=False):
        self.x = torch.as_tensor(x).to(torch.float16).to(device)
        self.y = self.x * (1 / (1 + torch.exp(-self.x)))
        if not pred:
            self.cache = self.x, self.y.to(torch.float16).to(device)
        return self.y

    def backward(self, grad_output):
        grad_output = torch.as_tensor(grad_output).to(torch.float16).to(device)
        x, y = self.cache
        sigmoid = 1 / (1 + torch.exp(-x))
        grad_input = grad_output * (sigmoid + x * sigmoid * (1 - sigmoid))
        return grad_input

class Xsigmoid():
    def __init__(self):
        pass
        # super(Xsigmoid, self).__init__()

    def forward(self, x):
        self.x = x.to(torch.float16).to(device)
        output = x / (1 + torch.exp(-x))
        return output

    def backward(self, dout):
        dout = dout.to(torch.float16).to(device)
        x = self.x
        sigmoid = 1 / (1 + torch.exp(-x))
        grad_input = dout * (1 + x * sigmoid * (1 - sigmoid)) / (1 + torch.exp(-x))**2
        return grad_input



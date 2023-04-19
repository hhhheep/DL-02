from read_ import *
import math
from moudel import *
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from nn import Softmax

def confusion_matrix(preds, labels):
    size = 50
    conf_matrix = np.zeros(size*size).reshape((size,size))
    # preds = np.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p-1, t-1] += 1
    return conf_matrix

def Navie_Letnet():
    train_inf = read_local_txt("train")
    vail_inf = read_local_txt("val")
    test_inf = read_local_txt("test")

    batch_size = 100
    epotches = 1
    train_len = train_inf.shape[0]
    train_class = train_inf.iloc[:, 1].unique()
    sum_ba = 0

    # D_in = 500 * 3 * 256*256
    H = 256
    # D_out = len(train_class)

    model = naive_LeNet5()

    losses = []
    optim = SGD(model.get_params(), lr=0.00001, reg=0)
    criterion = CrossEntropyLoss()

    train_acc = []
    vail_acc = []
    for epotch in range(epotches):
        train_inf1 = train_inf
        for n in tqdm(range(int(train_len / batch_size))):
            # if n == int(633):
            #     print("break")
            #     break

            batch, label, train_inf1, batch_size1 = train_batch(train_inf1, batch_size=batch_size)
            train_imges = read_image(batch)
            sum_ba += batch_size1

            Batc = torch.as_tensor(train_imges, dtype=torch.float16).view(batch_size1, 3, 256, 256)
            train_label = torch.as_tensor(label.to_numpy()).view(batch_size1, 1)

            vail_bat, vail_l, _, batch_size2 = train_batch(vail_inf, batch_size=50)
            vail_img = read_image(vail_bat)

            vail_img = torch.as_tensor(vail_img, dtype=torch.float16).view(batch_size2, 3, 256, 256)
            vail_l = torch.as_tensor(vail_l.to_numpy()).view(batch_size2, 1)

            # forward, loss, backward, step
            Y_pred = model.forward(Batc)
            V_pred = model.forward(vail_img, pred=True)
            loss, dout = criterion.get(Y_pred, train_label)
            loss = loss.to("cpu").detach().numpy()
            # train_l_one = torch.nn.functional.one_hot(train_label).view(batch_size1,50).to(device)
            # dout -= train_l_one
            _, Y_ = torch.max(Y_pred, dim=1)
            _, v_ = torch.max(V_pred, dim=1)

            T_acc = sum(np.diag(
                confusion_matrix(Y_.to("cpu").detach().numpy(), train_label.to("cpu").detach().numpy()))) / batch_size
            V_acc = sum(np.diag(
                confusion_matrix(v_.to("cpu").detach().numpy(), vail_l.to("cpu").detach().numpy()))) / batch_size

            train_acc.append(T_acc)
            vail_acc.append(V_acc)
            losses.append(loss)
            # epotch = 0
            print("-------" + "epotch" + str(epotch+1) + '/' + str(epotches) + "-------" + str(sum_ba) + "/" + str(
                train_len) + "------" + "train_acc:" + str(T_acc) + " //vali_acc:" + str(V_acc) + " //loss:" + str(
                loss))

        model.backward(dout)
        optim.step()

    xlab = len(train_acc)
    xlab = range(xlab)
    xlab = list(xlab)
    plt.plot(xlab, train_acc, 's-', color='r', label="gbm_train_acc")  # s-:方形
    plt.plot(xlab, vail_acc, 'o-', color='g', label="gbm_vali_acc")  # o-:圆形
    plt.ylim((0, 1))
    plt.xlabel("epotch")  # 横坐标名字
    plt.ylabel("accuracy")  # 纵坐标名字
    plt.title("Navie_Lenet")
    plt.legend(loc="best")  # 图例

    plt.show()

    test_img, test_lab, _, _ = train_batch(test_inf, batch_size=batch_size)
    test_img = read_image(test_img, img_size=256)
    test_lab = torch.as_tensor(test_lab.to_numpy()).view(batch_size, 1)

    T_pred = model.forward(torch.as_tensor(test_img), pred=True)
    _, T_ = torch.max(T_pred, dim=1)

    t_acc = sum(
        np.diag(confusion_matrix(T_.to("cpu").detach().numpy(), test_lab.to("cpu").detach().numpy()))) / batch_size
    print(t_acc)


def Letnet():
    train_inf = read_local_txt("train")
    vail_inf = read_local_txt("val")
    test_inf = read_local_txt("test")

    batch_size = 100
    epotches = 1
    train_len = train_inf.shape[0]
    train_class = train_inf.iloc[:, 1].unique()
    sum_ba = 0

    img_size = 256


    model = LeNet5()

    losses = []
    optim = SGD(model.get_params(), lr=0.00001, reg=0)
    criterion = CrossEntropyLoss()

    train_acc = []
    vail_acc = []
    for epotch in range(epotches):
        train_inf1 = train_inf
        for n in tqdm(range(int(train_len / batch_size))):
            batch, label, train_inf1, batch_size1 = train_batch(train_inf1, batch_size=batch_size)
            train_imges = read_image(batch,img_size=img_size)
            sum_ba += batch_size1

            Batc = torch.as_tensor(train_imges, dtype=torch.float16).view(batch_size1, 3, img_size, img_size)
            train_label = torch.as_tensor(label.to_numpy()).view(batch_size1, 1)

            vail_bat, vail_l, _, batch_size2 = train_batch(vail_inf, batch_size=50)
            vail_img = read_image(vail_bat,img_size=img_size)

            vail_img = torch.as_tensor(vail_img, dtype=torch.float16).view(batch_size2, 3, img_size, img_size)
            vail_l = torch.as_tensor(vail_l.to_numpy()).view(batch_size2, 1)

            # forward, loss, backward, step
            Y_pred = model.forward(Batc)
            V_pred = model.forward(vail_img, pred=True)
            loss, dout = criterion.get(Y_pred, train_label)
            loss = loss.to("cpu").detach().numpy()

            _, Y_ = torch.max(Y_pred, dim=1)
            _, v_ = torch.max(V_pred, dim=1)

            T_acc = sum(np.diag(
                confusion_matrix(Y_.to("cpu").detach().numpy(), train_label.to("cpu").detach().numpy()))) / batch_size
            V_acc = sum(np.diag(
                confusion_matrix(v_.to("cpu").detach().numpy(), vail_l.to("cpu").detach().numpy()))) / batch_size

            train_acc.append(T_acc)
            vail_acc.append(V_acc)
            losses.append(loss)
            # epotch = 0
            print("-------" + "epotch" + str(epotch+1) + '/' + str(epotches) + "-------" + str(sum_ba) + "/" + str(
                train_len) + "------" + "train_acc:" + str(T_acc) + " //vali_acc:" + str(V_acc) + " //loss:" + str(
                loss))

            model.backward(dout)
            optim.step()

    xlab = len(train_acc)
    xlab = range(xlab)
    xlab = list(xlab)
    plt.plot(xlab, train_acc, 's-', color='r', label="gbm_train_acc")  # s-:方形
    plt.plot(xlab, vail_acc, 'o-', color='g', label="gbm_vali_acc")  # o-:圆形
    plt.ylim((0, 1))
    plt.xlabel("epotch")  # 横坐标名字
    plt.ylabel("accuracy")  # 纵坐标名字
    plt.title("Letnet")
    plt.legend(loc="best")  # 图例

    plt.show()

    test_img, test_lab, _, _ = train_batch(test_inf, batch_size=batch_size)
    test_img = read_image(test_img,img_size=img_size)
    test_lab = torch.as_tensor(test_lab.to_numpy()).view(batch_size, 1)


    T_pred = model.forward(torch.as_tensor(test_img), pred=True)
    _, T_ = torch.max(T_pred, dim=1)



    t_acc = sum(
        np.diag(confusion_matrix(T_.to("cpu").detach().numpy(), test_lab.to("cpu").detach().numpy()))) / batch_size
    print(t_acc)


def NN_train():
    train_inf = read_local_txt("train")
    vail_inf = read_local_txt("val")
    test_inf = read_local_txt("test")

    batch_size = 500
    epotches = 1
    train_len = train_inf.shape[0]
    train_class = train_inf.iloc[:, 1].unique()
    sum_ba = 0

    # D_in = 500 * 3 * 256*256
    H = 256
    # D_out = len(train_class)
    img_size = 256
    model = TwoLayerNet(batch_size, 3 * 256 * 256, H, 50)

    losses = []
    optim = SGD(model.get_params(), lr=0.00001, reg=0)
    criterion = CrossEntropyLoss()

    train_acc = []
    vail_acc = []
    for epotch in range(epotches):
        train_inf1 = train_inf
        for n in tqdm(range(int(train_len / batch_size))):

            batch, label, train_inf1, batch_size1 = train_batch(train_inf1, batch_size=batch_size)
            train_imges = read_image(batch,img_size=img_size)
            sum_ba += batch_size1

            Batc = torch.as_tensor(train_imges, dtype=torch.float16).view(batch_size1, 3, 256, 256)
            train_label = torch.as_tensor(label.to_numpy()).view(batch_size1, 1)

            vail_bat, vail_l, _, batch_size2 = train_batch(vail_inf, batch_size=50)
            vail_img = read_image(vail_bat,img_size=img_size)

            vail_img = torch.as_tensor(vail_img, dtype=torch.float16).view(batch_size2, 3, 256, 256)
            vail_l = torch.as_tensor(vail_l.to_numpy()).view(batch_size2, 1)


            # forward, loss, backward, step
            Y_pred = model.forward(Batc)
            V_pred = model.forward(vail_img,pred=True)
            loss, dout = criterion.get(Y_pred, train_label)
            loss = loss.to("cpu").detach().numpy()

            _,Y_ = torch.max(Y_pred,dim=1)
            _,v_ = torch.max(V_pred,dim=1)


            T_acc = sum(np.diag(confusion_matrix(Y_.to("cpu").detach().numpy(),train_label.to("cpu").detach().numpy())))/batch_size
            V_acc = sum(np.diag(confusion_matrix(v_.to("cpu").detach().numpy(),vail_l.to("cpu").detach().numpy())))/batch_size

            train_acc.append(T_acc)
            vail_acc.append(V_acc)
            losses.append(loss)

            print("-------" + "epotch" + str(epotch+1) + '/' + str(epotches) +"-------" +str(sum_ba) + "/" + str(train_len) + "------" + "train_acc:" + str(T_acc) + " //vali_acc:" + str(V_acc) + " //loss:" + str(loss))

            model.backward(dout)
            optim.step()

    xlab = len(train_acc)
    xlab = range(xlab)
    xlab = list(xlab)
    plt.plot(xlab, train_acc, 's-', color='r', label="gbm_train_acc")  # s-:方形
    plt.plot(xlab, vail_acc, 'o-', color='g', label="gbm_vali_acc")  # o-:圆形
    plt.ylim((0, 1))
    plt.xlabel("epotch")  # 横坐标名字
    plt.ylabel("accuracy")  # 纵坐标名字
    plt.title("NN-net")
    plt.legend(loc="best")  # 图例

    plt.show()

    test_img = read_image(test_inf["image"],img_size=img_size)
    test_lab = torch.as_tensor(test_inf["label"].to_numpy()).view(test_inf.shape[0], 1)

    T_pred = model.forward(test_img, pred=True)
    _, T_ = torch.max(T_pred, dim=1)

    t_acc = sum(np.diag(confusion_matrix(T_.to("cpu").detach().numpy(), test_lab.to("cpu").detach().numpy()))) / batch_size
    print(t_acc)


if __name__ == '__main__':
    # NN_train()
    Letnet()
import numpy as np
import matplotlib.pyplot as plt
from os import path

def generate_binary_strings(bit_count):
    binary_strings = []
    def genbin(n, bs=''):
        if len(bs) == n:
            binary_strings.append(bs)
        else:
            genbin(n, bs + '0')
            genbin(n, bs + '1')


    genbin(bit_count)
    return binary_strings


#parse binary strings into weight matrices
#first 8 are first layer's weights;
#second 8 are second layer's weights;
#last 2 are third layer's weights

 #Element-wise ReLU
def relu(x):
    return np.maximum(np.zeros_like(x), x)

def ntwk(x1, x2,
        #weights
        w11, w12, w13, w14, w21, w22, w23, w24,
        u11, u12, u21, u22, u31, u32, u41, u42,
        v11, v21
    ):
    w = np.array([[w11, w12, w13, w14],
                    [w21, w22, w23, w24]])
    u = np.array([[u11, u12],
                    [u21, u22],
                    [u31, u32],
                    [u41, u42]])
    v = np.array([[v11],
                    [v21]])
    layer1out = relu((np.array([x1, x2]) @ w))
    layer2out = relu((layer1out @ u))
    layer3out = (layer2out @ v)
    return np.ndarray.item(layer3out)


vntwk = np.vectorize(ntwk)   

def plotntwk(binarystring):
    ws = []
    for b in binarystring:
        if b == '0':
            ws.append(-1)
        elif b == '1':
            ws.append(1)
    x = np.linspace(0, 1, 8)
    y = np.linspace(0, 1, 12)
    X, Y = np.meshgrid(x, y)
    data = vntwk(X, Y,
                ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7],
                ws[8], ws[9], ws[10], ws[11], ws[12], ws[13], ws[14], ws[15],
                ws[16], ws[17])
    #if not trivial, save:
    if not(np.all(data == data[0])):
        plt.imshow(data, origin='lower')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.savefig(path.join('enumerateimg', binarystring))
        with open(path.join('enumeratedata', f'{binarystring}.txt'), 'w') as f:
            f.write(f'{data}')
        plt.close()

binary_strings = generate_binary_strings(18)

for b in binary_strings:
    plotntwk(b)
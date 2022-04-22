import numpy as np
import matplotlib.pyplot as plt
from os import path

#Generate random networks until a nontrivial network is found
def randntwk(filename):
    #do while
    #https://stackoverflow.com/questions/46820182/randomly-generate-1-or-1-positive-or-negative-integer
    #this is fast "because any other implementation does something like this"
    r = np.random.rand(2,4)
    w1 = np.where(r < .5, -1, 1)

    #random float 0 to 1, then random sign
    r = np.random.rand(4)
    s = np.random.rand(4)
    b1 = np.where(s < .5, -r, r)

    r = np.random.rand(4,2)
    w2 = np.where(r < .5, -1, 1)

    r = np.random.rand(2)
    s = np.random.rand(2)
    b2 = np.where(s < .5, -r, r)

    r = np.random.rand(2,1)
    w3 = np.where(r < .5, -1, 1)

    r = np.random.rand(1)
    s = np.random.rand(1)
    b3 = np.where(s < .5, -r, r)


    #Element-wise ReLU
    def relu(x):
        return np.maximum(np.zeros_like(x), x)


    def ntwk(x1, x2):
        layer1out = relu((np.array([x1, x2]) @ w1) + b1)
        layer2out = relu((layer1out @ w2) + b2)
        layer3out = (layer2out @ w3) + b3
        return np.ndarray.item(layer3out)


    # data = []
    # binarydata = []
    # for x1 in np.linspace(-1, 1, 8):
    #     for x2 in np.linspace(-1, 1, 12):
    #         data.append(x1, x2, ntwk(x1, x2))

    vntwk = np.vectorize(ntwk)

    x = np.linspace(0, 1, 8)
    y = np.linspace(0, 1, 12)
    X, Y = np.meshgrid(x, y)
    data = vntwk(X, Y)
    
    databinary = np.where(data < 0, -1, 1)

    # def plot_triplets(data):
        

    fig, ax = plt.subplots()
    img = ax.imshow(data, origin='lower')
    fig.colorbar(img)
    plt.savefig(path.join('dataimg', filename))
    plt.close()
    fig, ax = plt.subplots()
    img = ax.imshow(databinary, origin='lower')
    fig.colorbar(img)
    plt.savefig(path.join('binarydataimg', filename))
    with open(path.join('data', f'{filename}.txt'), 'w') as f:
        f.write(f'{data}')
    with open(path.join('networks', f'{filename}.txt'), 'w') as f:
        f.write(f'layer 1 weights: {w1}')
        f.write(f'layer 1 bias: {b1}')
        f.write(f'layer 2 weights: {w2}')
        f.write(f'layer 2 bias: {b2}')
        f.write(f'layer 3 weights: {w3}')
        f.write(f'layer 3 bias: {b3}')
    plt.close()
        
for i in range(100):
    randntwk(f'{i}')
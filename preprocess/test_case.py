import os
import numpy as np


if __name__ == '__main__':
    data = np.asarray([[-0.1,0.3],[1.2,-0.4]])
    print data

    data[data<0] = 0
    data[data>1] = 1
    print data
import torch
from torch.autograd import Variable
if __name__ == '__main__':
    x = torch.randn(3, 3)
    x = x.view(-1)
    # y = x.sum()
    # print y.type()
    # assert False
    y = torch.min(x, torch.ones_like(x))
    print x
    print y




    # y = x.view(-1)
    # print y
    # z = (y > 0.2) & (y < 0.6)
    # print z








    # z = y.cuda()
    # print x.size()
    # print y
    # print y-z

    # print '{:d}'.format(1.2)
    # print y
    # z = (y > 0.5).type(torch.FloatTensor)
    # print z
    # print z.sum()/3
    # print z.type()
    # num_entries = z.size()[0]
    # print num_entries.type()
    # z = z.type(torch.FloatTensor)
    # print z
    # print y.type(), z.type()


    # print x.size()
    # print y.size()
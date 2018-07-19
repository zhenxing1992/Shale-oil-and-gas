import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.models as models
import pandas as pd
import time
import torch.nn.init as init

sample = pd.read_excel('./Data/train_coal.xlsx')   #the path of your training data
sample_x = sample.ix[:, 0:4].as_matrix().astype('float')   #[0:4] is your well log data
sample_x = torch.FloatTensor(sample_x)
sample_y = sample.ix[:, 4].as_matrix().astype('float')   #[4] is Toc, [5] is S1, and [6] is S2
sample_y = torch.FloatTensor(sample_y)

#gaussian initialization
def normalize(x):
    mean_data = torch.mean(x)
    std_data = torch.std(x)
    out = (x - mean_data) / (std_data)
    return out

test = pd.read_excel('./Data/test_coal.xlsx')
test_x = test.ix[:, 0:4].as_matrix().astype('float')
test_x = torch.FloatTensor(test_x)
test_y = test.ix[:, 6].as_matrix().astype('float')
test_y = torch.FloatTensor(test_y)

for j in range(3):
    sample_x[:, j] = normalize(sample_x[:, j])
    test_x[:, j] = normalize(test_x[:, j])


def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(1, 5, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv1d(5, 10, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2)
        )
        self.classifier1 = nn.Linear(10, 10)
        self.dropout = nn.Dropout(p=0.7)
        self.linear = nn.Linear(10, 1)

    #
    def forward(self, x):
        x = self.feature(x).view(-1, 10)
        output = self.dropout(self.classifier1(x))
        output = self.linear(output)

        return output
net = Net()
init_lr = 0.001
linear = nn.Linear(1, 1)
optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=init_lr, weight_decay=1e-4)
# for param in net.parameters():
#     param.requires_grad = False
Epoch = 3
lossfunc = nn.MSELoss()
best_R2 = 0
for epoch in range(Epoch):
    net.train()
    if (epoch+1) % 10000 == 0:
        init_lr = init_lr / 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = init_lr
    prediction = net(sample_x.unsqueeze(1))
    loss = lossfunc(prediction, sample_y.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    net.eval()
    prediction_t = net(sample_x.unsqueeze(1))
    seriesx_test = pd.Series(prediction_t.squeeze().data.numpy())
    seriesy_test = pd.Series(sample_y.squeeze().data.numpy())
    R1 = seriesx_test.corr(seriesy_test)
    if (R1*R1) > 0.75:
        net.eval()
        prediction = net(test_x.unsqueeze(1))
        seriesx_test = pd.Series(prediction.squeeze().data.numpy())
        seriesy_test = pd.Series(test_y.squeeze().data.numpy())
        R2 = seriesx_test.corr(seriesy_test)
        print (R2* R2)
        if best_R2 < R2:
            best_R2 = R2
            print(best_R2 * best_R2 * 100)
            torch.save(net.state_dict(), './Results/coal.pkl')

# net.load_state_dict(torch.load('./Results/coal.pkl'))
prediction = net(sample_x.unsqueeze(1))
prediction = linear(prediction)
loss = lossfunc(prediction, sample_y.unsqueeze(1))
print (loss)
store = pd.DataFrame(prediction.data.numpy())
store.to_excel('train.xlsx')
series_x = pd.Series(prediction.squeeze().data.numpy())
series_y = pd.Series(sample_y.squeeze().data.numpy())
R2 = series_x.corr(series_y)
print('R2 is: ', R2 * R2)
prediction_t = net(test_x.unsqueeze(1))
prediction_t = linear(prediction_t)
loss = lossfunc(prediction_t, test_y.unsqueeze(1))
print (loss)
store = pd.DataFrame(prediction_t.data.numpy())
store.to_excel('test.xlsx')
print(prediction_t.squeeze())
seriesx_test = pd.Series(prediction_t.squeeze().data.numpy())
seriesy_test = pd.Series(test_y.squeeze().data.numpy())
R2 = seriesx_test.corr(seriesy_test)
print('Test R2 is: ', R2 * R2)
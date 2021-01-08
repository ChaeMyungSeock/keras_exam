import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cuda 사용할 수 있으면 cuda로 아니면 cpu로

print(device)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=16,
                              kernel_size=3, padding=1, stride=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(16, affine=True)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        batch = x.size(0)  # x.shape[0]
        out0 = self.conv(x)
        out1 = self.batchnorm(out0)
        out2 = self.relu(out1)
        out3 = self.avg_pool(out2)
        out4 = out3.view(batch, -1)
        out5 = self.fc(out4)
        # out5 = nn.Functional.LogSotf

        return out5


model = SimpleModel().to(device)
# from torchsummary import summary
# from torchviz import make_dot
# from torch.autograd import Variable

# summary(model, inputsize=(3,32,32))
batch_size = 32
epochs = 3

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # pytorch에서는 model.parameters() 필요로함

transform = transforms.Compose([
    transforms.ToTensor(),  # 0 ~ 1 => 어떠한 값이든 tensor값으로 바꿔 줌
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])  # output[channel] = (input[channel] - mean[channel]) / std[channel] => -1 ~ 1

train_dataset = datasets.CIFAR10('train/',
                                 train=True,
                                 download=True,
                                 transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

valid_dataset = datasets.CIFAR10(root='test/',
                                 train=False,
                                 download=True,
                                 transform=transform)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=batch_size,
                          shuffle=False)

loss_dict = {}
val_loss_dict = {}



for epoch in range(1, epochs + 1):
    loss_list = []
    for img, label in train_loader:

        model.train()   # model.eval() 모델을 평가할 때
        # train에서 dropout의 경우 0.5일 때 0.5만 사용 but eval에서는 모두 사용
        # batchnormal의 경우 batch마다 값이 달라지지만 eval에서는 값이 고정되어 있음

        img = img.to(device) # gpu를 사용하면 gpu를 사용해서 계산하겠다. cuda c에서 global과 비슷
        label = label.to(device)

        output = model(img)
        loss = loss_fn(output, label)
        loss_list.append(loss.item())       # .item() 파라미터 확인


        optimizer.zero_grad()   # weight gradiant == 0
        loss.backward()         # weight gradiant가 update 됨
        optimizer.step()        # weight update

    loss_dict[epoch] = loss_list


    val_loss_list = []
    for val_img, val_label in valid_loader:
        model.eval()
        with torch.no_grad():  # 메모리는 적게들고 속도 빨라짐 why? 경사하강법에 의한 연산을 시행하지 않기 때문
            val_img = val_img.to(device)
            val_label = val_label.to(device)

            val_output = model(val_img)
            val_loss = loss_fn(val_output, val_label)
            val_loss_list.append(val_loss.item())  # .item() 파라미터 확인
    val_loss_dict[epoch] = val_loss_list

    print(f"[{epoch}/{epochs}] finished")
    print('==================')

torch.save(model.state_dict(), 'cifar10_model.pt')


# from torchsummary import summary
# from torchviz import make_dot
# from torch.autograd import Variable
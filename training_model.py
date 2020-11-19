import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.tensorboard import SummaryWriter

def load_data(data_path) :
    transform = transforms.Conpose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    cifar_train = dsets.CIFAR10(root=data_path,
                                train=True,
                                transform=transforms.transform)
    cifar_test = dsets.CIFAR10(root=data_path,
                                train=False,
                                transform=transforms.transform)
    train_loader = torch.utils.data.DataLoader(dataset = cifar_train,
                                                batch_size = 512,
                                                shuffle = True,
                                                drop_last = True)
    test_loader = torch.utils.data.DataLoader(dataset = cifar_test,
                                                batch_size = 4,
                                                shuffle = True,
                                                drop_last = True)                                           
    return train_loader, test_loader

def train_VGG(VGG, train_loader, test_loader) :
    cfg = [32,32,'M', 64,64,128,128,128,'M',256,256,256,512,512,512,'M']
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    vgg16 = VGG(vgg.make_layers(cfg), 10, True)
    citerion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(vgg16.parameters(), lr = 0.005, momentum = 0.9)
    lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.9)

    writer = SummaryWriter('runs/cifar_pipeline_test_1')

    epochs = 50
    # start train
    print("> start training...")
    for epoch in range(epochs) :
        running_loss = 0.0
        lr_sche.step()
        for i, data in enumerate(train_loader, 0) :
            inputs, labels = data
            
            optimizer.zero_grad()

            outputs = vgg16(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 30 == 29 : # 30 mini-batch 마다 출력
                # command에 출력
                loss_tracker(loss_plt, torch.Tensor([running_loss/30]), torch.Tensor([i + epoch*len(train_loader) ]))
                print('[%d, %5d] loss : %.3f' %
                        (epoch + 1, i + 1, running_loss / 30))

                # tensorboard 출력
                # running loss 기록
                writer.add_scalar('training loss', running_loss/30, epoch*len(train_loader) + i )
                # matplotlib figure 기록
                writer.add_figure('predictions vs actuals', plot_classes_preds(vgg16, inputs, labels),
                                    global_step = epoch * len(train_loader) + i)
                
                # running_loss 초기화
                running_loss = 0.0
    
    
    data_iter = iter(test_loader)
    images, labels = data_iter.next()
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    outputs = vgg16(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted : ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    
    correct = 0
    total = 0
    class_probs, class_preds = [], []
    print("> start testing...")
    with torch.no_grad() :
        for data in test_loader :
            images, labels = data
            output = vgg16(images)

            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            _, class_preds_batch = torch.max(output, 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)
            

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)
    print('test_probs : ', test_probs)
    print('test_preds : ', test_preds)

    # draw precision-recall curve
    def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0) :
        tensorboard_preds = test_preds == class_index
        tensorboard_probs = test_probs[:, class_index]

        writer.add_pr_curve(classes[class_index],
                            tensorboard_preds,
                            tensorboard_probs,
                            global_step = global_step)
        writer.closer()
    
    for i in range(len(classes)) :
        add_pr_curve_tensorboard(i, test_probs, test_preds)


# VGG 모델 코드 
class VGG(nn.Module) :
    def __init__(self, features, num_classes=1000, init_weights=True) :
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights :
            self._initialize_weights()

    def forward(self, x) :
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self) :
        for m in self.modules() :
            if isinstance(m, nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None :
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) :
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) :
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# predict result, prob 계산
def images_to_probs(network, images) :
    output = network(images)
    _, pred_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


# 예측 결과 / 확률 계산 matplotlib Figure
def plot_classes_preds(network, images, labels) :
    preds, probs = images_to_probs(network, images)
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4) :
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel = True)
        ax.set_title("{0}, {1:.1f}%\n(label : {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]
        ), color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


if __name__ == "__main__" :

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        '--train_data_path',
        type = str,
        help = 'input train data path'
    )

    args = argument_parser.parse_args()
    train_loader, test_loader = load_data(args.train_data_path)

    train_VGG(VGG, train_loader, test_loader)

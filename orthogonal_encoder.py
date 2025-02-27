import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import visdom
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

vis = visdom.Visdom(env='orthogonal')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Data(Dataset):
    def __init__(self, transform, path):
        self.path = glob.glob(path + '*/*/*.jpg')
        self.loader = torchvision.datasets.folder.default_loader
        self.transform = transform

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        # image, pose, id, color, type
        return self.transform(self.loader(self.path[index])), int(self.path[index][-26]), \
               int(self.path[index][-34:-31]), int(self.path[index][-30]), int(self.path[index][-28])


class OrthogonalEncoder(nn.Module):
    def __init__(self):
        super(OrthogonalEncoder, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        self.pose_E = nn.Sequential(
            resnet.layer4,
            resnet.avgpool,
        )
        self.id_E = nn.Sequential(
            resnet.layer4,
            resnet.avgpool,
        )
        self.pose_classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=8)
        )
        self.id_classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=575)
        )
        self.color_classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=10)
        )
        self.type_classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=9)
        )

    def forward(self, x):
        x = self.backbone(x)
        pose_feature = self.pose_E(x).view(x.size(0), -1)
        id_feature = self.id_E(x).view(x.size(0), -1)
        # pose feature, id feature, pose class, id class, orthogonal feature
        return pose_feature, id_feature, self.pose_classifier(pose_feature), self.id_classifier(id_feature), \
               self.color_classifier(id_feature), self.type_classifier(id_feature), \
               torch.bmm(pose_feature.unsqueeze(dim=1), id_feature.unsqueeze(dim=2))


if __name__ == '__main__':
    # model
    model = OrthogonalEncoder().to('cuda')
    # data
    train_path = '/home/ANYCOLOR2434/code/veri_pose/train/'
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500])
    ])
    trainset = Data(transform=train_transform, path=train_path)
    train_loader = DataLoader(trainset, batch_size=4, num_workers=4, pin_memory=True, shuffle=True)
    # setting
    criterion = nn.CrossEntropyLoss().to('cuda')
    orthogonal_criterion = nn.L1Loss().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)

    for epoch in tqdm(range(131)):
        print('epoch:', epoch)
        model.train()
        pose_loss = 0
        id_loss = 0
        color_loss = 0
        type_loss = 0
        orthogonal_loss = 0
        pose_acc = 0
        id_acc = 0
        color_acc = 0
        type_acc = 0
        total = 0
        for i, (img, pose, id, color, type) in enumerate(train_loader):
            img = img.to('cuda')
            pose = pose.to('cuda')
            id = id.to('cuda')
            color = color.to('cuda')
            type = type.to('cuda')
            label = torch.full((img.shape[0],), 0.0).to('cuda')
            optimizer.zero_grad()
            outputs = model(img)
            loss_pose = criterion(outputs[2], pose)
            loss_id = criterion(outputs[3], id)
            loss_color = criterion(outputs[4], color)
            loss_type = criterion(outputs[5], type)
            loss_orthogonal = orthogonal_criterion(outputs[6], label)
            loss = loss_id + loss_pose + loss_color + loss_type + loss_orthogonal
            loss.backward()
            optimizer.step()

            pose_loss += loss_id.item()
            id_loss += loss_pose.item()
            color_loss += loss_color.item()
            type_loss += loss_type.item()
            orthogonal_loss += loss_orthogonal.item()
            total += img.size(0)
            _, pose_predicted = torch.max(outputs[2].data, 1)
            _, id_predicted = torch.max(outputs[3].data, 1)
            _, color_predicted = torch.max(outputs[4].data, 1)
            _, type_predicted = torch.max(outputs[5].data, 1)
            pose_acc += (pose_predicted == pose).sum().item()
            id_acc += (id_predicted == id).sum().item()
            color_acc += (color_predicted == color).sum().item()
            type_acc += (type_predicted == type).sum().item()

        scheduler.step()

        if epoch % 10 == 0:
            os.makedirs('/home/ANYCOLOR2434/code/gan/weights', exist_ok=True)
            os.makedirs('/home/ANYCOLOR2434/code/gan/weights/Encoder', exist_ok=True)
            torch.save(model.state_dict(), ('/home/ANYCOLOR2434/code/gan/weights/Encoder/model_{}.pth'.format(epoch)))

        vis.line(
            Y=np.column_stack((pose_loss/len(train_loader), id_loss/len(train_loader), color_loss/len(train_loader),
                               type_loss/len(train_loader), orthogonal_loss/len(train_loader))),
            X=np.column_stack((epoch, epoch, epoch, epoch, epoch)),
            win='Learning curve',
            update='append',
            name=['pose_loss','id_loss','color_loss','type_loss','orthogonal_loss'],
            opts={
                'title': 'Learning curve',
            }
        )
        vis.line(
            Y=np.column_stack((pose_acc/total, id_acc/total, color_acc/total, type_acc/total)),
            X=np.column_stack((epoch, epoch, epoch, epoch)),
            win='accuracy curve',
            name=['pose_acc','id_acc','color_acc','type_acc'],
            update='append',
            opts={
                'title': 'accuracy curve',
            }
        )


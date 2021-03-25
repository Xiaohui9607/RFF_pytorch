import torch
import torchvision
import torchvision.transforms as transforms


def get_data(opt):
    '''
    see https://pytorch.org/vision/stable/datasets.html?highlight=dataset
    '''
    if opt.dataset == "cifar-10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchsize,
                                                  shuffle=True, num_workers=8)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchsize,
                                                 shuffle=False, num_workers=2)
        meta_data={'img_size':32, "n_class":10}

    elif opt.dataset == 'mnist':
        pass

    elif opt.dataset == '':
        pass

    return trainloader, testloader, meta_data


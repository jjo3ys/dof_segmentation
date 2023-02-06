import argparse
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

from dataset.tooth import SegmentationDataset

from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import SGD

from configs import config
from configs import update_config

from matplotlib import pyplot as plt

from model.unet import UNet

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/pidnet_small_tooth.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()
    
    if args.seed > 0:
        import random
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    
    gpus = [0]
        
    batch_size = len(gpus) * config.TRAIN.BATCH_SIZE_PER_GPU
    train_dataset = SegmentationDataset(root=config.DATASET.ROOT,
                                       list_path=config.DATASET.TRAIN_SET,
                                       num_classes=config.DATASET.NUM_CLASSES)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)
    
    test_dataset = SegmentationDataset(root=config.DATASET.ROOT,
                                       list_path=config.DATASET.TEST_SET,
                                       num_classes=config.DATASET.NUM_CLASSES)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)    

    trainSteps = len(train_dataset) // config.TRAIN.BATCH_SIZE_PER_GPU
    testSteps = len(test_dataset) // config.TRAIN.BATCH_SIZE_PER_GPU

    end_epoch = config.TRAIN.END_EPOCH
    real_end = 120+1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch
    
    model = UNet()
    model = nn.DataParallel(model).cuda()
    lossFunc = CrossEntropyLoss().cuda()

    opt = SGD(model.parameters(),
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            nesterov=config.TRAIN.NESTEROV,
            )
    train_loss = []
    test_loss = []
    for epoch in range(real_end):
        model.train()

        totalTrainLoss = 0
        totalTestLoss = 0

        for x, y, _ in trainloader:
            x = x.cuda()
            y = y.float().cuda()
            pred = model(x)
            loss = lossFunc(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            totalTrainLoss += loss

        with torch.no_grad():
            model.eval()

            for x, y, _ in testloader:
                x = x.cuda()
                y = y.float().cuda()

                pred = model(x)
                totalTestLoss += lossFunc(pred, y)
            
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps

        train_loss.append(float(avgTrainLoss.cpu()))
        test_loss.append(float(avgTestLoss.cpu()))

        print("[INFO] EPOCH: {}/{}".format(epoch + 1, real_end))
        print("Train loss: {:.6f}, Test loss: {:.4f}, ".format(avgTrainLoss,  avgTestLoss))

    torch.save(model.state_dict(), 'unet_model_state_dict.pt')

    fig, ax1 = plt.subplots()
    lines1 = ax1.plot(range(1, real_end+1), train_loss, 'b-', label='train loss')
    ax1.set_ylabel('train loss')

    ax2 = ax1.twinx()
    lines2 = ax2.plot(range(1, real_end+1), test_loss, 'r-', label='test loss')
    ax2.set_ylabel('test loss')
    lines = lines1 + lines2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels)
    plt.xlabel('epochs')
    plt.savefig('learnig_curve.png')

def load_model():
    import time
    args = parse_args()
    
    if args.seed > 0:
        import random
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    test_dataset = SegmentationDataset(root=config.DATASET.ROOT,
                                       list_path=config.DATASET.TEST_SET,
                                       num_classes=config.DATASET.NUM_CLASSES)


    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    model = UNet()
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('unet_model_state_dict.pt'))
    sv_path = os.path.join('./', 'val_results')
    if not os.path.exists(sv_path):
        os.mkdir(sv_path)
    with torch.no_grad():
        model.eval()
        for i, (x, y, file_name) in enumerate(testloader):
            if i == 0:
                start = time.time()
            x = x.cuda()
            y = y.float().cuda()

            pred = model(x)
            test_dataset.save_pred(pred, sv_path, file_name)
        print(time.time()-start)
if __name__ == '__main__':
    # load_model()
    main()
import signal
import cv2
import numpy as np
from tqdm import tqdm
tqdm.monitor_interval = 0
import torch
from torch.autograd import Variable

from models import Model_1
from dataset import MobilityDataset


# Configuration
from config import CONFIG
NUM_EPOCHS  = CONFIG['num_epochs']
USE_CUDA    = CONFIG['cuda_available']
BATCH_SIZE  = CONFIG['batch_size']
NUM_WORKERS = CONFIG['cpu_count']



# Handle ctrl+c gracefully
signal.signal(signal.SIGINT, lambda signum, frame: exit(0))




def testing_epoch(net, test_loader, desc):
    ''' Testing epoch '''
    # Turn off dropout, batch norm, etc..
    net.eval()

    # Single pass through testation data
    correct = 0
    total = 0
    iterator = tqdm(test_loader, desc=desc, ncols=100, leave=False)
    for i, test_data in enumerate(iterator):
        # Get input data and labels
        inputs = Variable(test_data[0])
        labels = Variable(test_data[1])
        if USE_CUDA:
            inputs = inputs.cuda(async=True)
            labels = labels.cuda(async=True)

        # Forward pass
        outputs = net(inputs)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += test_data[-1].size(0)
        correct += np.sum(predicted.cpu().numpy() == test_data[-1].numpy())
        accuracy = 100.0 * correct / total
        iterator.set_postfix({"Accuracy": "{:.4f}".format(accuracy)})

    return accuracy





def training_epoch(net, optimizer, epoch, train_loader):
    ''' Training epoch '''
    # Set the network to training mode
    net.train()

    # Set the loss function
    weights = torch.Tensor([0.000731528895391368, 0.0004748338081671415, 0.5, 0.021739130434782608, 0.05263157894736842, 0.125, 0, 0, 0.3333333333333333, 0.25])
    print(weights)
    loss_func = torch.nn.CrossEntropyLoss(weight=weights)
    if USE_CUDA:
        loss_func = loss_func.cuda()

    # Single pass through training data
    total = 0
    correct = 0
    losses = []
    stat_dict = {"Epoch": epoch}
    iterator = tqdm(train_loader, postfix=stat_dict, ncols=100, desc="Training")
    for i, train_data in enumerate(iterator):
        # Get input data and labels
        inputs = Variable(train_data[0])
        labels = Variable(train_data[1])
        if USE_CUDA:
            inputs = inputs.cuda(async=True)
            labels = labels.cuda(async=True)

        # Forward pass, calculate loss, backward pass
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update loss and accuracy
        if (i+1)%10 == 0:
            _, predicted = torch.max(outputs.data, 1)
            total += train_data[-1].size(0)
            correct += np.sum(predicted.cpu().numpy() == train_data[-1].numpy())
            accuracy = 100.0 * correct / total
            losses.append(loss.data[0])
            stat_dict['Loss'] = "{:.5f}".format(np.mean(losses))
            stat_dict['Acc'] = "{:.4f}".format(accuracy)
            iterator.set_postfix(stat_dict)

    # Return the training accuracy
    return accuracy





def main():
    net = torch.nn.DataParallel(Model_1())
    if USE_CUDA:
        net = net.cuda()

    # Get dataloaders
    train_loader = torch.utils.data.DataLoader(MobilityDataset(),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(MobilityDataset(test_set=True),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Set up optimizer with auto-adjusting learning rate
    parameters = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train
    for epoch in range(NUM_EPOCHS):
        scheduler.step()
        train_acc = training_epoch(net, optimizer, epoch, train_loader)

        test_acc = testing_epoch(net, test_loader, desc="Testing")
        print('Epoch {:02} test-set accuracy: {:.1f}%'.format(epoch, test_acc))


if __name__ == '__main__':
    main()

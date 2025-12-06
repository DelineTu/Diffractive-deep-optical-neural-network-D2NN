import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
from model import Onn
from torch import optim
from label_generator import label_generator, eval_accuracy
import matplotlib.pyplot as plt
from loss import npcc_loss
import time
import os

def train(onn, criterion, optimizer, train_loader, val_loader, save_path, epoch_num=50, device='cuda:0'):
    label_set = label_generator()
    train_losses = []
    train_accies = []
    val_losses = []
    val_accies = []

    for epoch in range(epoch_num):
        train_loss = 0.0
        acc_sum = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            targets = label_set[labels].to(device)
            optimizer.zero_grad()
            outputs = onn(inputs)
            I = torch.abs(outputs) ** 2

            loss = criterion(I, targets)

            # Check if loss is NaN
            if torch.isnan(loss):
                print(f"Epoch {epoch + 1}, Batch {i + 1}: Loss is NaN.")
                continue

            loss.backward()
            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(onn.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_acc, label_hat = eval_accuracy(I, labels)
            acc_sum += train_acc.item()

            if (i + 1) % 32 == 0:  # print every 32 mini-batches
                train_log = f'epoch {epoch + 1} {i + 1}, train loss: {train_loss / 32: 5f}, train accuracy: {acc_sum / 32: 5f}'

                train_losses.append(train_loss / 32)
                train_accies.append(acc_sum / 32)
                train_loss = 0.0
                acc_sum = 0.0
                torch.save(onn, save_path + '/onn' + str(epoch + 1) + '.pt')
                with torch.no_grad():
                    val_loss, val_acc, I_val, labels_val = validation(onn, val_loader, criterion, device)
                    val_log = f'validation loss: {val_loss: 5f}, validation accuracy: {val_acc: 5f}'
                    val_losses.append(val_loss)
                    val_accies.append(val_acc)
                print(train_log, '\n', val_log)
                with open(save_path + '/log.txt', "a", encoding='utf-8') as f:
                    f.write(train_log + '\n')
                    f.write(val_log + '\n')
    return onn, train_losses, train_accies, val_losses, val_accies, I_val, labels_val

def validation(onn, val_loader, criterion, device='cuda:0'):
    val_loss_sum = 0.0
    val_acc_sum = 0.0
    label_set = label_generator()
    num_batches = 0

    with torch.no_grad():  # Disable gradient computation during validation
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            targets = label_set[labels].to(device)

            outputs = onn(inputs)
            I = torch.abs(outputs) ** 2

            # Check if loss is NaN
            val_loss = criterion(I, targets)
            if torch.isnan(val_loss):
                print(f"Validation Batch {i + 1}: Loss is NaN.")
                continue

            val_acc, _ = eval_accuracy(I, labels)

            val_loss_sum += val_loss.item()
            val_acc_sum += val_acc.item()
            num_batches += 1

        # Calculate average loss and accuracy
        avg_val_loss = val_loss_sum / num_batches
        avg_val_acc = val_acc_sum / num_batches

    return avg_val_loss, avg_val_acc, I, labels

if __name__ == '__main__':
    file_path = 'model'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Optical parameters
    c = 3e8 * 1e3  # speed of light
    f = 400e9  # 400GHz
    lambda0 = c / f  # wavelength
    L = 80  # DOE size
    z = [30, 30, 30, 30]

    # Digital parameters
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    M = 256  # sample nums
    batch_size = 128

    # Data transformations
    trans = transforms.Compose([Resize(M), ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load data
    mnist_train = torchvision.datasets.MNIST(
        root="data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.MNIST(
        root="data", train=False, transform=trans, download=False)
    train_set, val_set, _ = torch.utils.data.random_split(mnist_train, [4096, 512, 60000 - 4096 - 512])

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=True, num_workers=4)

    # Initialize model
    onn = Onn(M, L, lambda0, z).to(device)
    epoch_num = 50
    optimizer = optim.Adam(onn.parameters(), lr=1e-2)
    criterion = npcc_loss

    # Start training
    start_time = time.time()
    onn, train_losses, train_accies, val_losses, val_accies, I_val, labels_val = \
        train(onn, criterion, optimizer, train_loader, val_loader, file_path, epoch_num)

    end_time = time.time()
    print(f'Running time: {end_time - start_time: 5f}s')

    # Show results
    epochs = [k for k in range(1, epoch_num + 1)]
    plt.figure(dpi=300, figsize=(12, 4))
    plt.subplot(121)
    plt.plot(epochs, train_losses, '-o')
    plt.plot(epochs, val_losses, '-s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.subplot(122)
    plt.plot(epochs, train_accies, '-o')
    plt.plot(epochs, val_accies, '-s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.show()

    plt.figure(dpi=300, figsize=(8, 8))
    b = I_val.cpu().data.numpy()
    for k in range(9):
        plt.subplot(3, 3, k + 1)
        plt.imshow(b[k, :].squeeze(0)[64:256 - 64, 64:256 - 64], cmap='gray')
        plt.title('True Label: ' + str(labels_val[k].cpu().numpy()))
        plt.axis('off')
    plt.show()

    plt.figure(dpi=300, figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(onn.DiffLayer1.params[0].detach().cpu().numpy(), cmap='gray')
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(onn.DiffLayer2.params[0].detach().cpu().numpy(), cmap='gray')
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(onn.DiffLayer3.params[0].detach().cpu().numpy(), cmap='gray')
    plt.colorbar()
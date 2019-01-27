import os
import shutil

import matplotlib
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm, trange

from PIL import Image

matplotlib.use('Agg')

from matplotlib import pyplot as plt

from clr import CyclicLR


def train(model, loader, epoch, optimizer, criterion, device, dtype, batch_size, log_interval, scheduler):
    model.train()
    correct1, correct5 = 0, 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        if isinstance(scheduler, CyclicLR):
            scheduler.batch_step()
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        corr = correct(output, target, topk=(1, 5))
        correct1 += corr[0]
        correct5 += corr[1]

        if batch_idx % log_interval == 0:
            tqdm.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. '
                'Top-1 accuracy: {:.2f}%({:.2f}%). '
                'Top-5 accuracy: {:.2f}%({:.2f}%).'.format(epoch, batch_idx, len(loader),
                                                           100. * batch_idx / len(loader), loss.item(),
                                                           100. * corr[0] / batch_size,
                                                           100. * correct1 / (batch_size * (batch_idx + 1)),
                                                           100. * corr[1] / batch_size,
                                                           100. * correct5 / (batch_size * (batch_idx + 1))))
    return loss.item(), correct1 / len(loader.dataset), correct5 / len(loader.dataset)


def test(model, loader, criterion, device, dtype):
    model.eval()
    test_loss = 0
    correct1, correct5 = 0, 0

    #torch.set_num_threads(1)
    data = np.random.randn(1,3,224,224).astype(np.float32)
    data = torch.autograd.Variable(torch.from_numpy(data)).cpu()
    output = model(data)

    # times = []
    
    # import timeit
    # for _ in range(20):
    #     t1 = timeit.default_timer()
    #     output = model(data)
    #     t2 = timeit.default_timer()
    #     print(t1-t1, model.compression_layer.running_stats['time']-t1, t2-t1)
    #     times.append(model.compression_layer.running_stats['time']-t1)

    # print(len(times))
    # print(np.mean(np.asarray(times)))
    # print(np.median(np.asarray(times)))
                
    sizes = []

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        print('data', data.shape, data.dtype)
        for i in range(data.shape[0]):
            d = data[i,:,:,:].cpu().detach().numpy()
            d = np.transpose(d, (1,2,0))            
            dmin = np.min(d)
            dmax = np.max(d)
            
            img = ((255 * (d - dmin)) / (dmax - dmin)).astype(np.uint8)
            imgmin = np.min(img)
            imgmax = np.max(img)
            
            img = Image.fromarray(img)
            img.save('/home/jemmons/inputs/input{}.png'.format(i))
            break
            
        with torch.no_grad():
            output = model(data)

            sizes += model.module.compression_layer.get_compressed_sizes()
            print(sizes[-1])
            
            test_loss += criterion(output, target).item()  # sum up batch loss
            corr = correct(output, target, topk=(1, 5))
        correct1 += corr[0]
        correct5 += corr[1]

        print('top-1', correct1, batch_idx+1, loader.batch_size, correct1 / ((batch_idx+1) * loader.batch_size))

        break
        
    print(len(sizes))
    print(np.mean(np.asarray(sizes)))
    print(np.median(np.asarray(sizes)))
        
    test_loss /= len(loader)

    tqdm.write(
        '\nTest set: Average loss: {:.4f}, Top1: {}/{} ({:.2f}%), '
        'Top5: {}/{} ({:.2f}%)'.format(test_loss, int(correct1), len(loader.dataset),
                                       100. * correct1 / len(loader.dataset), int(correct5),
                                       len(loader.dataset), 100. * correct5 / len(loader.dataset)))
    return test_loss, correct1 / len(loader.dataset), correct5 / len(loader.dataset)


def correct(output, target, topk=(1,)):
    """Computes the correct@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        res.append(correct_k)
    return res


def save_checkpoint(state, is_best, filepath='./', filename='checkpoint.pth.tar'):
    save_path = os.path.join(filepath, filename)
    best_path = os.path.join(filepath, 'model_best.pth.tar')
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, best_path)


def find_bounds_clr(model, loader, optimizer, criterion, device, dtype, min_lr=8e-6, max_lr=8e-5, step_size=2000,
                    mode='triangular', save_path='.'):
    model.train()
    correct1, correct5 = 0, 0
    scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size=step_size, mode=mode)
    epoch_count = step_size // len(loader)  # Assuming step_size is multiple of batch per epoch
    accuracy = []
    for _ in trange(epoch_count):
        for batch_idx, (data, target) in enumerate(tqdm(loader)):
            if scheduler is not None:
                scheduler.batch_step()
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            corr = correct(output, target)
            accuracy.append(corr[0] / data.shape[0])

    lrs = np.linspace(min_lr, max_lr, step_size)
    plt.plot(lrs, accuracy)
    plt.show()
    plt.savefig(os.path.join(save_path, 'find_bounds_clr.png'))
    np.save(os.path.join(save_path, 'acc.npy'), accuracy)
    return

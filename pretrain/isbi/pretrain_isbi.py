import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from minitorch.util import main, TensorBoard, checkpoint_to_tiktorch
from minitorch.model import UNet2dGN
from minitorch.criteria import SorensenDice, WeightedLoss
from minitorch.data import Isbi2012
from minitorch.data import Compose, Noise, Rotate2d, Flip2d


def pretrain_isbi(net, device, data_set, num_workers=0):

    # TODO elastic augment
    trafos = Compose(Noise(std=0.05),
                     Rotate2d(),
                     Flip2d())

    train_set = Isbi2012(train=True, root=data_set, transform=trafos)
    val_set = Isbi2012(train=False, root=data_set)

    train_loader = DataLoader(train_set, batch_size=1,
                              shuffle=True, pin_memory=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=1,
                            num_workers=num_workers)

    loss = WeightedLoss(SorensenDice(use_as_loss=True), crop_target=True)
    metric = WeightedLoss(SorensenDice(), crop_target=True)

    tb = TensorBoard('./logs')
    optimizer = optim.Adam(net.parameters())

    n_epochs = 100
    main(net, device, train_loader, val_loader,
         loss_function=loss, optimizer=optimizer,
         val_metric=metric, tb_logger=tb,
         n_epochs=n_epochs, log_interval=1,
         save_folder='./checkpoints')


if __name__ == '__main__':
    # data_set = '/home/cpape/Work/data/isbi2012/isbi2012_train_volume.h5'
    data_set = '/g/kreshuk/data/isbi2012_challenge/isbi2012_train_volume.h5'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device", device)

    model_kwargs = {'in_channels': 1, 'out_channels': 1,
                    'initial_features': 64}
    net = UNet2dGN(**model_kwargs)
    # pretrain_isbi(net, device, data_set, num_workers=4)

    # TODO check that minimal increment is true
    checkpoint_to_tiktorch(UNet2dGN, model_kwargs,
                           './checkpoints', './ISBI2012_UNet_pretrained',
                           (1, 572, 572), (32, 32), device=device)

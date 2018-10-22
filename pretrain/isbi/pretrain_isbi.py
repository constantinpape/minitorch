import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from minitorch.util import main, TensorBoard, checkpoint_to_tiktorch
from minitorch.model import UNet2dGN
from minitorch.data import Isbi2012
from minitorch.criteria import SorensenDice, WeightedLoss


def pretrain_isbi(net, device, data_set, num_workers=0):
    # TODO transforms
    train_set = Isbi2012(train=True, root=data_set)
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

    main(net, device, train_loader, val_loader,
         loss_function=loss, optimizer=optimizer,
         val_metric=metric, tb_logger=tb,
         n_epochs=50, log_interval=1,
         save_folder='./checkpoints')


if __name__ == '__main__':
    data_set = '/home/cpape/Work/data/isbi2012/isbi2012_train_volume.h5'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_kwargs = {'in_channels': 1, 'out_channels': 1,
                    'initial_features': 64}
    net = UNet2dGN(**model_kwargs)
    pretrain_isbi(net, device, data_set)

    # TODO check that minimal increment is true
    checkpoint_to_tiktorch(UNet2dGN, model_kwargs,
                           './checkpoints', './ISBI2012_UNet_pretrained',
                           (1, 572, 572), (32, 32))

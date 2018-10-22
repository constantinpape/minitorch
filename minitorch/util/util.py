import os
import torch
import torch.nn as nn


# apply training for one epoch
def train(model, loader, device,
          optimizer, loss_function,
          epoch, log_interval=100,
          tb_logger=None):

    # set the model to train mode
    model.train()
    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model, calculate loss and run backwards pass
        prediction = model(x)
        loss = loss_function(prediction, y)
        loss.backward()
        optimizer.step()

        # TODO use logging instead
        # log to console
        if batch_id % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_id * len(x),
                  len(loader.dataset),
                  100. * batch_id / len(loader), loss.item()))

        # log to tensorboard if we have a logger
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            # call log train
            tb_logger.log_train(step, loss, x, y, prediction)


# run validation after training epoch
def validate(model, loader, device,
             loss_function, step,
             metric=None, tb_logger=None):
    # set model to eval mode
    model.eval()
    # running loss and metric values
    val_loss = 0
    val_metric = 0

    # disable gradients during validation
    with torch.no_grad():

        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            val_loss += loss_function(prediction, y).item()
            if metric is not None:
                val_metric += metric(prediction, y).item()

    # normalize loss and metric
    val_loss /= len(loader.dataset)
    if metric is not None:
        val_metric /= len(loader.dataset)

    if tb_logger is not None:
        tb_logger.log_val(step, val_loss, x, y, prediction,
                          metric=None if metric is None else val_metric)

    # TODO use logging instead
    print('\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n'.format(val_loss,
                                                                              val_metric))
    return None if metric is None else val_metric


def main(model, device,
         train_loader, val_loader,
         loss_function, optimizer,
         val_metric=None, tb_logger=None,
         **config):
    """
    """
    # read config:
    # number of epoches
    n_epochs = config.pop('n_epochs', 25)
    # logging interval during training
    log_interval = config.pop('log_interval', 100)
    # save folder for models and checkpoints
    save_folder = config.pop('save_folder', '.')

    # TODO use logging instead
    if config:
        for k, v in config.items():
            print("Parameter %s with value %s is not supported" % (str(k), str(v)))

    # send model, loss and metric to device
    model.to(device)
    if isinstance(loss_function, nn.Module):
        loss_function.to(device)
    if val_metric is not None and isinstance(val_metric, nn.Module):
        val_metric.to(device)

    best_score = 0
    steps_per_epoch = len(train_loader)
    for epoch in range(n_epochs):
        train(model, train_loader, device,
              optimizer, loss_function,
              epoch, log_interval=log_interval,
              tb_logger=tb_logger)
        step = n_epochs * steps_per_epoch
        metric_value = validate(model, val_loader, device,
                                loss_function, step,
                                metric=val_metric, tb_logger=tb_logger)
        # TODO implement lr decay based on val score
        # TODO implement proper checkpoiting and save val score and step and ...
        torch.save(model, os.path.join(save_folder, 'model.torch'))
        if metric_value is not None and metric_value > best_score:
            best_score = metric_value
            torch.save(model, os.path.join(save_folder, 'best_model.torch'))
    return model

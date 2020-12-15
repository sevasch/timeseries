import datetime
import os
import numpy as np
import torch
import wandb


def train_iterative(model, optimizer, loss_fn,
          loader_train, loader_val,
          n_iterations: int, n_steps_per_iteration:int,
          lr_scheduler=None,
          log_project: str='default_project', log_description: str='default_name', log_wandb: bool=False):
    
    # NAMING
    now = datetime.datetime.now()
    timestamp = str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2) + '_' \
                + str(now.hour).zfill(2) + str(now.minute).zfill(2)
    run_name = timestamp + '_' + log_description
    run_dir = os.path.join('saved_models', run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # LOGGING SETUP
    if log_wandb:
        wandb.login(key='f5788ddad7e204d7c6b5921d5259a7f0ab332c68')
        wandb.init(project=log_project, name=log_description)
        wandb.watch(model)

    # MIXED PRECISION SETUP
    scaler = torch.cuda.amp.GradScaler()

    # TRAINING
    def train_step(sample):
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            prediction = model(sample['input'].cuda())
            loss = loss_fn(prediction, sample['target'].cuda())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return loss.detach().cpu()

    def evaluate_step(sample):
        model.eval()
        with torch.cuda.amp.autocast(), torch.no_grad():
            prediction = model(sample['input'].cuda())
            loss = loss_fn(prediction, sample['target'].cuda())

        return loss.detach().cpu()

    iterator_train = iter(loader_train)
    sample_no = 0  # keeps track of sample in loader to reshuffle
    epoch = 0
    it_loss_train = []
    it_loss_val = []
    for iteration in range(n_iterations):
        # TRAINING
        loss_train = []
        for step in range(n_steps_per_iteration):
            sample = next(iterator_train)  # grab next batch
            loss_train.append(train_step(sample))

            # check if samples left in dataloader, otherwise reshuffle
            sample_no += 1
            if sample_no >= len(loader_train):
                sample_no = 0
                epoch += 1
                iterator_train = iter(loader_train)  # iter -> triggers __iter__() -> shuffles data
                print('reshuffle for epoch: ' + str(epoch))

        # VALIDATION
        loss_val = []
        for sample in loader_val:
            loss_val.append(evaluate_step(sample))

        print('iteration ' + str(iteration + 1) +  '/' + str(n_iterations) + ' finished, training loss: ' + str(np.asarray(loss_train).mean()) + ', evaluation loss: ' +  str(np.asarray(loss_val).mean()))

        # LEARNING RATE UPDATE
        if lr_scheduler:
            lr_scheduler.step()

        # LOGGING
        it_loss_train.append(np.asarray(loss_train).mean())
        it_loss_val.append(np.asarray(loss_val).mean())

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        if log_wandb:
            wandb.log({'loss_t': np.asarray(loss_train).mean(),
                       'loss_v': np.asarray(loss_val).mean(),
                       'learning_rate': lr}, commit=True)

        # SAVE MODEL CHECKPOINT
        if np.asarray(loss_val).mean() == min(it_loss_val):
            ckpt = {'model': model,
                    'model_state_dict': model.state_dict()}
            torch.save(ckpt, os.path.join(run_dir, 'best.pt'))#'iter' + str(iteration + 1).zfill(2) + ('_best.pt')))

    return it_loss_train, it_loss_val


def train_epochs(model, optimizer, loss_fn,
                 loader_train, loader_val,
                 n_epochs: int,
                 lr_scheduler=None,
                 log_project: str = 'default_project', log_description: str = 'default_name', log_wandb: bool=False):
    # NAMING
    now = datetime.datetime.now()
    timestamp = str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2) + '_' \
                + str(now.hour).zfill(2) + str(now.minute).zfill(2)
    run_name = timestamp + '_' + log_description
    run_dir = os.path.join('saved_models', run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # LOGGING SETUP
    if log_wandb:
        wandb.login(key='f5788ddad7e204d7c6b5921d5259a7f0ab332c68')
        wandb.init(project=log_project, name=log_description)
        wandb.watch(model)

    # MIXED PRECISION SETUP
    scaler = torch.cuda.amp.GradScaler()

    # TRAINING
    def train():
        model.train()
        losses = []
        for sample in loader_train:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                prediction = model(sample['input'].cuda())
                # print('train: ' + str(prediction[0].item()) +' ' +str(sample['target'][0].item()))
                loss = loss_fn(prediction, sample['target'].cuda())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.detach().cpu())
        return np.asarray(losses).mean()

    def evaluate():
        model.eval()
        losses = []
        for sample in loader_val:
            with torch.cuda.amp.autocast(), torch.no_grad():
                prediction = model(sample['input'].cuda())
                # print('val: ' + str(prediction[0].item()) +' ' +str(sample['target'][0].item()))
                loss = loss_fn(prediction, sample['target'].cuda())
            losses.append(loss.detach().cpu())
        return np.asarray(losses).mean()

    ep_loss_train = []
    ep_loss_val = []
    for epoch in range(n_epochs):
        ep_loss_train.append(loss_train := train())
        ep_loss_val.append(loss_val := evaluate())

        print('epoch ' + str(epoch + 1) + '/' + str(n_epochs) + ' finished, training loss: '
              + str(loss_train) + ', evaluation loss: ' + str(loss_val))

        # LEARNING RATE UPDATE
        if lr_scheduler:
            lr_scheduler.step()

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        if log_wandb:
            wandb.log({'loss_t': loss_train,
                       'loss_v': loss_val,
                       'learning_rate': lr}, commit=True)

        # SAVE MODEL CHECKPOINT
        if loss_val == min(ep_loss_val):
            ckpt = {'model': model,
                    'model_state_dict': model.state_dict()}
            torch.save(ckpt, os.path.join(run_dir, 'best.pt'))#'iter' + str(epoch + 1).zfill(2) + ('_best.pt')))

    return ep_loss_train, ep_loss_val
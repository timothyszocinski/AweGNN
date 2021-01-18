import os
import time
from scipy.stats import pearsonr
import numpy as np
import torch as pt
import torch.nn as nn
from torch import optim
from Layers.GGR import ESGGR

class AweGNN(nn.Module):

    def __init__(self, layer_list=[], dataset='solv', model_num='', eta_rng=None, kappa_rng=(2, 5), tau_rng=(0.5, 1.5), ker='lor',
                 opt='AMSGrad', betas=(0.9, 0.999)):

        # Allow methods from nn.Module
        super(AweGNN, self).__init__()

        # Initialize the Geometric Graph Representation layer and the input normalization layer
        self.GGR = ESGGR(eta_rng=eta_rng, kappa_rng=kappa_rng, ker=ker, tau_rng=tau_rng)
        self.scaling = nn.BatchNorm1d(self.GGR.num_feat)

        # Add the hidden layer to the module
        self.hidden_layers = nn.ModuleList(layer_list)

        # Find the number of features of the last hidden layer
        i = 1
        while True:
            print(i)
            try:
                num_neurons_last_layer = layer_list[-i].out_features
                break
            except:
                i += 1

        # Add an output
        self.output_layer = nn.Linear(num_neurons_last_layer, 1)

        # Initialize a chosen optimizer
        if opt == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), 0, betas=betas)
        if opt == 'AMSGrad':
            self.optimizer = optim.Adam(self.parameters(), 0, betas=betas, amsgrad=True)
        if opt == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), 0, momentum=betas[0], centered=False)
        if opt == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), 0, momentum=betas[0], nesterov=False)
        if opt == 'Adagrad':
            self.optimizer = optim.Adagrad(self.parameters(), 0)

        # Record intial options for saving
        self.kappa_rng = kappa_rng
        self.eta_rng = eta_rng
        self.tau_rng = tau_rng
        self.ker = ker
        self.opt = opt
        self.betas = betas
        self.dataset = dataset
        self.model_num = model_num

        # Initialize modes
        self.save_mode = False
        self.stats_mode = False

        # Other attributes for saving
        self.save_sched = [0]
        self.save_directory = os.getcwd()
        self.model_stats = None

        # Keep a running tally of the amount of time it takes to complete an epoch
        self.time_list = []

    def forward(self, x):
        x = self.GGR(x)
        x = self.scaling(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def test(self, x):
        self.eval()
        x = self(x)
        self.train()
        return x

    def display_stats(self, y_train, y_train_hat, y_test, y_test_hat, epoch):

        # Display the relevant statistics
        if len(self.time_list) != 0:
            print('Epoch %d completed in %.2f sec. Average completion time is %.2f sec'%(epoch, self.time_list[-1], sum(self.time_list)/(epoch)))
        print('-----')
        print('Train MAE:', pt.sum(pt.abs(y_train_hat - y_train) / y_train.shape[0]).item())
        print('Train RMSE:', pt.sqrt(pt.sum((y_train_hat - y_train)**2) / y_train.shape[0]).item())
        print('Train pearson:', pearsonr(y_train_hat.detach().cpu().view(-1), y_train.cpu().detach().view(-1))[0])
        print('')
        print('Test MAE:', pt.sum(pt.abs(y_test_hat - y_test) / y_test.shape[0]).item())
        print('Test RMSE:', pt.sqrt(pt.sum((y_test_hat - y_test)**2) / y_test.shape[0]).item())
        print('Test R2 score:', pearsonr(y_test_hat.cpu().detach().view(-1), y_test.cpu().detach().view(-1))[0]**2)
        print('----------------------------------')

    def save(self, y_train_hat, y_test_hat, epoch):

        # Get strings representing the hidden layers, betas, and learning schedule
        lay_str = '-'.join([str(w.shape[0]) for w in self.parameters() if len(w.shape) == 2])[:-3]
        lr_sched_str = str(self.lr_sched[0])+'-'+str(self.lr_sched[1])
        betas_str = str(self.betas[0])+'-'+str(self.betas[1])

        # Figure out tau or eta initialization
        if self.tau_rng != None:
            param_str = f'_tau{self.tau_rng[0]}-{self.tau_rng[1]}'
        else:
            param_str = f'_eta{self.eta_rng[0]}-{self.eta_rng[1]}'

        # Write the base filename
        stats_filename = (self.dataset+'_epochs'+str(self.total_epochs)+'_batch'+str(self.batch_size)+'_lay'+lay_str+
                        '_lr'+str(self.lr)+'_alpha'+str(self.alpha)+param_str+'_kappa'+str(self.kappa_rng[0])+'-'+
                        str(self.kappa_rng[1])+'_ker'+self.ker+'_opt'+self.opt+'_lrsch'+lr_sched_str+'_betas'+
                        betas_str+'_model'+self.model_num)

        # Initialize recording of the statistics for the first time
        if epoch == 0:
            self.model_stats = {'train_predictions':pt.zeros(len(y_train_hat), self.total_epochs+1),
                                'test_predictions':pt.zeros(len(y_test_hat), self.total_epochs+1),
                                'saved_etas':pt.zeros(self.GGR.num_grps, self.total_epochs+1),
                                'saved_kappas':pt.zeros(self.GGR.num_grps, self.total_epochs+1),
                                'model_state':{}}
        # Or change the existing statistics
        else:
            self.model_stats['train_predictions'][:, epoch] = y_train_hat.view(-1).detach().cpu()
            self.model_stats['test_predictions'][:, epoch] = y_test_hat.view(-1).detach().cpu()
            self.model_stats['saved_etas'][:, epoch] = self.GGR.eta.detach().cpu()
            self.model_stats['saved_kappas'][:, epoch] = self.GGR.kappa.detach().cpu()

        # Record statistics dictionary at special times (default: beginning and end)
        if epoch in self.save_sched:
            stats_dict = {'model_state_dict':self.state_dict(),
                          'optimizer_state_dict':self.optimizer.state_dict(),
                          'save_schedule':self.save_sched,
                          'scheduler':self.scheduler,
                          'time_list':self.time_list}

            self.model_stats['model_state'][epoch] = stats_dict

        # Save the model statitics dictionary into the specified save directory (default: current directory)
        pt.save(self.model_stats, f'{self.save_directory}/{stats_filename}')

    def fit(self, D_train, y_train, D_test, y_test, lr=0.1, start=1, total_epochs=2000, batch_size=100, alpha=0, lr_sched=(1, 0.999)):

        # Set the variables for potential saving access
        self.lr = lr
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.lr_sched = lr_sched
        self.save_sched.append(total_epochs)

        # Set the loss function
        loss_function = nn.MSELoss()

        # Sets the parameter learning rates
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['initial_lr'] = lr

        # Set the learning rate scheduler
        if lr_sched != None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, lr_sched[0], lr_sched[1])
        else:
            self.scheduler = None

        # Split the data into distance data and sparse index tracker, and record number of samples for easy access
        D, spar_ind = D_train
        num_samples = len(D)

        # Save or display stats
        if start == 1 and (self.save_mode == True or self.stats_mode == True):

            # Run through full train and test data
            y_train_hat = self.test(D_train)
            y_test_hat = self.test(D_test)

            # Save and/or display stats
            if self.save_mode == True:
                self.save(y_train_hat, y_test_hat, epoch=0)
            if self.stats_mode == True:
                self.display_stats(y_train_hat, y_train, y_test_hat, y_test, epoch=0)

        # Begin the fitting at the given starting epoch
        for epoch in range(start, total_epochs+1):

            # Start the timer for the epoch
            start_time = time.time()

            print('epoch', epoch)
            print('----------------------------------')

            # Mix the indices
            idx = np.random.choice(num_samples, num_samples, replace=False)

            # Go through each mini-batch
            for i in range(0, num_samples, batch_size):

                print('batch', i//batch_size + 1)

                # Handle end-of-batch sizes
                if i+batch_size >= num_samples:
                    real_batch_size = num_samples - i
                else:
                    real_batch_size = batch_size

                # Get the indices for the mini-batch
                batch_idx = list(idx[i:i+real_batch_size])

                # Feed in the data according to the batch indices, then calculate loss
                y_hat = self(([D[i] for i in batch_idx], spar_ind[batch_idx]))
                loss = loss_function(y_hat, y_train[batch_idx])

                # Regularization
                for w in self.parameters():
                    if len(w.shape) == 2:
                        loss += alpha*pt.sum(w**2)

                print('train loss:', loss.item())
                print('--------------')

                # Back propagation and reset gradient
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Update the scheduler if there is one
            if lr_sched != None:
                self.scheduler.step()

            # Add time to list
            self.time_list.append(time.time()-start_time)

            # Save or display stats
            if self.save_mode == True or self.stats_mode == True:

                # Run through full train and test data
                y_train_hat = self.test(D_train)
                y_test_hat = self.test(D_test)

                # Save and/or display stats
                if self.save_mode == True:
                    self.save(y_train_hat, y_test_hat, epoch=epoch)
                if self.stats_mode == True:
                    self.display_stats(y_train_hat, y_train, y_test_hat, y_test, epoch=epoch)

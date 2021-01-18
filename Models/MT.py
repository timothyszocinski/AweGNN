import os
import time
from itertools import chain
from scipy.stats import pearsonr
import numpy as np
import torch as pt
import torch.nn as nn
from torch import optim
from Layers.GGR import ESGGR

class MTAweGNN(nn.Module):

    def __init__(self, layer_list, eta_rng=None, kappa_rng=(2, 5), tau_rng=(0.5, 1.5), ker='lor', datasets=['LC50DM','LC50','IGC50','LD50'],
                 model_num='', opt='AMSGrad', betas=(0.9, 0.999)):

        # Allow methods from nn.Module
        super(MTAweGNN, self).__init__()

        # Initialize the geometric graph representation layer and the input normalization layer
        self.GGR = ESGGR(eta_rng=eta_rng, kappa_rng=kappa_rng, ker=ker, tau_rng=tau_rng)
        self.scaling = nn.BatchNorm1d(self.GGR.num_feat)

        # Add the hidden layers
        self.layers = nn.ModuleList(layer_list)

        # Find the number of features of the last hidden layer
        i = 1
        while True:
            print(i)
            try:
                num_neurons_last_layer = layer_list[-i].out_features
                break
            except:
                i += 1

        # Add the output layers
        self.out = nn.ModuleList([nn.Linear(num_neurons_last_layer, 1) for _ in datasets])

        # Initialize the chosen optimizer
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
        self.datasets = datasets
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

    def forward(self, data_list):
        # Record a staggered number of samples in each data set for later
        n = [0]
        for i in range(len(data_list)):
            n.append(n[-1] + len(data_list[i][0]))

        # Combine the distances and sparse indices from each data set separately
        dist = list(chain(*[data[0] for data in data_list]))
        spar_ind = np.concatenate([data[1] for data in data_list], axis=0)

        # Combine into one piece of data for input
        X = (dist, spar_ind)

        # Run through the input layer
        X = self.GGR(X)
        X = self.scaling(X)

        # Run through the hidden layers
        for layer in self.layers:
            X = layer(X)

        # Split the samples up, then run through the separate outputs
        hidden_out = [X[n[i]:n[i+1], :] for i in range(len(data_list))]
        return tuple(self.out[i](hidden_out[i]) for i in range(len(data_list)))

    def test(self, x_list):
        self.eval()
        y_list = self(x_list)
        self.train()
        return y_list

    def display_stats(self, y_train_list, y_train_hat_list, y_test_list, y_test_hat_list, epoch):

        # Display the relevant statistics
        if len(self.time_list) != 0:
            print('Epoch %d completed in %.2f sec. Average completion time is %.2f sec'%(epoch, self.time_list[-1], sum(self.time_list)/(epoch)))
        for i, data_name in enumerate(self.datasets):
            print(data_name)
            print('-----')
            print('Train MAE:', pt.sum(pt.abs(y_train_hat_list[i] - y_train_list[i]) / y_train_list[i].shape[0]).item())
            print('Train RMSE:', pt.sqrt(pt.sum((y_train_hat_list[i] - y_train_list[i])**2) / y_train_list[i].shape[0]).item())
            print('Train pearson:', pearsonr(y_train_hat_list[i].detach().cpu().view(-1), y_train_list[i].cpu().detach().view(-1))[0])
            print('')
            print('Test MAE:', pt.sum(pt.abs(y_test_hat_list[i] - y_test_list[i]) / y_test_list[i].shape[0]).item())
            print('Test RMSE:', pt.sqrt(pt.sum((y_test_hat_list[i] - y_test_list[i])**2) / y_test_list[i].shape[0]).item())
            print('Test R2 score:', pearsonr(y_test_hat_list[i].cpu().detach().view(-1), y_test_list[i].cpu().detach().view(-1))[0]**2)
            print('----------------------------------')

    def save(self, y_train_hat_list, y_test_hat_list, epoch):

        # Get strings representing the hidden layers, betas, and learning schedule
        lay_str = '-'.join([str(w.shape[0]) for w in self.parameters() if len(w.shape) == 2])[:-2*len(self.datasets)]
        lr_sched_str = str(self.lr_sched[0])+'-'+str(self.lr_sched[1])
        betas_str = str(self.betas[0])+'-'+str(self.betas[1])

        # Figure out tau or eta initialization
        if self.tau_rng != None:
            param_str = f'_tau{self.tau_rng[0]}-{self.tau_rng[1]}'
        else:
            param_str = f'_eta{self.eta_rng[0]}-{self.eta_rng[1]}'

        # Write the base filename
        stats_filename = ('MT_'+'_'.join(self.datasets)+'_epochs'+str(self.total_epochs)+'_batch'+str(self.num_batches)+
                          '_lay'+lay_str+'_lr'+str(self.lr)+'_alpha'+str(self.alpha)+param_str+'_kappa'+str(self.kappa_rng[0])+
                          '-'+str(self.kappa_rng[1])+'_ker'+self.ker+'_opt'+self.opt+'_lrsch'+lr_sched_str+'_betas'+
                          betas_str+'_model'+self.model_num)

        # Initialize recording of the statistics for the first time
        if epoch == 0:
            self.model_stats = {'saved_etas':pt.zeros(self.GGR.num_grps, self.total_epochs+1),
                                'saved_kappas':pt.zeros(self.GGR.num_grps, self.total_epochs+1),
                                'model_state':{}}
            for i, data_name in enumerate(self.datasets):
                self.model_stats[f'{data_name}_train_predictions'] = pt.zeros(len(y_train_hat_list[i]), self.total_epochs+1)
                self.model_stats[f'{data_name}_test_predictions'] = pt.zeros(len(y_test_hat_list[i]), self.total_epochs+1)
        # Or change the existing statistics
        else:
            self.model_stats['saved_etas'][:, epoch] = self.GGR.eta.detach().cpu()
            self.model_stats['saved_kappas'][:, epoch] = self.GGR.kappa.detach().cpu()
            for i, data_name in enumerate(self.datasets):
                self.model_stats[f'{data_name}_train_predictions'][:, epoch] = y_train_hat_list[i].view(-1).detach().cpu()
                self.model_stats[f'{data_name}_test_predictions'][:, epoch] = y_test_hat_list[i].view(-1).detach().cpu()

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

    def fit(self, D_train_list, y_train_list, D_test_list, y_test_list, lr=0.001, total_epochs=2000, num_batches=20, alpha=0, start=1,
            lr_sched=None):

        # Set the variables for potential saving access
        self.lr = lr
        self.total_epochs = total_epochs
        self.num_batches = num_batches
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

        # Gather the distance data, sparse indices, number of samples, and batch_sizes for each data set separately
        D_list = [pair[0] for pair in D_train_list]
        spar_ind_list = [pair[1] for pair in D_train_list]
        num_samp_list = [len(D) for D in D_list]
        batch_size_list = [num_samp//num_batches for num_samp in num_samp_list]

        # Preliminarily save or display stats
        if start == 1 and (self.save_mode == True or self.stats_mode == True):

            # Run through full train and test data
            y_train_hat_list = self.test(D_train_list)
            y_test_hat_list = self.test(D_test_list)

            # Save and/or display stats
            if self.save_mode == True:
                self.save(y_train_hat_list, y_test_hat_list, epoch=0)
            if self.stats_mode == True:
                self.display_stats(y_train_hat_list, y_train_list, y_test_hat_list, y_test_list, epoch=0)

        # Begin the fitting at the given starting epoch
        for epoch in range(start, total_epochs+1):

            # Start the timer for the epoch
            start_time = time.time()

            print('epoch', epoch)
            print('----------------------------------')

            # Mix the indices
            idx_list = [np.random.choice(num_samp, num_samp, replace=False) for num_samp in num_samp_list]

            for param_group in self.optimizer.param_groups: # sets the parameter learning rates
                print(param_group['lr'])

            print('epoch', epoch)
            print('----------------------------------')

            # Go through each mini-batch
            for batch_num in range(1, num_batches+1):

                print('batch', batch_num)

                # Get the batch indices for each data set
                batch_idx_list = []
                for i, batch_size in enumerate(batch_size_list):

                    start = (batch_num - 1) * batch_size  # Get the starting index
                    if batch_num == num_batches:  # Go to the end of the list for the last batch
                        end = num_samp_list[i]
                    else:
                        end = batch_num * batch_size
                    batch_idx_list.append(list(idx_list[i][start:end]))

                # Get the batches for each data set
                batch_list = []
                for i in range(len(D_list)):
                    batch_list.append(([D_list[i][ind] for ind in batch_idx_list[i]], spar_ind_list[i][batch_idx_list[i]]))

                # Run batches through to get all the outputs
                y_hats = self(batch_list)

                # Get the average of the losses
                loss = sum([loss_function(y_hats[i], y_train_list[i][batch_idx_list[i]]) for i in range(len(D_list))]) / len(D_list)

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

            # Preliminarily save or display stats
            if self.save_mode == True or self.stats_mode == True:

                # Run through full train and test data
                y_train_hat_list = self.test(D_train_list)
                y_test_hat_list = self.test(D_test_list)

                # Save and/or display stats
                if self.save_mode == True:
                    self.save(y_train_hat_list, y_test_hat_list, epoch=epoch)
                if self.stats_mode == True:
                    self.display_stats(y_train_hat_list, y_train_list, y_test_hat_list, y_test_list, epoch=epoch)

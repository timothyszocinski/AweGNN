import os
import sys
import pickle

sys.path.append('Layers')
sys.path.append('Layers/Functions')

import torch as pt
import torch.nn as nn
import numpy as np

from Models.ST import AweGNN
from Models.MT import MTAweGNN


# Dictionary that corresponds to entries in the command line that is used to set the variables for running the algorithm.
# One dash means use the next command as a string and two dashes means to change the setting to true.
par = {'-layers':'400-400-20-20', '-dataset':'IGC50', '-device':'cpu', '--save':False, '-total_epochs':'2000', '-lr':'0.1',
       '-batch':'100', '-eta_rng':'None', '-kappa_rng':'5-8', '-ker':'lor', '-run':'', '-alpha':'0', '-tau_rng':'0.5-1.5',
       '--display_stats':False, '-start':'1', '-save_sched':'None', '-opt':'AMSGrad', '-lr_sched':'1-0.999',
       '-betas':'0.9-0.999', '-save_directory':'current'}

# Extract information from the entries in the command line
for i in range(1, len(sys.argv)):
    if '--' == sys.argv[i][:2]:
        par[sys.argv[i]] = True
    elif '-' == sys.argv[i][:1]:
        par[sys.argv[i]] = sys.argv[i+1]

# Set the default dtype here
pt.set_default_dtype(pt.float64)


def main():

    # Get the actual dataset name, i.e. w/o the multi_task prefix if present
    data_name = par['-dataset'].split('_')[-1]

    # Load based on if MT or ST model
    if 'MT' in par['-dataset']:
        D_train_list, y_train_list, D_test_list, y_test_list = load_multi_task_data(par['-dataset'].split('_')[1:], par['-device'])
    else:
        D_train, y_train, D_test, y_test = load_dataset(data_name, par['-device'])

    # Manipulate certain user arguments to input into models
    lay_data = [int(lay) for lay in par['-layers'].split('-')]
    betas = tuple([float(b) for b in par['-betas'].split('-')])
    eta_rng = tuple([float(num) for num in par['-eta_rng'].split('-')]) if par['-eta_rng'] != 'None' else None
    tau_rng = tuple([float(num) for num in par['-tau_rng'].split('-')]) if par['-tau_rng'] != 'None' else None
    kappa_rng = tuple([float(num) for num in par['-kappa_rng'].split('-')])
    save_schedule = [int(epoch_str) for epoch_str in par['-save_sched'].split('-') if epoch_str != 'None']
    lr_sched = (int(par['-lr_sched'].split('-')[0]), float(par['-lr_sched'].split('-')[1])) if par['-lr_sched'] != 'None' else None

    # Get the layer list for input
    layer_list = [nn.Linear(400, lay_data[0]), nn.BatchNorm1d(lay_data[0]), nn.ReLU()]
    for i in range(1, len(lay_data)):
        layer_list.append(nn.Linear(lay_data[i-1], lay_data[i]))
        layer_list.append(nn.BatchNorm1d(lay_data[i]))
        layer_list.append(nn.ReLU())

    # Initialize your model
    if 'MT' in par['-dataset']:
        model = MTAweGNN(layer_list, eta_rng=eta_rng, kappa_rng=kappa_rng, tau_rng=tau_rng, ker=par['-ker'],
                            datasets=par['-dataset'].split('_')[1:], model_num=par['-run'], opt=par['-opt'],
                            betas=betas)
    else:
        model = AweGNN(layer_list, eta_rng=eta_rng, kappa_rng=kappa_rng, tau_rng=tau_rng, ker=par['-ker'],
                            dataset=par['-dataset'], model_num=par['-run'], opt=par['-opt'], betas=betas)

    # Set the model to cuda if the option was chosen
    if par['-device'] != 'cpu':
        model.cuda()
    print(model)

    # Set display and save options
    model.stats_mode = True if par['--display_stats'] else False
    model.save_mode = True if par['--save'] else False
    model.save_sched = model.save_sched + save_schedule
    if par['-save_directory'] != 'current':
        model.save_directory = par['-save_directory']

    # Fit the model
    if 'MT' in par['-dataset']:
        labels = par['-dataset'].split('_')[1:]
        model.fit(D_train_list, y_train_list, D_test_list, y_test_list, lr=float(par['-lr']),
                total_epochs=int(par['-total_epochs']), num_batches=int(par['-batch']), alpha=float(par['-alpha']),
                start=int(par['-start']), lr_sched=lr_sched)
    else:
        model.fit(D_train, y_train, D_test, y_test, lr=float(par['-lr']), total_epochs=int(par['-total_epochs']),
                batch_size=int(par['-batch']), alpha=float(par['-alpha']), start=int(par['-start']),
                lr_sched=lr_sched)

    # Print training time stats
    total_seconds = sum(model.time_list)
    hours, remainder = total_seconds // 3600, total_seconds % 3600
    minutes, seconds = remainder // 60, remainder % 60
    print('Model trained in %d hours %d minutes and %d seconds' % (hours, minutes, seconds))


# Loading functions

def load_pickle_data(filename):
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def load_dataset(data_name, device_name='cpu'):
    train_data = load_pickle_data('Datasets/%s/%s_train_data'%(data_name, data_name))
    y_train = pt.tensor(np.loadtxt('Datasets/%s/%s_train_target.csv'%(data_name, data_name), ndmin=2), device=pt.device(device_name))
    test_data = load_pickle_data('Datasets/%s/%s_test_data'%(data_name, data_name))
    y_test = pt.tensor(np.loadtxt('Datasets/%s/%s_test_target.csv'%(data_name, data_name), ndmin=2), device=pt.device(device_name))
    return train_data, y_train, test_data, y_test

def load_multi_task_data(data_name_list, device_name='cpu'):
    train_data_list, y_train_list, test_data_list, y_test_list = [], [], [], []
    for data_name in data_name_list:
        print('-----Loading '+data_name+' Data-----')
        train_data, y_train, test_data, y_test = load_dataset(data_name, device_name)
        train_data_list.append(train_data)
        y_train_list.append(y_train)
        test_data_list.append(test_data)
        y_test_list.append(y_test)
    return train_data_list, y_train_list, test_data_list, y_test_list


main()

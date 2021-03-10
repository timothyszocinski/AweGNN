# AweGNN
Auto-parametrized weighted element-specific Graph Neural Network


# Description
A neural network that can be applied to molecular data sets that 
incorporates a feature generation layer that is updated throughout 
the training that is based on applying kernel-based functions with 
tunable parameters to element-specific interactions within or 
between molecular structures.

# Installation
Add a path to the directory containing the main.py file.

# Usage
When inside the main directory, you can run python main.py, and it 
will run with all of the default settings. To use different options, 
you must add additional arguments:

-layers [hidden layer sequence]; ex. -layers 400-400-20-20 (network 
has a hidden layer with 400 neurons followed by another hidden layer 
with 400 neurons, followed by two more layers of 20 neurons each; all 
with the appropriate input and output layers automatically introduced)

-dataset [dataset name]; i.e., IGC50, LD50, solv (for solvation), etc. or
'MT_' followed by datasets separated by underscores (for multi-task model), 
i.e. MT_LC50DM_LC50_IGC50_LD50 for a multi-task network including all of 
the four toxicity data sets.

--save; Tells the model to save the model statistics in a sepcified manner.

-total_epochs [number of epochs]; Total number of epochs of training.

-lr [learning rate]

-batch [batch size (for single-task) or number of batches (for multi-task)]

-eta_rng [sequence of min-max]; i.e., -eta_rng 3-8
initializes the eta parameter for each element-specific group from a 
uniform distribution of the interval [3, 8]. If -eta_rng None is used, 
then the -tau_rng option is considered (both should not be none!)

-kappa_rng [sequence of min-max]; as in eta_rng above, 
except that there is no None option.

-ker [choice of kernel]; i.e., -ker exp (for the exponential kernel) or 
-ker lor (for the Lorentz kernel)

-run [run number or model number]; Used for labeling saved models

-alpha [regularization constant]

-tau_rng [sequence of min-max]; As -eta_rng option, but 
only one value of tau is chosen for all element-specific groups.

--display_stats; Calculates the full training and test statistics at 
each epoch of training

-save_sched [sequence of epochs]; ex. -save_sched 200-400 will save 
the full model's parameters after training for 200 epochs and after 
400 epochs, if --save option is used. The model before training and 
after the final epoch of training will automatically be saved any 
time the --save option is used. If -save_sched None is used, then 
only the default initial and final saves will be made.

-opt [choice of optimizer]; Optimizer choices are 'Adam', 'AMSGrad', 
'RMSProp', 'SGD', and 'Adagrad'

-lr_sched [sequence of (epoch gap)-(lr decay)]; i.e., -lr_sched 1-0.999 
will decay the learning rate by a factor of 0.999 after every epoch of 
training.

-betas [sequence of beta1-beta2]; Involves beta values for certain 
optimizers, and for SGD with momentum or RMSProp momentum values, we 
only use beta1.

-save_directory [directory]; This determines the directory in which 
you would like the dictionary of saved parameters and statistics of 
your model to be located if the --save option is used.

--validation; Uses the last 10% of the samples of the training set 
as the test set and the remaining training samples as the training 
set.




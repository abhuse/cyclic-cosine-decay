# Pytorch Cyclic Cosine Decay Learning Rate Scheduler

A learning rate scheduler for Pytorch. This implements 2 modes:
* Geometrically increasing cycle restart intervals, as demonstrated by: 
[\[Loshchilov & Hutter 2017\]: SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
* Fixed cycle restart intervals, as seen in: 
[\[Athiwaratkun et al 2019\]: There Are Many Consistent Explanations of Unlabeled Data: 
Why You Should Average](https://arxiv.org/abs/1806.05594)

### Parameters
* ***optimizer*** *(Optimizer)* - Wrapped optimizer.
* ***init_interval*** *(int)* -  Initial decay cycle interval.
* ***min_lr*** *(float or iterable of floats)* - Minimal learning rate.
* ***restart_multiplier*** *(float)* - Multiplication coefficient for 
 increasing cycle intervals, if this parameter is set, *restart_interval* 
 must be *None*.
* ***restart_interval*** *(int)* - Restart interval for fixed 
cycle intervals, if this parameter is set, *restart_multiplier* 
must be *None*.
* ***restart_lr*** *(float or iterable of floats)* - Optional, 
the learning rate at cycle restarts,
if not provided, initial learning rate will be used.
* ***last_epoch*** *(int)* - Last epoch.


The learning rates are decayed for ***init_interval*** epochs from 
initial values passed to *optimizer* to **min_lr** using 
cosine decay function from *\[Loshchilov & Hutter 2017]*. 
The cycle is then restarted:
* If ***restart_multiplier*** is provided, the cycle interval at 
 each restart is multiplied by given parameter, this corresponds
 to *\[Loshchilov & Hutter 2017]* implementation.
* If ***restart_interval*** is provided, all subsequent cycles
 have fixed interval equal to provided value, this mode was 
used in *\[Athiwaratkun et al 2019\]*.
 
Note that ***restart_multiplier*** and ***restart_interval***
are mutually exclusive, i.e. if one is provided, 
the other must be *None*.
If both are *None*, learning rates will remain to be equal to 
*min_lr* for remaining epochs.

When cycle restarts, learning rates are reset to the values provided
to *optimizer* unless ***restart_lr*** is provided, 
in which case learning rates will be set to ***restart_lr***.

***min_lr*** and ***restart_lr*** can be float or iterable of floats
in case multiple parameter groups are provided to the *optimizer*. 
In latter, *len(min_lr)* and *len(restart_lr)* must be 
equal to *len(optimizer.param_groups)*.


### Usage example
Usage examples are provided in [this notebook](example.ipynb).
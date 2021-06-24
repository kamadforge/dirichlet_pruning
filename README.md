###To run Lenet compression;

#### 1. First train the model

```
python lenet_network_pruning.py --early_stopping 30
```

Then provide the name of the trained mode

#### 2. To prune:

Example (choose your own model from step 1):
```
python lenet_network_pruning.py --resume=True --prune_bool=True --path mnist_trainval0.8_epo8_acc98.70
```

#### 3. To prune and retrain:
```
python lenet_network_pruning.py --resume=True --prune_bool=True --path mnist_trainval0.8_epo8_acc98.70 --retrain=True
```
or
```
python lenet_network_pruning.py --resume=True --prune_bool=True --path mnist_trainval0.8_epo8_acc98.70 --switch_comb load --retrain=True
```

Notes:

The default early-stopping value is 500, for trying the code choose a lower value.
`--early_stopping 30`
Once you run the switch training once you can load the parameters
`--switch_comb load`
Choose an architecture to prune:
`--arch 8,9,36,17`
Select between mnist and fashionmnist
`--dataset fashionmnist`




### WideResNet

Along with other settings, the default value for num_epoch=200 can be changed in the config.py


#### 1. To train the base network run:

python main.py


#### 2. Then compute the switch vectors

python main_switch.py

It is enough to run it for even 1 iteration, 3-5 are recommended.


#### 3. To prune and retrain the previously trained model:

python main_prune.py --arch 75,85,80,80,159,159,154,159,315,315,314,316


### VGG

Please find all the available flags in the `VGG/vgg_load.py`

For basic run:

``cd VGG``

#### 1. To train the base network run:

```python vgg_load.py --resume 0 --prune_bool 0 --retrain_bool 0```



#### 2b. Select a pruning method

For example, select fisher pruning:

``python vgg_load.py --resume 0 --prune_bool 0 --retrain_bool 0 --method fisher``










## Starting training

pass in the config file as with option `--hyper`:

```python prediction_train_sawyer.py --hyper ../../tensorflow_data/sawyer/<folder_with_configfile>/conf.py```

to resume training from checkpoint add the option:


 ```--pretrained ../../tensorflow_data/sawyer/<folder_with_configfile>/modeldata/model<iter_num>```

for example
 ```--pretrained ../../tensorflow_data/sawyer/cdna/modeldata/model20002```

 you can specify which gpu to use by setting:

 ```--device <i_gpu>```


## Visualizing Video

for visualizing video prediction with action sequences and start-images from the test set add the option:

`--visualize model<iter_num>`

for visualizing motions in different direction and the predicted probability distributions of the designated pixel add:

`--diffmotions`




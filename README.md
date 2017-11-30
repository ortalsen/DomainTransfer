# DomainTransfer
Using AutoDIAL framework for training US-CT end to end network working on real US images.
Based on the paper: Maria Carlucci, Fabio, et al. "AutoDIAL: Automatic DomaIn Alignment Layers." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017
Datasets should be stored in the following hierarchy: 
datasets--->Name_of_data_set--->source--->samples
datasets--->Name_of_data_set--->source--->labels
datasets--->Name_of_data_set--->target--->samples
For training run from terminal: python main.py --phase=train --dataset_dir=Name_of_data_set (...any other parameter)
For test run from terminal: python main.py --pashe=test --dataset_dir=Name_of_data_set --domain=target
For displaying results in tensorboard run from terminal: tensorboard --logdir=./logs

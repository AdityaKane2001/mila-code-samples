# RegNets model and training script

This is the model definition and script for training [RegNets](https://arxiv.org/abs/2003.13678) on ImageNet-1k.

The trained models were merged to [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/regnet), and later to [KerasCV](https://github.com/keras-team/keras-cv/pull/739).

File descriptions:
1. [check_models.py](https://github.com/AdityaKane2001/mila-code-samples/blob/main/regnets/check_models.py): accuracy verification script
2. [dataset.py](https://github.com/AdityaKane2001/mila-code-samples/blob/main/regnets/dataset.py): dataset class including preprocessing pipeline
3. [evaluate.py](https://github.com/AdityaKane2001/mila-code-samples/blob/main/regnets/evaluate.py#L8): model to calculate accuracies on ImageNet validation split
4. [regnet.py](https://github.com/AdityaKane2001/mila-code-samples/blob/main/regnets/regnet.py): model definition file (copied over from keras/applications/regnet.py)
5. [train.py](https://github.com/AdityaKane2001/mila-code-samples/blob/main/regnets/train.py): training script for all models
6. [utils.py](https://github.com/AdityaKane2001/mila-code-samples/blob/main/regnets/utils.py): utilities including schedulers, loggers and callbacks. 


# hydra-small-project-extended
Hydra with structured config
Small Hydra project from https://www.udemy.com/course/sustainable-and-scalable-machine-learning-project-development

# Run
```
python -m venv hydra
python -m pip install hydra-core pytorch-lightning torch torchvision tensorboard pydantic
python .\train.py
```

# Run on vm.massedcompute.com

```
Ubuntu@0017-dsm-prxmx30051:~/hydra-small-project-extended$ python3 train.py 
Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to /home/Ubuntu/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 97.8M/97.8M [00:00<00:00, 165MB/s]
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
You are using a CUDA device ('NVIDIA RTX A6000') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
Missing logger folder: tb_logs/cifar10
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar10/cifar-10-python.tar.gz
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 170498071/170498071 [00:01<00:00, 89537022.90it/s]
Extracting ./data/cifar10/cifar-10-python.tar.gz to ./data/cifar10
Files already downloaded and verified
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name                | Type               | Params
-----------------------------------------------------------
0 | model               | CIFAR10Model       | 23.5 M
1 | loss_function       | CrossEntropyLoss   | 0     
2 | train_accuracy      | MulticlassAccuracy | 0     
3 | validaiton_accuracy | MulticlassAccuracy | 0     
4 | test_accuracy       | MulticlassAccuracy | 0     
-----------------------------------------------------------
23.5 M    Trainable params
0         Non-trainable params
23.5 M    Total params
94.114    Total estimated model params size (MB)
Sanity Checking: |                                                                                                              | 0/? [00:00<?, ?it/s]/home/Ubuntu/.local/lib/python3.10/site-packages/torch/utils/data/dataloader.py:558: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 6, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(_create_warning_msg(
Epoch 0:   0%|                                                                                                                | 0/703 [00:00<?, ?it/s]/home/Ubuntu/.local/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
Epoch 9: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 703/703 [00:28<00:00, 24.70it/s, v_num=0]`Trainer.fit` stopped: `max_epochs=10` reached.                                                                                                       
Epoch 9: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 703/703 [00:28<00:00, 24.38it/s, v_num=0]
/home/Ubuntu/.local/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/checkpoint_connector.py:145: `.test(ckpt_path=None)` was called without a model. The best model of the previous `fit` call will be used. You can pass `.test(ckpt_path='best')` to use the best model or `.test(ckpt_path='last')` to use the last model. If you pass a value, this warning will be silenced.
Files already downloaded and verified
Files already downloaded and verified
Restoring states from the checkpoint path at /home/Ubuntu/hydra-small-project-extended/checkpoints/cifar10-epoch=09-val_loss=0.00.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /home/Ubuntu/hydra-small-project-extended/checkpoints/cifar10-epoch=09-val_loss=0.00.ckpt
Testing DataLoader 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 156/156 [00:02<00:00, 65.70it/s]
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      test_accuracy         0.43940305709838867
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Ubuntu@0017-dsm-prxmx30051:~/hydra-small-project-extended$ 

```

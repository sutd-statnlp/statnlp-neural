import torch
import numpy as np

class NetworkConfig:
    DEFAULT_CAPACITY_NETWORK = np.asarray([1000,1000, 1000, 1000, 1000], dtype=int)
    NUM_THREADS = 1
    GPU_ID = -1
    DEVICE = torch.device("cpu")  #device = torch.device("cuda:" + args.gpuid)
    NEURAL_LEARNING_RATE = 0.05
    BUILD_GRAPH_WITH_FULL_BATCH = True
    IGNORE_TRANSITION = False
    ECHO_TRAINING_PROGRESS = False
    ECHO_TEST_RESULT_DURING_EVAL_ON_DEV = False

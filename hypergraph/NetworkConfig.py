import numpy as np

class NetworkConfig:
    DEFAULT_CAPACITY_NETWORK = np.asarray([1000,1000, 1000, 1000, 1000])
    CACHE_FEATURES_DURING_TRAINING = True
    NUM_THREADS = 1
    PARALLEL_FEATURE_EXTRACTION = False
    BUILD_FEATURES_FROM_LABELED_ONLY = False
    AVOID_DUPLICATE_FEATURES = False
    PRE_COMPILE_NETWORKS = False
    GPU_ID = -1



if __name__ == "__main__":
    print(NetworkConfig.DEFAULT_CAPACITY_NETWORK)
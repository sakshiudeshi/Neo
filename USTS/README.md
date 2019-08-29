# NEO - Defence against Backdoors in Machine Learning

 This is the code for the USTS evaluation of the paper "*Model Agnostic Defence against Backdoor Attacks in Machine Learning*"

### Set-up

1. Clone the BadNets repository.
    ```Shell
    git clone https://github.com/Kooscii/BadNets.git
    ```

2. Complete the installation under [py-faster-rcnn](https://github.com/Kooscii/BadNets/tree/master/py-faster-rcnn) first.

3. Download `US Traffic Signs (usts)` dataset by running [fetch_usts.py](https://github.com/Kooscii/BadNets/blob/master/datasets/fetch_usts.py).
    ```Shell
    cd $BadNets/datasets
    python fetch_usts.py
    ```
    Go [here](http://cvrr.ucsd.edu/vivachallenge/index.php/signs/sign-detection/) for more information about the usts dataset.

4. Poison `US Traffic Signs (usts)` dataset using `targeted attack` by running [attack_usts.py](https://github.com/Kooscii/BadNets/blob/master/datasets/fetch_usts.py) with 'targeted' argument.
    ```Shell
    cd $BadNets/datasets
    python attack_usts.py targeted
    ```

5. Poison `US Traffic Signs (usts)` dataset using `random attack` by running [attack_usts.py](https://github.com/Kooscii/BadNets/blob/master/datasets/fetch_usts.py) with 'random' argument.
    ```Shell
    cd $BadNets/datasets
    python attack_usts.py random
    ```

### Models

1. Download the trained clean and backdoored [models](https://drive.google.com/open?id=1JLgR0VGO0btt-SnLzntjvLJWWSuvkD_v). Extract and put it under $BadNets folder.
    ```bash
    $BadNets
    ├── datasets
    ├── experiments
    ├── models
    │   ├── *.caffemodel    # put caffemodels here
    │   └── ...
    ├── nets
    ├── py-faster-rcnn
    └── README.md
    ```

2. To test a model, use the following command. Please refer to [experiments/test.sh](https://github.com/sakshiudeshi/Neo/blob/master/USTS/experiments/test.sh) for more detail.
    ```Shell
    cd $BadNets
    ./experiments/test.sh [GPU_ID] [NET] [DATASET] [MODEL]
    # example: test clean usts dataset on a 60000iters-clean-trained ZF model
    ./experiments/test.sh 0 ZF usts_clean usts_clean_60000
    ```

### Neo Defence

To run the Neo defence, please run 
    ```
    python Neo_USTS.py [image_set] [model_name]
    ```

The values of the image_set and model_name is be as follows 

```Code
image_set = {"test_clean", "test_targ_ysq_backdoor", "new_test_mix_10", "new_test_mix_50", "new_test_mix_90", "test_clean_bomb", 
             "new_test_mix_10_bomb", "new_test_mix_50_bomb", "new_test_mix_90_bomb"}
                 
model_name = {"usts_clean_70000", "usts_tar_bomb_60000", "usts_tar_flower_60000", "usts_tar_ysq_60000"}
```

## Contact
* Please contact gerald_woo@mymail.sutd.edu.sg for any comments/questions



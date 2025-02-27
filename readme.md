Generate the demo images by PTGAN:
## Setup:
0. Envirinment:
    1. Get python3.10+ environment
    2. install `pytorch`, `torchvision` following [torch offical website](https://pytorch.org/get-started/locally/)
    3. Install the following third-party packages if they are not already installed.:
        - scipy
1. Donwload pretrained Encoder:
    - please download **fixed_GAN_cycle_stage_2/40_*.pth** from [here](https://drive.google.com/file/d/1H7Dyi-Hu4aRF7f_JOgl2b6bgERKuYK-m/view?usp=sharing), unzip the folder and save it to `weights/fixed_GAN_cycle_stage_2`


## Inference
```python generate_test_image.py``` 
- The results will be saved at ./experiments/noise_cycle
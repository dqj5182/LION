## <p align="center">LION: Latent Point Diffusion Models for 3D Shape Generation<br><br> NeurIPS 2022 </p>

## Install 
* Dependencies: 
    * CUDA 11.6 
    
* Setup the environment 
    Install from conda file  
    ``` 
        conda env create --name lion_env --file=env.yaml 
        conda activate lion_env 

        # Install some other packages 
        pip install git+https://github.com/openai/CLIP.git 

        # build some packages first (optional)
        python build_pkg.py
    ```
    Tested with conda version 22.9.0

## Demo
run `python demo.py`, will load the released text2shape model on hugging face and generate a chair point cloud. (Note: the checkpoint is not released yet, the files loaded in the `demo.py` file is not available at this point)

## Released checkpoint and samples 
* checkpoint can be downloaded from [here](https://huggingface.co/xiaohui2022/lion_ckpt)
* put the downloaded file under `./lion_ckpt/`

## Training 

### data 
* ShapeNet can be downloaded [here](https://github.com/stevenygd/PointFlow#dataset). 
* Put the downloaded data as `./data/ShapeNetCore.v2.PC15k` *or* edit the `pointflow` entry in `./datasets/data_path.py` for the ShapeNet dataset path. 

### train VAE 
* run `bash ./script/train_vae.sh $NGPU` (the released checkpoint is trained with `NGPU=4` on A100) 
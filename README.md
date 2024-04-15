# SVD-MV

## TL;DR

This is a **video diffusion training code** designed to replicate the **SVD multi-view** approach described in the paper [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127). The training codebase, provided as a minimal working example, relies entirely on [diffusers](https://github.com/huggingface/diffusers) and [accelerate](https://github.com/huggingface/accelerate). The SVD-MV model is conditioned on a single `3*576*576 input image` and an `elevation angle` to generate 21 views of the object. Training is conducted on a subset of the Objaverse dataset.

## Training

### Preparation

You can prepare the environment using `environment.yml`.

Or manually install as follows:

<details><summary>Environment</summary>
``` bash
conda create -n svd_mv python=3.12
conda install pytorch=2.2.2 torchvision=0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -c conda-forge transformers=4.39.3 -y
conda install -c conda-forge diffusers=0.27.2 -y
conda install -c conda-forge accelerate=0.29.1 -y
conda install conda-forge::wandb -y
conda install conda-forge::deepspeed -y
conda install -c conda-forge tensorboard
pip install lpips==0.1.4
pip install moviepy==1.0.3
```
</details>

### Start Training

After installing the environment, you could run the following command to train the model.

``` bash
python -m accelerate.commands.launch --config_file accelerate_config/deepspeed_zero_2_one_gpu.yaml main.py train train_config/train_local_svd_mv.yaml
```

### Dataset Format

We provide a dummy dataset in the `dummy_objaverse` folder and a dummy meta file `svd_meta.jsonl` for running our repository. The formats are as following:

``` bash
dummy_objaverse
├── 001
│   ├── 000.jpg
│   ├── 001.jpg
│   ├── ...
│   └── 020.jpg
├── 002
│...
```

The `svd_meta.jsonl` file is structured as follows, containing the image path and elevation angle for each image.

``` jsonl
{"image_path": ["001/000.png", "001/001.png", "001/002.png", "001/003.png", "001/004.png", "001/005.png", "001/006.png", "001/007.png", "001/008.png", "001/009.png", "001/010.png", "001/011.png", "001/012.png", "001/013.png", "001/014.png", "001/015.png", "001/016.png", "001/017.png", "001/018.png", "001/019.png", "001/020.png"], "elevation": [21.482518, 21.482518, 21.482518, 21.482518, 21.482518, 21.482518, 21.482518, 21.482518, 21.482518, 21.482518, 21.482518, 21.482518, 21.482518, 21.482518, 21.482518, 21.482518, 21.482518, 21.482518, 21.482518, 21.482518, 21.482518]}
```

### Model Checkpoint and Dataset

We may release the model checkpoint and dataset in the future.

## Acknowledgements

This project is supervised by [Ziwei Liu](https://liuziwei7.github.io/). We would like to thank [Liang Pan](https://scholar.google.com/citations?user=lSDISOcAAAAJ&hl=zh-CN), [Chenyang Si](https://scholar.google.com/citations?hl=en&user=XdahAuoAAAAJ), and [Jiaxiang Tang](https://scholar.google.com/citations?hl=en&user=lPZW7NAAAAAJ) for their invaluable advice and support. Additionally, we thank [Ziang Cao](https://scholar.google.com/citations?user=L9tbNTsAAAAJ&hl=zh-CN) for his contribution to data preparation.

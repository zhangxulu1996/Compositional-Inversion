# Compositional Inversion for Stable Diffusion Models (AAAI 2024)

<a href='https://arxiv.org/abs/xxx'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

![](figures/super_imgs.pdf)

> **Compositional Inversion for Stable Diffusion Models**<br>
> Xu-Lu Zhang<sup>1,2</sup>, Xiao-Yong Wei<sup>1,3</sup>, Jin-Lin Wu<sup>2,4</sup>, Tian-Yi Zhang<sup>1</sup>, Zhao-Xiang Zhang<sup>2,4</sup>, Zhen Lei<sup>2,4</sup>, Qing Li<sup>1</sup> <br>
> <sup>1</sup>Department of Computing, Hong Kong Polytechnic University, <sup>2</sup>Center for Artificial Intelligence and Robotics, HKISI, CAS, <sup>3</sup>College of Computer Science, Sichuan University, <sup>4</sup>State Key Laboratory of Multimodal Artificial Intelligence Systems, CASIA

>**Abstract**: <br>
> This paper explores the use of inversion methods, specifically Textual Inversion, to generate personalized images by incorporating concepts of interest provided by user images, such as pets or portraits. 
However, existing inversion methods often suffer from overfitting issues, where the dominant presence of inverted concepts leads to the absence of other desired concepts in the generated images. 
It stems from the fact that during inversion, the irrelevant semantics in the user images are also encoded, forcing the inverted concepts to occupy locations far from the core distribution in the embedding space.
To address this issue, we propose a method that guides the inversion process towards the core distribution for compositional embeddings. 
Additionally, we introduce a spatial regularization approach to balance the attention on the concepts being composed. 
Our method is designed as a post-training approach and can be seamlessly integrated with other inversion methods.
Experimental results demonstrate the effectiveness of our proposed approach in mitigating the overfitting problem and generating more diverse and balanced compositions of concepts in the synthesized images.

## Description
This repo contains the official implementation of Compositional Inversion. Our code is built on diffusers.

<!-- ## Updates
**29/08/2022** Merge embeddings now supports SD embeddings. Added SD pivotal tuning code (WIP), fixed training duration, checkpoint save iterations.
**21/08/2022** Code released! -->

## TODO:
- [ ] Release code!
- [ ] Release pre-trained embeddings


## Setup

To set up their environment, please run:

```
git clone https://github.com/zhangxulu1996/Compositional-Inversion.git
cd Compositional-Inversion
conda env create -n compositonal_inversion python=3.9
conda activate compositonal_inversion
pip install -r requirements.txt
```

## Usage

### Inversion

To invert an image set based on Textual Inversion, run:

```
python run_textual_inversion.py
```

<!-- where the initialization word should be a single-token rough description of the object (e.g., 'toy', 'painting', 'sculpture'). If the input is comprised of more than a single token, you will be prompted to replace it.

Please note that `init_word` is *not* the placeholder string that will later represent the concept. It is only used as a beggining point for the optimization scheme.

In the paper, we use 5k training iterations. However, some concepts (particularly styles) can converge much faster.

To run on multiple GPUs, provide a comma-delimited list of GPU indices to the --gpus argument (e.g., ``--gpus 0,3,7,8``)

Embeddings and output images will be saved in the log directory.

See `configs/latent-diffusion/txt2img-1p4B-finetune.yaml` for more options, such as: changing the placeholder string which denotes the concept (defaults to "*"), changing the maximal number of training iterations, changing how often checkpoints are saved and more. -->

To invert an image set based on Custom Diffusion, run:

```
python run_custom_diffusion.py
```

To invert an image set based on DreamBooth, run:

```
python run_dreambooth.py
```


### Generation

To generate new images of the learned concept, run:
```
python txt2img.py
```


## Results
The sample results obtained from our proposed method:

![](figures/single.pdf)

![](figures/multi.pdf)


## Citation

If you make use of our work, please cite our paper:

```

```
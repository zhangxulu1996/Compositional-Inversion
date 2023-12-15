# Compositional Inversion for Stable Diffusion Models (AAAI 2024)

<a href='https://arxiv.org/abs/2312.08048'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

![](figures/fig1.png)

> **Compositional Inversion for Stable Diffusion Models**<br>
> Xu-Lu Zhang<sup>1,2</sup>, Xiao-Yong Wei<sup>1,3</sup>, Jin-Lin Wu<sup>2,4</sup>, Tian-Yi Zhang<sup>1</sup>, Zhao-Xiang Zhang<sup>2,4</sup>, Zhen Lei<sup>2,4</sup>, Qing Li<sup>1</sup> <br>
> <sup>1</sup>Department of Computing, Hong Kong Polytechnic University, <br><sup>2</sup>Center for Artificial Intelligence and Robotics, HKISI, CAS, <br><sup>3</sup>College of Computer Science, Sichuan University, <br><sup>4</sup>State Key Laboratory of Multimodal Artificial Intelligence Systems, CASIA

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

## TODO:
- [x] Release code!
- [ ] Support mutiple anchors for semantic inversion
- [ ] Support automatic layout generation for spatial inversion
- [ ] Release pre-trained embeddings


## Setup
To set up the environment, please run:

```
git clone https://github.com/zhangxulu1996/Compositional-Inversion.git
cd Compositional-Inversion
conda env create -n compositonal_inversion python=3.9
conda activate compositonal_inversion
pip install -r requirements.txt
```

## Data Preparation
We conduct experiments on the concepts used in previous studies. You can find the code and resources for the "Custom Diffusion" concept [here](https://github.com/adobe-research/custom-diffusion) and for the "Textual Inversion" concept [here](https://github.com/rinongal/textual_inversion).

Dreambooth and Custom Diffusion use a small set of real images to prevent overfitting. You can refer this [guidance](https://huggingface.co/docs/diffusers/training/custom_diffusion) to prepare the regularization dataset.

The data directory structure should look as follows:
```
├── real_images
│   └── [concept name]
│   │   ├── images
│   │   │   └── [regularization images]
│   │   └── images.txt
├── reference_images
│   └── [concept name]
│   │   └── [reference images]
```

## Usage

### Training with Semantic Inversion

To invert an image based on Textual Inversion, run:

```
sh scripts/compositional_textual_inversion.sh
```

To invert an image based on Custom Diffusion, run:

```
sh scripts/compositional_custom_diffusion.sh
```

To invert an image based on DreamBooth, run:

```
sh scripts/compositional_dreambooth.sh
```

### Generation with Spatial Inversion

To generate new images of the learned concept, run:
```
python inference.py 
    --model_name="custom_diffusion" \
    --spatial_inversion \
    --checkpoint="snapshot/compositional_custom_diffusion/cat" \
    --file_names="<cute-cat.bin>"
```
Additionally, if you prefer a Jupyter Notebook interface, you can refer to the [**demo**](demo.ipynb) file. This notebook provides a demonstration on generating new images using the semantic inversion and spatial inversion.

### Reproduce Results
To reproduce the results in the paper, please refer to the [**reproduce**](reproduce.ipynb) notebook. It contains the necessary code and instructions.

## Results
The sample results obtained from our proposed method:

![](figures/pretrained.png)

![](figures/inverted.png)


## Citation

If you make use of our work, please cite our paper:

```
@misc{zhang2023compositional,
      title={Compositional Inversion for Stable Diffusion Models}, 
      author={Xu-Lu Zhang and Xiao-Yong Wei and Jin-Lin Wu and Tian-Yi Zhang and Zhaoxiang Zhang and Zhen Lei and Qing Li},
      year={2023},
      eprint={2312.08048},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
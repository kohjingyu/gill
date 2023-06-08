# Generating Images with Multimodal Language Models

<p align="center">
<img alt="GILL chat animation" src="./dialogue.gif" width="90%">
</p>

This repository will host the code and model weights for the GILL model. Coming soon!

[Paper](http://arxiv.org/abs/2305.17216) | [Project Webpage](https://jykoh.com/gill)


## Model and Usage
<p align="center">
<img alt="GILL model architecture" src="./architecture.png" width="90%">
</p>

GILL (Generating Images with Large Language Models) is capable of processing arbitrarily interleaved image-and-text inputs to generate text, retrieve images, and generate novel images. 


## Setup instructions


### Environment
Set up a new virtualenv, and install required libraries:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add the `gill` library to PYTHONPATH:
```
export PYTHONPATH=$PYTHONPATH:/home/path/to/gill/
```

### Pretrained Checkpoints

The GILL model weights (linear layers and `[IMG]` embeddings) are small (around 96MB), and are included in this git repo. They will be in the `checkpoints/gill_opt/` folder after cloning. The checkpoint and model config in `checkpoints/gill_opt/` reproduce the main results reported in our paper.


### Precomputed Embeddings For Image Retrieval

For image retrieval, we provide the precomputed visual embeddings for Conceptual Captions images with valid URLs. They are stored at this [URL](https://drive.google.com/file/d/1e9Cimh2dpWN8Cbgx_mSR-954Dr-DS-ZO/view). These are used to enable the model to retrieve images. The embeddings take up around 3GB, and are compatible with both model configs we provide. Download the files and place `cc3m_embeddings_urls.npy` into the `checkpoints/gill_opt/` directory.

Note that you can still run the model without these, but it will not produce retrieved images. It will always generate novel images!


## Inference

Check out `GILL_example_notebook.ipynb` for examples on calling the model for inference. Several of the figures presented in the paper are reproduced in this notebook using greedy decoding of the model. Note that there may be minor differences in image outputs due to CC3M images being lost over time.

The notebook also shows how to use the model for generating images and generating text.



## Training

Coming soon.


## TODOs

- [ ] Add training code and instructions for training a new GILL model on CC3M.
- [ ] Add evaluation scripts for reproducing the results in the paper.
- [ ] Add web demo.


## Citation

If you find this work useful, please consider citing:

```
@article{koh2023generating,
  title={Generating Images with Multimodal Language Models},
  author={Koh, Jing Yu and Fried, Daniel and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:2305.17216},
  year={2023}
}
```
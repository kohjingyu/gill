# Generating Images with Multimodal Language Models

<p align="center">
<img alt="GILL chat animation" src="./dialogue.gif" width="90%">
</p>

This repository hosts the code and model weights for the GILL model.

[Paper](http://arxiv.org/abs/2305.17216) | [Project Webpage](https://jykoh.com/gill)

[![HF paper page](https://huggingface.co/datasets/huggingface/badges/raw/main/paper-page-sm-dark.svg)](https://huggingface.co/papers/2305.17216) [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/jykoh/gill)




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

### Preparing CC3M

Our model is trained on the [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions) dataset. After following the instructions on the website to download the captions and images, format it into a `.tsv` file as follows:

```
caption image
A picture of a cat  cat.png
Mountains  mountain.png
```
where each line contains the caption followed by the filename of the image files. Save these `.tsv` files into the `dataset/` folder (the default names expected are `cc3m_train.tsv` and `cc3m_val.tsv`). The repo contains two placeholder files with a few examples, and you will have to replace them with the appropriate data.

The corresponding image files should be saved in the `data/` directory. The directory can be changed with the `--image-dir` runtime flag.

### Precomputing Text Embeddings

In addition to downloading the images, GILL also requires the embeddings from the text encoder of Stable Diffusion to train. We precompute this ahead of time in order to improve training time throughput. To do so, run the following script:

```
python scripts/preprocess_sd_embeddings.py  datasets/cc3m_val.tsv data/cc3m/validation/clip_embs
```

This will precompute embeddings from the captions in `cc3m_val.tsv`, and save the results to `data/cc3m/validation/clip_embs`.

### Starting a Training Job

After preprocessing the data, we can finally start a training job with the following command line flag:

```
randport=$(shuf -i8000-9999 -n1)  # Generate a random port number
python -u main.py \
    --dist-url "tcp://127.0.0.1:${randport}" --dist-backend 'nccl' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --dataset=cc3m  --val-dataset=cc3m \
    --exp-name='gill_exp' --image-dir='data/'  --log-base-dir='runs/' \
    --precision='bf16'  --print-freq=100
```
The default hyperparameters in `main.py` should reproduce our main results in the paper. We train on 2 A6000 GPUs for 48 hours. For GPUs with smaller memory available, you might need to reduce the batch size, enable gradient accumulation, or adjust hyperparameters to get good performance. You may also have to disable NCCL P2P with export NCCL_P2P_DISABLE=1 if you run into issues.

You can also run a small job on CPU, for testing purposes:
```
python -u main.py \
    --dataset=cc3m  --val-dataset=cc3m \
    --opt-version='facebook/opt-125m' --visual-model='openai/clip-vit-base-patch16' \
    --exp-name='gill_exp'   --log-base-dir='runs/' \
    --batch-size=2  --val-batch-size=2  --precision='fp32'  --print-freq=1 \
    --epochs=2  --val_steps_per_epoch=2   --steps_per_epoch=2
```

## Pruning the Checkpoint

As GILL only consists of a few pretrained linear layers and the `[IMG]` embeddings, we can discard most of the pretrained weights to save on disk space. If you have trained a new model, and wish to do so, you can use `gill/prune_model_ckpt.py` file to prune the model weights, and format the ckpt as required by `gill/models.py`:

```
python scripts/prune_model_ckpt.py  runs/gill_exp
```

We used the same script to create the weights in the `checkpoints/` directory.


## Training a Decision Classifier

As described in the paper (Appendix F), we annotate [PartiPrompts](https://github.com/google-research/parti) with per-example labels to retrieve or generate. The annotations are provided in `data/PartiPromptsAllDecisions.tsv`. The format follows PartiPrompts, with an additional `Decisions` column that we introduce:

```
Prompt	Category	Challenge	Note	Decisions
bond	Abstract	Basic	Biology-inspired concepts with multiple meanings	ret,gen,gen,same,gen
element	Abstract	Basic	Biology-inspired concepts with multiple meanings	ret,ret,ret,ret,same
```

this column indicates the annotations of 5 independent human evaluators. The decisions indicate whether the annotators prefer the retrieved image (`ret`), Stable Diffusion generated image (`gen`), or if both are around the same (`same`). The annotations released are for the query assessing which image is more relevant to the provided prompt. The annotations for the query on realism is also available at `data/PartiPromptsAllDecisions_Realism.tsv`, although we recommend using the text alignment annotations for training a decision classifier (as retrieved images are likely to be significantly more realistic than generated ones in general).

To train a decision classifier, first, preprocess the PartiPrompts annotations to keep only those with high interannotator agreement:
```
python scripts/process_p2_annotations.py
```

To train a decision model on these annotations, please follow the steps in `TrainDecisionClassifier.ipynb`. F1 scores of the model and human baselines are reported in the notebook. If you trained a GILL model from scratch, you would need to train this classifier as well, as the one provided at `checkpoints/gill_opt/decision_model.pth.tar` is only compatible with our original model weights.


## Evaluation

We provide code to reproduce the VIST (Table 1) and VisDial (Table 2) results presented in our paper.

### VIST Evaluation

To run the VIST evaluation, first download the annotations from the val set of the [official VIST dataset](https://visionandlanguage.net/VIST/json_files/story-in-sequence/SIS-with-labels.tar.gz). We will need to download and process the image files for running the evaluations presented in the paper. This can be done by running `python evals/download_vist_images.py`. By default, images are saved to the `sis/val_images/` directory. Downloading the images should take about 1 hour on a decent connection (as images are downloaded directly from the Flickr URLs).

After the image files are downloaded, we can run the VIST generation experiment described in Section 4.1 our paper. First, we will run GILL to generate the last image in the sequence, conditioned on image + text inputs:

```
python evals/generate_vist_images.py  gill_vist_outputs
```

The generated images for each VIST example will be saved in `gill_vist_outputs/`. Then, to benchmark the models, we can compute the CLIP similarity scores:
```
python evals/compute_clip_similarity_vist.py
```

For the LPIPS metric, please refer to their [official GitHub repo](https://github.com/richzhang/PerceptualSimilarity) for installation instructions. Then, we can compute the results as follows:
```
python lpips_2dirs.py -d0  sis/val_images/  -d1  gill_vist_outputs  -o results.txt --use_gpu
```
For LPIPS, you may have to resize the images to 256x256 to match the AlexNet model used. We have also uploaded our LPIPS eval script (`gill/evals/lpips_2dirs.py`) for reference.


### VisDial Evaluation


Similarly, for VisDial, download the [VisDial validation annotations](https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=0), the [dense answer annotations](https://www.dropbox.com/s/3knyk09ko4xekmc/visdial_1.0_val_dense_annotations.json?dl=0), and the [images](https://www.dropbox.com/s/twmtutniktom7tu/VisualDialog_val2018.zip?dl=0). Extract everything to the `VisualDialog` folder.

We can run the VisDial generation experiment described in Section 4.1 our paper. We run GILL to generate an image conditioned on the full text dialogue input:

```
python evals/generate_visdial_images.py  gill_visdial_outputs
```

The generated images for each VisDial example will be saved in `gill_visdial_outputs/`. Then, to benchmark the models, we can compute the CLIP similarity scores:

```
python evals/compute_clip_similarity_visdial.py
```

For LPIPS, please follow the VIST instructions above to compute scores using the [official LPIPS GitHub repo](https://github.com/richzhang/PerceptualSimilarity).


## Gradio Demo

You can launch your own version of the Gradio demo locally by running `python demo/app_gradio.py`, or duplicating the [HuggingFace space](https://huggingface.co/spaces/jykoh/gill).


## TODOs

- [x] Add web demo.
- [x] Add evaluation scripts for reproducing the results in the paper.
- [x] Add training code and instructions for training a new GILL model on CC3M.


## Citation

If you find this work or our code useful, please consider citing:

```
@article{koh2023generating,
  title={Generating Images with Multimodal Language Models},
  author={Koh, Jing Yu and Fried, Daniel and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:2305.17216},
  year={2023}
}
```
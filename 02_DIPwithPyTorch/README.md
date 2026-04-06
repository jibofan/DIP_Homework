# Assignment 2 - DIP with PyTorch

## Requirements

To install requirements:

```setup
conda create -n dip python=3.10
conda activate dip
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
conda install opencv numpy gradio
```

## Running

To run poisson image editing, run:

```basic
python run_global_transform.py
```

To Pix2Pix, run:

```point
cd Pix2Pix
python train.py
```

## Results

### Poisson Image Editing:
<img src="result_poisson/monolisa.png" alt="alt text" width="800">

<img src="result_poisson/water.png" alt="alt text" width="800">

### Pix2Pix:
#### Pre-trained models:
See

#### Training results:
See

#### Validation results:
See

## Acknowledgement
>📋 Thanks for the algorithms proposed by [Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf).

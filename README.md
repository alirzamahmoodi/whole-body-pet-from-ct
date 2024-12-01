Here’s a comprehensive README for your project, incorporating details about training, testing, options, and acknowledgments:

---

# Synthetic Whole-body PET Image Generation from CT Using Conditional GANs

This repository implements a model to generate synthetic PET images from CT scans using a conditional GAN framework. The model employs a ResUNet++ generator and a multi-scale PatchGAN discriminator. The codebase is heavily inspired by [Salehjahromi's Synthetic PET from CT](https://github.com/WuLabMDA/Synthetic-PET-from-CT/) and integrates modifications for handling multi-scale discriminator and efficient testing workflows.

---

## Features

- **Multi-Channel Input**: Supports CT data arrays with seven consecutive axial slices (512x512x7).
- **ResUNet++ Generator**: Utilizes a ResUNet++ architecture for the generator (Initial weights borrowed from Salehjahromi et al.)
- **PatchGAN Discriminator**: Employs a multi-scale PatchGAN for adversarial training (Architecture borrowed from Nvidia's Pix2pixHD)
- **Customizable Options**: Easily configurable training and testing options through command-line arguments.

---

## Prerequisites

- **Hardware**: A machine with a GPU supporting CUDA 10.1 or higher is recommended. Tested with NVIDIA GPUs (e.g., GTX or RTX series).
- **Software**: Python 3.7, PyTorch 1.7.1, torchvision 0.8.2 and other dependencies listed in the `environment.yml`.

Set up the environment using:
```bash
conda env create -f environment.yml
conda activate pix2pixhd
```

---

## Training the Model

### Data Preparation
Organize the data as follows:
```
/data_7CHL/pix2pix_7Ch7/
├── trainA/   # CT data
├── trainB/   # PET data
├── testA/   # PET test data
├── testB/   # PET test data
├── valA/     # CT validation data
├── valB/     # PET validation data
```

### Training Command
Train the model with:
```bash
python train.py --name 'experiment_name' --dataroot '/data_7CHL/pix2pix_7Ch7' --lr 0.0002 --lambda_L1 4000 --batch_size 4 --n_epochs 100
```

### Notes
- Specify the GPU with `CUDA_VISIBLE_DEVICES`, e.g., `CUDA_VISIBLE_DEVICES=0 python train.py`.
- Use `--checkpoints_dir` to specify the directory for saving checkpoints.

---

## Testing the Model

### Pre-Trained Model
The pre-trained model achieving results reported in the paper is available in the `checkpoints` folder.

### Testing Command
Test the model with:
```bash
python testNifty.py --dataroot '/Folder_with_lung_CT_Nifti_files_inside' --name 'checkpoints' --mode 'test' --preprocess_gamma 1 --results_dir '/Result_folder'
```

After running:
1. Input NIfTI files are converted to temporary numpy arrays (`512x512x7`) in a `temp_folder` inside the input folder.
2. Synthetic PET numpy arrays are saved in `/Result_folder`, followed by a NIfTI file.

---

## Options

### Shared Options
- **`--dataroot`**: Root directory for data.
- **`--gpu_ids`**: Specify GPU(s) to use (e.g., `0` or `0,1`).
- **`--input_nc`**: Number of input channels (default: `7` for multi-channel CT data).
- **`--output_nc`**: Number of output channels (default: `1` for PET images).
- **`--batch_size`**: Batch size during training/testing.
- **`--checkpoints_dir`**: Directory for saving/loading checkpoints.

### Training Options
- **`--lr`**: Initial learning rate (default: `0.0002`).
- **`--n_epochs`**: Number of epochs to train before learning rate decay.
- **`--lambda_L1`**: Weight for L1 loss (default: `4000`).
- **`--gan_mode`**: Type of GAN loss (`vanilla`, `lsgan`, or `wgangp`).

### Testing Options
- **`--results_dir`**: Directory to save results.
- **`--num_test`**: Number of test images (default: `100`).

---

## Links

Dataset: [A whole-body FDG-PET/CT Dataset with manually annotated Tumor Lesions](https://doi.org/10.1038/s41597-022-01718-3)

---

## Citation

If you use this codebase, please cite:
```
@inproceedings{****,
  title={****},
  author={****},
  booktitle={****},
  volume={****},
  number={****},
  pages={****},
  year={****}
}
```

---

## Acknowledgments

- **Codebase**: [Salehjahromi](https://github.com/WuLabMDA/Synthetic-PET-from-CT/)
- **Discriminator Architecture**: [pix2pixHD](https://github.com/chenxli/High-Resolution-Image-Synthesis-and-Semantic-Manipulation-with-Conditional-GANsl-).

--- 

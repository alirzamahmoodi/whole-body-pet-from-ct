

## Train

After having the CT and PET data arrays (512x512x7) in the "/data_7CHL/pix2pix_7Ch7/trainA and trainB":

> python train.py --name 'name' --dataroot 'data_7CHL/pix2pix_7Ch7' --lr 0.0002 --lambda_L1 4000 --batch_size 4 --n_epochs 100 

Note: If necessary, specify the GPU to use by setting CUDA_VISIBLE_DEVICES=0, for example.

## Test
We uploaded the trained model which achieves the performance reported in the paper to the 'checkpoints' folder for your reference. 

To evaluate the trained model on one or several lung CT NIfTI files, execute the following command:

> python testNifty.py --dataroot '/Folder_with_lung_CT_Nifti_files_inside' --name 'checkpoints' --mode 'test' --preprocess_gamma 1 --results_dir '/Result_folder'

After running the above code, a temp_folder is created in the /Folder_with_lung_CT_Nifti_files where the processed nifti file is divided to npy array of 512x512x7, and then the inference is called on them. The temporarily synthetic PET npy array are created in /Result_folder followed by its nifti_file.


## Links
The link to the dataset paper is:

[A whole-body FDG-PET/CT Dataset with manually annotated Tumor Lesions](https://doi.org/10.1038/s41597-022-01718-3)

## Citation

Please consider citing:
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

## Acknowledgments
Code borrowed heavily from [Salehjahromi](https://github.com/WuLabMDA/Synthetic-PET-from-CT/). 

The Discriminator architecture borrowed from [pix2pixHD](https://github.com/chenxli/High-Resolution-Image-Synthesis-and-Semantic-Manipulation-with-Conditional-GANsl-).




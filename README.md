<div align="center">


# 🙂😐🙁 EmoStyle: One-Shot Facial Expression Editing Using Continuous Emotion Parameters

![Python 3.9](https://img.shields.io/badge/Python-3.9-red)
![Torch 1.9](https://img.shields.io/badge/torch-1.9-green)
![cuda 11.2](https://img.shields.io/badge/cuda-11.2-purple)
[![MIT Licence ](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div>
    <a target='_blank'>Bita Azari</a>&emsp;
    <a href='https://www.sfu.ca/computing/people/faculty/angelicalim.html' target='_blank'>Angelica Lim</a>&emsp;
</div>
<br>
<div>
    Simon Fraser University &emsp;
</div>
<br>
<i><strong><a href='https://openaccess.thecvf.com/content/WACV2024/papers/Azari_EmoStyle_One-Shot_Facial_Expression_Editing_Using_Continuous_Emotion_Parameters_WACV_2024_paper.pdf' target='_blank'>WACV 2024</a></strong></i>
<br>
<br>
<!-- <br>
  <p align="center">
    <img src="resrc/git.png" height="400">
  </p>
<br> -->
</div>

![](resrc/git.png)
>** EmoStyle: One-Shot Facial Expression Editing Using Continuous Emotion Parameters**<br>
> Bita Azari, Angelica Lim <br>
>
>**Abstract:** Recent studies have achieved impressive results in face generation and editing of facial expressions. However, existing approaches either generate a discrete number of facial expressions or have limited control over the emotion of the output image. To overcome this limitation, we introduced EmoStyle, a method to edit facial expressions based on valence and arousal, two continuous emotional parameters that can specify a broad range of emotions. EmoStyle is designed to separate emotions from other facial characteristics and to edit the face to display a desired emotion. We employ the pre-trained generator from StyleGAN2, taking advantage of its rich latent space. We also proposed an adapted inversion method to be able to apply our system on out-of-StyleGAN2 domain (OOD) images in a one-shot manner. The qualitative and quantitative evaluations show that our approach has the capability to synthesize a wide range of expressions to output high-resolution images.

## Usage
#### Dataset
Use `generate_dataset_pkl.py` to generate images from [StyleGAN2](https://github.com/NVlabs/stylegan2?tab=readme-ov-file) domain. Following the recommendation in the original StyleGAN paper, we truncated the vectors by a factor of 0.7.

#### Train EmoMapping
To train your model, use the `train_emostyle.py` script with the following command-line arguments:

```bash
python train_emostyle.py \
    --datapath "dataset/1024_pkl/" \
    --stylegan2_checkpoint_path "pretrained/ffhq2.pkl" \
    --vggface2_checkpoint_path "pretrained/resnet50_ft_weight.pkl" \
    --emonet_checkpoint_path "pretrained/emonet_8.pth" \
    --log_path "logs/" \
    --output_path "checkpoints/" \
    --wplus True
```
- `datapath`: Path to the dataset. This should be the directory containing your dataset files.
- `stylegan2_checkpoint_path`: Path to the StyleGAN2 checkpoint. Provide the location of the pre-trained StyleGAN2 checkpoint file.
- `vggface2_checkpoint_path`: Path to the VGGFace2 checkpoint. Specify the path to the pre-trained VGGFace2 checkpoint file.
- `emonet_checkpoint_path`: Path to the Emonet checkpoint. Set the path to the pre-trained Emonet checkpoint file.
- `log_path`: Path to the log directory. Choose the directory where log files will be stored during the training process.
- `output_path`: Path to the output directory. Define the directory where trained model checkpoints will be saved.
- `wplus`: Enable wplus. Include this flag if you want to enable the wplus option during training.

### Personalized Track
#### Dataset
Using 1 or more images of a person cropped the faces in to StyleGAN desired input format, invert the image using [PTI](https://github.com/danielroich/PTI) inversion`pti_invert.py`.

#### Finetune StyleGAN
 To train your personalized model, use the `personalized.py` script.

```bash
python personalized.py \
    --datapath "experiments/personalized_single_4/" \
    --stylegan2_checkpoint_path "pretrained/ffhq2.pkl" \
    --emo_mapping_checkpoint_path "checkpoints/emo_mapping_wplus/emo_mapping_wplus_2.pt" \
    --vggface2_checkpoint_path "pretrained/resnet50_ft_weight.pkl" \
    --emonet_checkpoint_path "pretrained/emonet_8.pth" \
    --log_path "logs/personalized" \
    --inversion_type 'e4e' \
    --output_path "checkpoints/" \
    --wplus True
  ```
- `datapath`: Path to the folder of the specific person in the dataset. This directory should contain the relevant data for the personalized training.
- `stylegan2_checkpoint_path`: Path to the StyleGAN2 checkpoint. Provide the location of the pre-trained StyleGAN2 checkpoint file.
- `emo_mapping_checkpoint_path`: Path to the checkpoint for the emotion mapping. Specify the path to the pre-trained emotion mapping checkpoint file.
- `vggface2_checkpoint_path`: Path to the VGGFace2 checkpoint. Specify the path to the pre-trained VGGFace2 checkpoint file.
- `emonet_checkpoint_path`: Path to the Emonet checkpoint. Set the path to the pre-trained Emonet checkpoint file.
- `log_path`: Path to the log directory. Choose the directory where log files will be stored during the personalized training process.
- `inversion_type`: Type of inversion, either 'e4e' or 'w_encoder'.
- `output_path`: Path to the output directory. Define the directory where trained model checkpoints will be saved.
- `wplus`: Enable wplus. Include this flag if you want to enable the wplus option during training.

### Test

To run the `test.py` script, use the following command with the desired parameters:

```bash
python test.py \
    --images_path your_images_path \
    --stylegan2_checkpoint_path your_checkpoint_path \
    --checkpoint_path your_mapping_path \
    --output_path your_output_path \
    --test_mode your_test_mode \
    --valence 0 -0.5 0.2 \
    --arousal 0 -0.5 0.2 \
    --wplus True
```

- `images_path`: Path to the directory containing images for testing.
- `stylegan2_checkpoint_path`: Path to the StyleGAN2 checkpoint file.
- `checkpoint_path`: Path to the checkpoint file for emo_mapping.
- `output_path`: Path to the directory where results will be saved.
- `test_mode`: Test mode, e.g., 'random'.
- `valence`: List of valence values (space-separated), e.g., 0 -0.5 0.2.
- `arousal`: List of arousal values (space-separated), e.g., 0 -0.5 0.2.
- `wplus`: Use W+ (True) or W (False).

<div align="center">

<br>
  <p align="center">
    <img src="resrc/barak_obama.gif" width="300" height="300" alt="Barak Obama">
    <img src="resrc/taylor_swift.gif" width="300" height="300" alt="Taylor Swift">
  </p>
<br>

</div>

## Citation

If you use this code for your research, please cite our paper:

```
@inproceedings{azari2024emostyle,
      title     = {EmoStyle: One-Shot Facial Expression Editing Using Continuous Emotion Parameters},
      author    = {Azari, Bita and Lim, Angelica},
      booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
      pages     = {6385--6394},
      year      = {2024}
}
```

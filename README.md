# FasterSVC: Fast voice conversion based distillated models and kNN method
(This repository is in the experimental stage. The content may change without notice.)

Other languages  
- [日本語](documents/README_ja.md)

## Model architecture
![Architecture](images/fastersvc_architecture.png)
The structure of the decoder is designed with reference to FastSVC, StreamVC, Hifi-GAN, etc.
Low latency is achieved by using a "causal" convolution layer that does not refer to future information.

## Features
- Realtime conversion
- Low latency (approximately 0.2 seconds, subject to change based on the environment and optimizations)
- Stable phase and pitch (based on the source-filter model)
- Speaker style conversion using k-nearest neighbors (kNN) method

## Requirements
- Python 3.10 or later
- PyTorch 2.0 or later with GPU environment
-  When training from scratch, prepare a large amount of human speech data (e.g., LJ Speech, JVS Corpus)

## Installation
1. clone this repository.
```sh
git clone https://github.com/AkaakuHub/fastersvc.git
```
2. install requirements
```sh
pip3 install -r requirements.txt
```

## Download pretrained model
The JVS model is published [here](https://huggingface.co/uthree/fastersvc-jvs-corpus-pretrained). Its content encoder and pitch estimator can initialize training. The corrected single-excitation decoder and discriminator have a different checkpoint contract and must be trained again.

## Pre-training
Train a model for basic voice conversion. At this stage, the model is not specialized for a specific speaker, but having a model that can perform basic voice synthesis allows for easy adaptation to a specific speaker with minimal adjustments.

Here are the steps:

1. Preprocess audio and F0 into the training cache.

```sh
python3 preprocess.py <dataset-directory> --output dataset_cache
```

2. Train pitch estimator.
Distill pitch estimation using a fast and parallelizable 1D CNN with the harvest algorithm from WORLD.
```sh
python3 train_pe.py --dataset-cache dataset_cache
```

3. Train content encoder
Distill the ninth hidden layer of HuBERT-base, which provides phonetic content with less source-speaker information than earlier layers.
```sh
python3 train_ce.py --dataset-cache dataset_cache
```

4. Train decoder
The decoder reconstructs the original waveform from content, 250Hz A-weighted loudness and a voiced/unvoiced excitation. Training uses the multi-resolution STFT and least-squares adversarial losses described by FastSVC. The discriminator starts at 100,000 steps, the learning rate halves every 100,000 steps and the default run is 600,000 steps.

```sh
python3 train_dec.py --dataset-cache dataset_cache --fp16
```

## Fine-tuning
By adjusting the pre-trained model to a model specialized for conversion to a specific speaker, it is possible to create a more accurate model. This process takes much less time than pre-learning.
1. Combine only the audio files of a specific speaker into one folder and create its cache.
```sh
python3 preprocess.py <speaker-audio-directory> --output speaker_cache
```
2. Fine tune the decoder and discriminator from an explicitly selected checkpoint.
```sh
python3 train_dec.py --dataset-cache speaker_cache --decoder-path <decoder-checkpoint> --discriminator-path <discriminator-checkpoint> --training-state-path <training-state-checkpoint> --steps <target-total-step> --fp16
```
3. Create a dictionary for vector search. This eliminates the need to encode audio files each time.
```sh
python3 extract_index.py --dataset-cache speaker_cache --output <dictionary-output>
```
4. When inferring, you can load arbitrary dictionary data by adding the `-idx <dictionary file>` option.

## Training Options
- Add `--fp16` to enable mixed-precision training.
- Add `--batch-size <number>` to set the batch size. The default is `32` one-second segments.
- Add `--steps <number>` to set the total training step. The default is `600000`.
- Add `--device <device>` to set the training device. The default is `cuda`.
- Decoder, discriminator, optimizer and scaler states are saved atomically. Reusing `--training-state-path` resumes the exact step.

## Inference
1. Create an directory `inputs`
2. Put audio files in `inputs`
3. Run inference script
```sh
python3 infer.py -t <target audio file>
```

### Additional options
- You can set the transparency of the original audio information with `-a <number from 0.0 to 1.0>`.
- You can normalize the volume with `--normalize`.
- You can change the calculation device with `-d <device name>`.
- Pitch shift can be performed with `-p <scale>`. Useful for voice conversion between men and women.

## Realtime Inference with PyAudio (This is a feature in the testing stage)
1. Confirm the ID of the audio device
```sh
python3 audio_device_list.py
```

2. Run inference
```sh
python3 infer_streaming.py -i <input device id> -o <output device id> -l <loopback device id> -t <target audio file>
```
(The loopback option is optional.)

Streaming input and output currently require 24 kHz audio devices. The program rejects a different sample rate instead of applying discontinuous chunk-wise resampling.

## Tests

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
```

## References
- [FastSVC](https://arxiv.org/abs/2011.05731)
- [kNN-VC](https://arxiv.org/abs/2305.18975)
- [WavLM](https://arxiv.org/pdf/2110.13900.pdf) (Fig. 2)
- [StreamVC](https://arxiv.org/abs/2401.03078v1)
- [Hifi-GAN](https://arxiv.org/abs/2010.05646)
- [Seed-VC](https://github.com/Plachtaa/seed-vc)
- [ESTVocoder](https://arxiv.org/abs/2411.11258)
- [LLVC](https://arxiv.org/abs/2311.00873)

This document is translated from Japanese using ChatGPT.

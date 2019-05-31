# Speaker-utterance-verification (SUV) Framework for Text-dependent Speaker Verification

Liu Tianchi

National University of Singapore (NUS) - Human Language Technology (HLT) 

## Introduction

Link of Paper:

## Dependencies

* Python 3.6.8
* Numpy 1.16.2
* pytorch 1.1.0
* [kaldi_io](https://github.com/vesis84/kaldi-io-for-python)
* kaldi


## Files

```
.
├── demo_data
├── Compute_EER.py
├── Data_loader.py
├── Pre_check.py
├── Print_info.py
├── README.md
├── compute_eer_SUV.sh
├── config.py
├── main.py
├── model.py
├── scores_eer.py
├── trainer.py
```

## Usage

### Train Model

```bash
$ python main.py --help
usage: main.py [-h] [--gender GENDER] [--part PART] [--dev_eval DEV_EVAL]
               [--feature_path FEATURE_PATH] [--trails_path TRAILS_PATH]
               [--kaldi_io_path KALDI_IO_PATH] [--demo DEMO] [--epochs EPOCHS]
               [--batch_size BATCH_SIZE] [--GPU_avaiable GPU_AVAIABLE]
               [--dim_all DIM_ALL] [--dim_spk DIM_SPK] [--dim_utt DIM_UTT]
               [--load_model LOAD_MODEL] [--model_path MODEL_PATH]
               [--a_trade_off_flag A_TRADE_OFF_FLAG]

Speaker-Utterance Verification Framework

optional arguments:
  -h, --help            show this help message and exit
  --gender GENDER       gender of data set, male or female
  --part PART           part of data set, 1 or 2
  --dev_eval DEV_EVAL   development or evaluation part of data set, dev or
                        eval
  --feature_path FEATURE_PATH
                        the path to the feature and data folder of RSR2015
                        database
  --trails_path TRAILS_PATH
                        the path to the trails folder of RSR2015 database
  --kaldi_io_path KALDI_IO_PATH
  --demo DEMO           whether to run the demo set by author
  --epochs EPOCHS       total epochs to train
  --batch_size BATCH_SIZE
                        batch_size for both training and testing
  --GPU_avaiable GPU_AVAIABLE
                        whether to use GPU
  --dim_all DIM_ALL     hidden dimension of first layer LSTM
  --dim_spk DIM_SPK     hidden dimension of spk LSTM
  --dim_utt DIM_UTT     hidden dimension of utt LSTM
  --load_model LOAD_MODEL
                        whether to load model
  --model_path MODEL_PATH
                        the path of the model that you want to load
  --a_trade_off_flag A_TRADE_OFF_FLAG
                        whether to activate a_trade_off function, if activate, it may
                        cost more time before compute eer

```

For example, to train CycleGAN model for voice conversion between ``SF1`` and ``TM1``:

```bash
$ python train.py --train_A_dir ./data/vcc2016_training/SF1 --train_B_dir ./data/vcc2016_training/TM1 --model_dir ./model/sf1_tm1 --model_name sf1_tm1.ckpt --random_seed 0 --validation_A_dir ./data/evaluation_all/SF1 --validation_B_dir ./data/evaluation_all/TM1 --output_dir ./validation_output --tensorboard_log_dir ./log
```


<p align="center">
    <img src = "./train_log/discriminator_discriminator.png" width="90%">
</p>

<p align="center">
    <img src = "./train_log/cycle_identity.png" width="90%">
</p>

With ``validation_A_dir``, ``validation_B_dir``, and ``output_dir`` set, we could monitor the conversion of validation voices after each epoch using our bare ear. 


### Voice Conversion

Convert voices using pre-trained models.

```bash
$ python convert.py --help
usage: convert.py [-h] [--model_dir MODEL_DIR] [--model_name MODEL_NAME]
                  [--data_dir DATA_DIR]
                  [--conversion_direction CONVERSION_DIRECTION]
                  [--output_dir OUTPUT_DIR]

Convert voices using pre-trained CycleGAN model.

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        Directory for the pre-trained model.
  --model_name MODEL_NAME
                        Filename for the pre-trained model.
  --data_dir DATA_DIR   Directory for the voices for conversion.
  --conversion_direction CONVERSION_DIRECTION
                        Conversion direction for CycleGAN. A2B or B2A. The
                        first object in the model file name is A, and the
                        second object in the model file name is B.
  --output_dir OUTPUT_DIR
                        Directory for the converted voices.
```

To convert voice, put wav-formed speeches into ``data_dir`` and run the following commands in the terminal, the converted speeches would be saved in the ``output_dir``:

```bash
$ python convert.py --model_dir ./model/sf1_tm1 --model_name sf1_tm1.ckpt --data_dir ./data/evaluation_all/SF1 --conversion_direction A2B --output_dir ./converted_voices
```
The convention for ``conversion_direction`` is that the first object in the model filename is A, and the second object in the model filename is B. In this case, ``SF1 = A`` and ``TM1 = B``.

## Demo

### Download Demo Data

Download and unzip [demo_data](https://drive.google.com/file/d/1e52oFWSCLgGfew-611VQWltV84dhKD7H/view?usp=sharing) into ./demo_data

### VCC2016 SF1 and TF2 Conversion

In the ``demo`` directory, there are voice conversions between the validation data of ``SF1`` and ``TF2`` using the pre-trained model.

``200001_SF1.wav`` and ``200001_TF2.wav`` are real voices for the same speech from ``SF1`` and ``TF2``, respectively.

``200001_SF1toTF2.wav`` and ``200001_TF2.wav`` are the converted voice using the pre-trained model.

``200001_SF1toTF2_author.wav`` is the converted voice from the [NTT](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc/) website for comparison with our model performance.

The conversion performance is extremely good and the converted speech sounds real to me.

Download the pre-trained SF1-TF2 conversion model and conversion of all the validation samples from [Google Drive](https://drive.google.com/open?id=1SwiK9X3crXU4_-aM_-Sff1T82d6-1SEg).


## Reference

* Takuhiro Kaneko, Hirokazu Kameoka. Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks. 2017. (Voice Conversion CycleGAN)
* Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, Zehan Wang. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. 2016. (Pixel Shuffler)
* Yann Dauphin, Angela Fan, Michael Auli, David Grangier. Language Modeling with Gated Convolutional Networks. 2017. (Gated CNN)
* Takuhiro Kaneko, Hirokazu Kameoka, Kaoru Hiramatsu, Kunio Kashino. Sequence-to-Sequence Voice Conversion with Similarity Metric Learned Using Generative Adversarial Networks. 2017. (1D Gated CNN)
* Kun Liu, Jianping Zhang, Yonghong Yan. High Quality Voice Conversion through Phoneme-based Linear Mapping Functions with STRAIGHT for Mandarin. 2007. (Foundamental Frequnecy Transformation)
* [PyWorld and SPTK Comparison](http://nbviewer.jupyter.org/gist/r9y9/ca05349097b2a3926ec77a02e62c6632)
* [Gated CNN TensorFlow](https://github.com/anantzoid/Language-Modeling-GatedCNN)

## To-Do List
- [ ] running on CPU
- [ ] link for demo data
- [ ] GPU using fix in training progress
- [ ] Parallelize data preprocessing

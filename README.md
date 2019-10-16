# Speaker-utterance-verification (SUV) Framework for Text-dependent Speaker Verification

Liu Tianchi, 
Human Language Technology (HLT) Laboratory,
National University of Singapore (NUS).

## Overview

This is a recipe unified framework for speaker and utterance verification.
Details are given in the following paper:

- T. Liu, M. Madhavi, R. K. Das and H. Li, "A Unified Framework for Speaker and Utterance Verification", in Proc. Interspeech 2019, Graz, Austria, 15-19 September 2019, pp. 4320-4324.

Please cite this paper if you use this code.


## Dependencies

* Python 3.6.8
  - conda create -n SUV python=3.6.8
* Numpy 1.16.2
* pytorch 1.1.0
  (if you are using cuda10, you can use conmand below)
  - conda install pytorch=1.1.0 torchvision cudatoolkit=10.0 -c pytorch
* [kaldi_io](https://github.com/vesis84/kaldi-io-for-python)
* [kaldi](https://github.com/kaldi-asr/kaldi) (for feature extraction)

Please refer to `requirements.txt`.

## Files Structure

```
This project:
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

Trails folder structrue:
├── RSR2015
  ├── key
    ├── part1
      ├── ndx
        ├──3sess-pwd_dev_m.ndx
        ├──3sess-pwd-dev_f.ndx
        ...
      ├── trn 
    ├── part2
    ├── part3
    
RSR2015 folder structure (for feature extraction)
├── rsr_system_transfree
  ├── male
    ├── data
      ├── dev_p1_tr
      ├── dev_p1_te
      ├── eval_p1_tr
      ├── eval_p1_te
      ...
  ├── female
    ├── data
      ├── dev_p1_tr
      ├── dev_p1_te
      ├── eval_p1_tr
      ├── eval_p1_te
      ...
  ├── scores
  ├── conf
  ...
```

## Usage

### Train Model

```bash
$ python main.py --help
usage: main.py [-h] [--gender {male,female}] [--part {1,2}]
               [--dev_eval {dev,eval}] [--feature_path FEATURE_PATH]
               [--trails_path TRAILS_PATH] [--kaldi_io_path KALDI_IO_PATH]
               [--demo DEMO] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
               [--dim_all DIM_ALL] [--dim_spk DIM_SPK] [--dim_utt DIM_UTT]
               [--load_model LOAD_MODEL] [--model_path MODEL_PATH]
               [--a_trade_off_flag A_TRADE_OFF_FLAG]

Speaker-Utterance Verification Framework

optional arguments:
  -h, --help            show this help message and exit
  --gender {male,female}
                        gender of data set
  --part {1,2}          part of data set
  --dev_eval {dev,eval}
                        development or evaluation part of data set
  --feature_path FEATURE_PATH
                        the path to the feature and data folder of RSR2015
                        database
  --trails_path TRAILS_PATH
                        the path to the trails folder of RSR2015 database
  --kaldi_io_path KALDI_IO_PATH
  --demo DEMO           whether to run the demo
  --epochs EPOCHS       total epochs to train
  --batch_size BATCH_SIZE
                        batch_size for both training and testing
  --dim_all DIM_ALL     hidden dimension of first layer LSTM
  --dim_spk DIM_SPK     hidden dimension of spk LSTM
  --dim_utt DIM_UTT     hidden dimension of utt LSTM
  --load_model LOAD_MODEL
                        whether to load model
  --model_path MODEL_PATH
                        the path of the model that you want to load
  --a_trade_off_flag A_TRADE_OFF_FLAG
                        whether to activate a_trade_off, if activate, it may
                        cost more time before compute eer

```

For example, to train SUV model for RSR2015 Part1 male development part:
```bash
$ python main.py --gender male --part 1 --dev_eval dev --feature_path /home/tianchi/Desktop/kaldi/egs/rsr_system_transfree/ --trails_path /home/tianchi/database/RSR2015/key/
```
And the path in this command should be changed according to your path before running.

## Demo

### Demo Data

We prepared feature data in '.pickle' format for running the demo. However, because RSR2015 is a private dataset, we can not provide the data or extracted feature. Please kindly follow the folder structure above and install kaldi and kaldi-io to extract feature and run the training steps.

## Note
The results in the paper are evaluated in Python2 environment with torch==0.4. However, as its support is ending, we have converted codes to Python3 platform for future use and hence there may be a minor changes. 

## License:
The codes in this repository are licensed under the GNU General Public License Version 3. For commercial use of this code and models, separate commercial licensing is also available. 



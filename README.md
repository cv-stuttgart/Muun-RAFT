# Muun-RAFT

**[WACV 2026] Reviving Unsupervised Optical Flow: Concept Reevaluation, Multi-Scale Advances and Full Open-Source Release**

Authors: Azin Jahedi, Marc Rivinius, Noah Berenguel Senn, Andrés Bruhn

Paper: [[WACV 2026 Paper]][paper] [[WACV Oral Presentation]][presentation]


This is our <u>Mu</u>lti-Scale <u>Un</u>supervised <u>RAFT</u> `Muun-RAFT` 🌑 repository.

[paper]: https://openaccess.thecvf.com/content/WACV2026/html/Jahedi_Reviving_Unsupervised_Optical_Flow_Concept_Reevaluation_Multi-Scale_Advances_and_Full_WACV_2026_paper.html
[presentation]: https://youtube.com/watch?v=ef6lb0CdkZE

For our <u>S</u>imple <u>Un</u>supervised <u>RAFT</u> (`Sun-RAFT` ☀️) repository check out:
[https://github.com/cv-stuttgart/Sun-RAFT](https://github.com/cv-stuttgart/Sun-RAFT)



## Requirements

The code has been tested with PyTorch 1.10.2+cu113. Using other Cuda drivers may lead to slight differences in the results.
Install the required dependencies via
```
pip install -r requirements.txt
```
To compile the CUDA correlation module run the following once:
```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```


## Pre-Trained Checkpoints

You can find our trained models under [Releases](https://github.com/cv-stuttgart/Muun-RAFT/releases/tag/v1.0.0).



## Evaluation

To evaluate our pre-trained checkpoints please run the following:

```Shell
python evaluate.py --model models/Muun-RAFT_sintel-test.pth --dataset sintel 
```


## Training

To train the network on Chairs and then on Sintel (sequentially) please run:

```Shell
python train_un.py --config config/Muun-RAFT_chairs-sintel.json
```

and to train the network on Chairs and then on KITTI (sequentially) please run:

```Shell
python train_un.py --config config/Muun-RAFT_chairs-kitti.json
```

The log files and checkpoints will be saved under the `checkpoints` folder.
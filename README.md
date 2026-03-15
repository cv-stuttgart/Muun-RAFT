# Muun-RAFT

**[WACV 2026] Reviving Unsupervised Optical Flow: Concept Reevaluation, Multi-Scale Advances and Full Open-Source Release**

Authors: Azin Jahedi, Marc Rivinius, Noah Berenguel Senn, Andrés Bruhn

Paper: [[WACV 2026 Paper]][paper] [[WACV Oral Presentation]][presentation]


This is our <u>Mu</u>lti-Scale <u>Un</u>supervised <u>RAFT</u> `Muun-RAFT` 🌑 repository.

[paper]: https://openaccess.thecvf.com/content/WACV2026/html/Jahedi_Reviving_Unsupervised_Optical_Flow_Concept_Reevaluation_Multi-Scale_Advances_and_Full_WACV_2026_paper.html
[presentation]: https://youtube.com/watch?v=ef6lb0CdkZE

For our <u>S</u>imple <u>Un</u>supervised <u>RAFT</u> (`Sun-RAFT` ☀️) repository check out:
[https://github.com/cv-stuttgart/Sun-RAFT](https://github.com/cv-stuttgart/Sun-RAFT)


If you find our work useful please [cite via BibTeX](CITATIONS.bib).

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

## License
- Our code is licensed under the BSD 3-Clause **No Military** License. See [LICENSE](LICENSE).
- The provided checkpoints are under the [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) license.

## Acknowledgement
Parts of this repository are adapted from [Unflow](https://github.com/simonmeister/UnFlow) ([MIT license](https://github.com/simonmeister/UnFlow/blob/master/LICENSE)), [SMURF](https://github.com/google-research/google-research/tree/master/smurf) ([Apache 2.0 license](https://github.com/google-research/google-research/blob/master/LICENSE)), [RAFT](https://github.com/princeton-vl/RAFT) ([MIT license](https://github.com/princeton-vl/RAFT/blob/master/LICENSE)), and [CCMR](https://github.com/cv-stuttgart/CCMR) ([license](https://github.com/cv-stuttgart/CCMR/blob/main/LICENSE)).
We thank the authors.
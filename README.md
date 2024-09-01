# ser-uncertainty-quantification
Evaluating Uncertainty Quantification Approaches on Real-world Speech Emotion Recognition Tasks.


This repository contains all the code to reproduce the findings from the paper:
[Are you sure? Analysing Uncertainty Quantification Approaches for Real-world
Speech Emotion Recognition](https://arxiv.org/abs/2407.01143)

## Getting Started


```bash
$ pip install -r requirements.txt
```

This project uses [hydra](https://hydra.cc/) to store the configuration of different configurations.

You can find the configurations under ```src\conf\experiments\[experiment name]```

e.g. to run the experiments for the baseline based on Cross Entropy run:

```bash
$ cd src
$ python3 main.py +experiments=emo_cat_ce
```

Note: We do not have the rights to redistribute the
[MSP Podcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) data set ourselves,
therefore you must obtain it separately or exclude it from the training/test data.

## License

This repository may only be used for non-commercial purposes 
([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).

## Citation

```bibtex
@inproceedings{schrufer24_interspeech,
  title     = {Are you sure? Analysing Uncertainty Quantification Approaches for Real-world Speech Emotion Recognition},
  author    = {Oliver Schrüfer and Manuel Milling and Felix Burkhardt and Florian Eyben and Björn Schuller},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {3210--3214},
  doi       = {10.21437/Interspeech.2024-977},
}
```
# Fine-Tuning Deep Learning Model for Detecting Indonesian Traffic Accident Tweet

This repository is the official implementation of the conference paper titled  [Fine-Tuning Deep Learning Model for Detecting Indonesian Traffic Accident Tweet](https://ieeexplore.ieee.org/document/10331034). The paper was presented at the 6th International Conference of Computer and Informatics Engineering (IC2IE) in 2023.

> Authors: Rizky Adi, Fumiyo Fukumoto, Bassamtiano Renaufalgi Irnawan

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python trainer_finetune.py --model <pre-trained model name> --learning_rate <learning rate> --batch_size <batch size> --max_length <input token max length> --cnn <is using pre-trained model + cnn architecture>
```

> You can select between these pre-trained model name [IndoBERT, IndoBERTweet, IndoRoBERTa_OSCAR, IndoRoBERTa_Wiki].

## Pre-trained Models

These are the pre-trained models we used in our experiments:

- [IndoBERT](https://huggingface.co/indolem/indobert-base-uncased) by Koto et al. 
- [IndoBERTweet](https://huggingface.co/indolem/indobertweet-base-uncased) by Koto et al. 
- [Indonesian RoBERTa](https://huggingface.co/indolem/indobertweet-base-uncased) by Wongso et al.
- [RoBERTa Indonesian](https://huggingface.co/cahya/roberta-base-indonesian-522M) by Cahya.

> Big thanks to these people who provide the pre-trained models weight for free.

## Results

Our model achieves the following performance :

| Model         | Accuracy  | F1-Score |
| ------------------ |---------------- | -------------- |
| Vanilla IndoBERT |  92%  | 88.73% |
| Vanilla IndoBERTweet |  91.5%  | 88.11% |
| Vanilla Indonesian RoBERTa (OSCAR) |  92%  | 88.73% |
| Vanilla RoBERTa Indonesia (Wiki) |  87.5%  | 82.76% |
| IndoBERT + CNN |  91%  | 82.76 |
| **IndoBERTweet + CNN** |  **93%**  | **90.67%** |
| Indonesian RoBERTa (OSCAR) + CNN |  88.5%  | 84.56% |
| RoBERTa Indonesia (Wiki) + CNN |  90%  | 85.92% |

> Please refer to our paper for the details 


## Contributing

Please cite [the following paper](https://ieeexplore.ieee.org/document/10331034) if you use our work.

```
@INPROCEEDINGS{10331034,
  author={Adi, Rizky and Fukumoto, Fumiyo and Irnawan, Bassamtiano Renaufalgi},
  booktitle={2023 6th International Conference of Computer and Informatics Engineering (IC2IE)}, 
  title={Fine-Tuning Deep Learning Model for Detecting Indonesian Traffic Accident Tweet}, 
  year={2023},
  volume={},
  number={},
  pages={19-24},
  doi={10.1109/IC2IE60547.2023.10331034}}
```

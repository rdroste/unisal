# Unified Image and Video Saliency Prediction


This repository provides the code for the paper *Unified Image and Video Saliency Prediction* by R. Droste, J. Jiao and J. Alison Noble.

[https://arxiv.org/abs/2003.05477](https://arxiv.org/abs/2003.05477)

[YouTube](https://www.youtube.com/watch?v=4CqMPDI6BqE)

```bibtex
@ARTICLE{drostejiao2020,
       author = {{Jiao}, Jianbo and {Droste}, Richard and {Noble}, J. Alison},
        title = "{Unified Image and Video Saliency Modeling}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning},
         year = 2020,
        month = mar,
          eid = {arXiv:2003.05477},
        pages = {arXiv:2003.05477},
archivePrefix = {arXiv},
       eprint = {2003.05477},
 primaryClass = {cs.CV}
}
```

---
<img src="https://github.com/rdroste/unisal/blob/master/figures/teaser.png" alt="Performance overview" width="50%">

Comparison of UNISAL with current state-of-the-art methods on the DHF1K benchmark

---

<img src="https://github.com/rdroste/unisal/blob/master/figures/architecture_a.png" alt="Method" width="100%">

UNISAL method overview

---
## Dependencies

To install the dependencies into a new conda environment, simpy run:
```bash
conda env create -f environment.yml
source activate unisal
```

Alternatively, you can install them manually:
```bash
conda create --name unisal
source activate unisal
conda install pytorch=1.0 torchvision cudatoolkit=9.2 -c pytorch
conda install opencv=3.4 -c conda-forge
conda install scipy
pip install fire==0.2 tensorboardX==1.6
```

---
## Demo
We provide demo code that generates saliency predictions for example files from the DHF1K, Hollywood-2, UCF-Sports, SALICON and MIT1003 datasets.
The predictions are generated with the pretrained weights in *training_runs/pretrained_unisal*.  
Follow these steps:

1. Download the example [.zip file](https://drive.google.com/file/d/1Rbz5rnyKlUhLhw8Ko9tgZDU1rCtGYRFL/view?usp=sharing) or [.tar.gz file](https://drive.google.com/file/d/1FM4tT0lkaNqFzrWV4N4fNnzfPuUormXD/view?usp=sharing) and extract the contents into the `examples` of the repository folder

2. Generate the demo predictions for the examples by running
`python run.py predict_examples`

The predictions are written to `saliency` sub-directories to the examples folders.

---
## Training, scoring and test set predictions

The code for training and scoring the model and to generate test set predictions is included.

### Data

For training and test set predictions, the relevant datasets need to be downloaded.
* DHF1K, Hollywood-2 and UCF Sports:   
https://github.com/wenguanwang/DHF1K

* SALICON:   
http://salicon.net/challenge-2017/

* MIT1003:   
http://people.csail.mit.edu/tjudd/WherePeopleLook/index.html

* MIT300   
http://saliency.mit.edu/results_mit300.html 

Specify the paths of the downloaded datasets with the environment variables
`DHF1K_DATA_DIR`,
`SALICON_DATA_DIR`,
`HOLLYWOOD_DATA_DIR`,
`UCFSPORTS_DATA_DIR`
`MIT300_DATA_DIR`,
`MIT1003_DATA_DIR`.


### Training

To train the model, simpy run: 
```bash
python run.py train
```

By default, this function computes the scores of the DHF1K and SALICON validation sets
and the Hollywood-2 and UCF Sports test sets after the training is finished.
The training data and scores are saved in the `training_runs` folder.
Alternatively, the training path can be overwritten with the environment variable `TRAIN_DIR`. 

### Scoring
Any trained model can be scored with:
```bash
python run.py score_model --train_id <name of training folder>
```
If `--train_id` is omitted, the provided pre-trained model is scored.
The scores are saved in the corresponding training folder.


### Test set predictions
To generate predictions for the test set of each datasets follow these steps: 

1. Specify the directory where the predictions should be saved with the environment variable `PRED_DIR`.

2. Generate the predictions by running `python run.py generate_predictions --train_id <name of training folder>`

If `--train_id` is omitted, predictions of the provided pretrained model are generated.

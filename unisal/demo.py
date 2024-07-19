from pathlib import Path
import os
from . import train
from . import data

TRAINER_ZOO = {}
PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))


def load_trainer(train_id: str = "pretrained_unisal"):
    """Instantiate Trainer class from saved kwargs."""
    train_dir = Path(os.environ["TRAIN_DIR"])
    train_dir = train_dir / train_id
    print(f"initalizing trainer from {train_dir}...")
    return train.Trainer.init_from_cfg_dir(train_dir)


def get_trainer(model_path: str):
    """get trainer from memory if already loaded, else load it"""
    global TRAINER_ZOO
    if model_path not in TRAINER_ZOO.keys():
        trainer = load_trainer()
        trainer.model.load_weights_from_path(model_path)
        TRAINER_ZOO[model_path] = trainer
    return TRAINER_ZOO[model_path]


def predict_image(
    img_rgb,
    model_path: str = os.path.join(
        PROJECT_DIR, "../training_runs/pretrained_unisal/weights_best.pth"
    ),
):
    trainer = get_trainer(model_path)
    pred_seq = trainer.inference(img_rgb=img_rgb)
    smap = data.smap_postprocess(pred_seq[:, 0, ...])
    return smap

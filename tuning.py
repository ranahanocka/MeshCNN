from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from train_pl import *
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback


args = PLOptions().parse()

def train_segmentation(config):
    # args = PLOptions().parse()
    args.num_aug = config.get('num_aug')
    model = MeshSegmenter(args)
    callback_tune = TuneReportCallback(metrics='val_iou', on="validation_end")
    callback_lightning = ModelCheckpoint(monitor='val_iou', mode='max', save_top_k=3,
                                         filename='{epoch:02d}-{val_acc_epoch:.2f}', )
    trainer = Trainer.from_argparse_args(args, callbacks=[callback_tune, callback_lightning])
    trainer.fit(model)


if __name__== '__main__':
    # Execute the hyperparameter search

    config = {
        'num_aug': tune.choice([10, 30])
    }

    # analysis = tune.run(
    #     train_segmentation,
    #     config=config, num_samples=1, resources_per_trial={"cpu": 1}, mode='max')

    analysis = tune.run(
        tune.with_parameters(train_segmentation),
        config=config, num_samples=1, resources_per_trial={"gpu": 1, 'cpu': 1})
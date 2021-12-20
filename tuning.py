import json

import torch.cuda
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from train_pl import *
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

args = PLOptions().parse()

def train_segmentation(config):
    for k, v in config.items():
        args.__dict__[k] = v
    
    model = MeshSegmenter(args)
    callback_tune = TuneReportCallback(metrics='val_iou', on="validation_end")
    callback_lightning = ModelCheckpoint(monitor='val_iou', mode='max', save_top_k=3,
                                         filename='{epoch:02d}-{val_acc_epoch:.2f}', )

    # callback_tune_f1 = TuneReportCallback(metrics='val_f1', on="validation_end")
    # callback_lightning_f1 = ModelCheckpoint(monitor='val_f1', mode='max', save_top_k=3,
    #                                      filename='{epoch:02d}-{val_acc_epoch:.2f}', )

    trainer = Trainer.from_argparse_args(args, callbacks=[callback_tune, callback_lightning])
    trainer.fit(model)
    torch.cuda.empty_cache()


if __name__== '__main__':
    # Execute the hyperparameter search
    config = {
        'resblocks': tune.grid_search([2, 3, 4]),
        'ncf': tune.grid_search([[64, 128, 256, 512], [32, 64, 128, 256], [16, 32, 64, 128]]),
        'slide_verts': tune.grid_search([0.1, 0.2]),
        'lr': tune.grid_search([0.01, 0.001]),
        'optimizer': tune.grid_search(['adam', 'sgd', 'adamw']),
        'warmup_epochs': tune.grid_search([200, 100, 50]),
        'weight_decay': tune.grid_search([0, 0.0002]),
    }

    ## CPU only
    # analysis = tune.run(
    #     train_segmentation,
    #     config=config, num_samples=1, resources_per_trial={"cpu": 1}, mode='max')

    # GPU
    analysis = tune.run(
        tune.with_parameters(train_segmentation),
        config=config, num_samples=1, resources_per_trial={"gpu": 1, 'cpu': 1})

    # Saving the results
    best_config = analysis.get_best_config(metric='val_iou', mode="max")
    print("Best config: ", best_config)

    file = open(os.path.join(args.checkpoints_dir, 'roof_seg', 'best_config.json'), 'w')
    json.dump(best_config, file)

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    print(df)
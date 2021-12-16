import json

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from train_pl import *
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback


args = PLOptions().parse()

def train_segmentation(config):
    # args = PLOptions().parse()
    if config.get('num_aug'):
        args.num_aug = config.get('num_aug')
    if config.get('resblocks'):
        args.resblocks = config.get('resblocks')
    if config.get('ncf'):
        args.ncf = config.get('ncf')
    if config.get('slide_verts'):
        args.slide_verts = config.get('slide_verts')
    if config.get('lr'):
        args.lr = config.get('lr')
    
    model = MeshSegmenter(args)
    callback_tune = TuneReportCallback(metrics='val_iou', on="validation_end")
    callback_lightning = ModelCheckpoint(monitor='val_iou', mode='max', save_top_k=3,
                                         filename='{epoch:02d}-{val_acc_epoch:.2f}', )

    # callback_tune_f1 = TuneReportCallback(metrics='val_f1', on="validation_end")
    # callback_lightning_f1 = ModelCheckpoint(monitor='val_f1', mode='max', save_top_k=3,
    #                                      filename='{epoch:02d}-{val_acc_epoch:.2f}', )

    trainer = Trainer.from_argparse_args(args, callbacks=[callback_tune, callback_lightning])
    trainer.fit(model)


if __name__== '__main__':
    # Execute the hyperparameter search

    config = {
        # 'num_aug': tune.grid_search([10, 20, 30]),
        # 'resblocks': tune.grid_search([2, 3, 4, 5]),
        'ncf': tune.grid_search([[64, 128, 256, 512], [32, 64, 128, 256]]),
        # 'slide_verts': tune.grid_search([0.08, 0.1, 0.12, 0.16, 0.2]),
        # 'lr': tune.grid_search([0.00005, 0.0002, 0.0005]) 
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
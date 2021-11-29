from .train_options import TrainOptions

class PLOptions(TrainOptions):
    def initialize(self):
        TrainOptions.initialize(self)
        parser.add_argument('--gpus', type=int, default=1)
        parser.add_argument('--max_epochs', type=int, default=60)
        parser.add_argument('--learning_rate', default=1e-3)
        parser.add_argument('--max_image_size', default=128)
        parser.add_argument('--num_classes', default=3)
        parser.add_argument('--pretrained', default=True)

        parser.add_argument('--train_data', default='../../data/windows/set_1/train')
        parser.add_argument('--test_data', default='../../data/windows/set_1/test')
        parser.add_argument('--label_file', default='../../data/windows/labels.txt')
        parser.add_argument('--train_augmentation', default=True)

        parser.add_argument('--progress_bar_refresh_rate', type=int, default=20)
        parser.add_argument('--default_root_dir', default='../../models/test_classification/densenet161/',
                            help='pytorch-lightning log path')

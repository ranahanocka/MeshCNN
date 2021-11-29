from .train_options import TrainOptions

class PLOptions(TrainOptions):
    def initialize(self):
        TrainOptions.initialize(self)
        self.parser.add_argument('--gpus', type=int, default=1)
        self.parser.add_argument('--max_epochs', type=int, default=200)
        self.parser.add_argument('--nclasses', type=int, default=2)
        self.parser.add_argument('--input_nc', type=int, default=5)

        self.parser.add_argument('--progress_bar_refresh_rate', type=int, default=20)
        self.parser.add_argument('--default_root_dir', default='checkpoints/',
                            help='pytorch-lightning log path')

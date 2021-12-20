from .train_options import TrainOptions

class PLOptions(TrainOptions):
    def initialize(self):
        TrainOptions.initialize(self)
        self.parser.add_argument('--gpus', type=int, default=1)
        self.parser.add_argument('--max_epochs', type=int, default=200)
        self.parser.add_argument('--warmup_epochs', type=int, default=50)
        self.parser.add_argument('--nclasses', type=int, default=2)
        self.parser.add_argument('--input_nc', type=int, default=5)
        self.parser.add_argument('--class_weights', nargs='+', default=[0.5, 2], type=float)
        self.parser.add_argument('--from_pretrained', type=str, default=None)
        self.parser.add_argument('--optimizer', choices=['adam', 'sgd', 'adamw'], type=str, default='adam')
        self.parser.add_argument('--weight_decay', type=float, default=0.0002)

        self.parser.add_argument('--progress_bar_refresh_rate', type=int, default=20)
        self.parser.add_argument('--default_root_dir', default='checkpoints/',
                            help='pytorch-lightning log path')

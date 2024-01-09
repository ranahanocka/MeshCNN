from abc import abstractmethod, ABC
import torch

# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn


class AbstractPointEncoding(nn.Module, ABC):
    """
    Abstract class for positional encodings.
    A point encoding is a function that maps a point in 3D space to an encoding vector.
    """

    def __init__(self, opt):
        super(AbstractPointEncoding, self).__init__()
        self.opt = opt

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Point encoding function
        :param x: point coordinates in (batch_size, 3) shape torch.Tensor
        :return: point encoding in (batch_size, out_dim) shape
        """
        pass


# Positional encoding (section 5.1)
class NerfEmbedder(AbstractPointEncoding):
    def __init__(self, opt):
        self.max_freq_log2 = opt.max_freq_log2 if hasattr(opt, "max_freq_log2") else 5
        self.num_freqs = opt.num_freqs if hasattr(opt, "num_freqs") else 2
        self.log_sampling = opt.log_sampling if hasattr(opt, "log_sampling") else True
        self.input_dims = opt.input_dims if hasattr(opt, "input_dims") else 3
        self.create_embedding_fn()
        super(NerfEmbedder, self).__init__(opt)

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if True:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = d - 1
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class PositionalEncoding3D(AbstractPointEncoding):
    """
    This class implements a positional encoding for 3D points. The encoding is inspired by the positional encoding used in the Transformer architecture.

    Attributes:
        max_freq_log2 (int): The maximum frequency in log2 scale. Default is 5.
        num_freqs (int): The number of frequency bands. Default is 2.
        log_sampling (bool): If True, the frequency bands are logarithmically spaced. Default is True.
        embed_fns (list): A list of lambda functions that apply sinusoidal and cosinusoidal transformations to the input points.
        out_dim (int): The output dimension of the positional encoding.
    """

    def __init__(self, opt):
        self.max_freq_log2 = opt.max_freq_log2 if hasattr(opt, "max_freq_log2") else 5
        self.num_freqs = opt.num_freqs if hasattr(opt, "num_freqs") else 2
        self.log_sampling = opt.log_sampling if hasattr(opt, "log_sampling") else True
        super(PositionalEncoding3D, self).__init__(opt)
        self.embed_fns = []
        out_dim = 0

        freq_bands = (
            torch.linspace(
                2.0 ** 0.0,
                2.0 ** self.max_freq_log2,
                self.num_freqs,
                dtype=torch.float64,
            )
            if not self.max_freq_log2
            else 2.0 ** torch.linspace(0.0, self.max_freq_log2, self.num_freqs)
        )

        def get_embed_fn(dim, freq, op):
            return lambda x: op(x[:, dim] * freq)

        for dim in range(3):  # 3D coordinates
            for freq in freq_bands:
                self.embed_fns.append(get_embed_fn(dim, freq, op=torch.sin))
                self.embed_fns.append(get_embed_fn(dim, freq, op=torch.cos))
                out_dim += 2

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([fn(x) for fn in self.embed_fns], 1)


class NoPointEncoding(AbstractPointEncoding):
    """
    No point encoding at all.
    """

    def __init__(self, opt):
        super(NoPointEncoding, self).__init__(opt)
        self.out_dim = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


point_encoders = {
    "positional_encoding_3d": PositionalEncoding3D,
    "no_encode": NoPointEncoding,
    "nerf_encoding": NerfEmbedder,
}


def point_encoder_fabric(opt) -> AbstractPointEncoding:
    if opt.point_encode not in point_encoders.keys():
        raise ValueError("Unknown point encoder id: {}".format(opt.point_encode))
    return point_encoders[opt.point_encode](opt)


# Example usage:
# positional_encoding = PositionalEncoding3D(max_freq_log2=5, num_freqs=10, log_sampling=True)

# Assuming your point tensor has shape (batch_size, 3) for x, y, z coordinates
# point_coordinates = torch.randn((64, 3))

# Apply positional encoding
# positional_encoded_point = positional_encoding(point_coordinates)

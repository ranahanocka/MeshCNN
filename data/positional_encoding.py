from abc import abstractmethod, ABC

import torch
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


class PositionalEncoding3D(AbstractPointEncoding):
    """
    Positional encoding for 3D points.
    This encoding is inspired by the positional encoding used in the Transformer architecture.
    """

    def __init__(self, opt):
        self.max_freq_log2 = opt.max_freq_log2 if hasattr(opt, "max_freq_log2") else 5
        self.num_freqs = opt.num_freqs if hasattr(opt, "num_freqs") else 2
        self.log_sampling = opt.log_sampling if hasattr(opt, "log_sampling") else True
        super(PositionalEncoding3D, self).__init__()
        self.embed_fns = []
        out_dim = 0

        for dim in range(3):  # 3D coordinates
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

            for freq in freq_bands:
                self.embed_fns.append(lambda x: torch.sin(x[:, dim] * freq))
                self.embed_fns.append(lambda x: torch.cos(x[:, dim] * freq))
                out_dim += 2

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([fn(x) for fn in self.embed_fns], -1)


class NoPointEncoding(AbstractPointEncoding):
    """
    No point encoding at all.
    """

    def __init__(self, opt):
        super(NoPointEncoding, self).__init__(opt)
        self.out_dim = 3

    def forward(self, x):
        return x


point_encoders = {
    "positional_encoding_3d": PositionalEncoding3D,
    "no_encode": NoPointEncoding,
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

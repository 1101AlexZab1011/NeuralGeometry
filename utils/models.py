import torch
from deepmeg.utils.params import LFCNNParameters
from deepmeg.models.interpretable import LFCNN, HilbertNet
from deepmeg.interpreters import LFCNNInterpreter
from deepmeg.experimental.models import LFCNNW, SPIRIT, FourierSPIRIT, CanonicalSPIRIT
from deepmeg.experimental.interpreters import LFCNNWInterpreter, SPIRITInterpreter
from deepmeg.experimental.params import SPIRITParameters


def get_model_by_name(
    model_name: str,
    X: torch.Tensor,
    Y: torch.Tensor,
    n_latent: int = 8,
    filter_size: int = 50,
    pool_factor: int = 10
):
    match model_name:
        case 'lfcnn':
            model = LFCNN(
                n_channels=X.shape[1],
                n_latent=n_latent,
                n_times=X.shape[-1],
                filter_size=filter_size,
                pool_factor=pool_factor,
                n_outputs=Y.shape[1]
            )
            interpretation = LFCNNInterpreter
            parametrizer = LFCNNParameters
        case 'lfcnnw':
                model = LFCNNW(
                    n_channels=X.shape[1],
                    n_latent=n_latent,
                    n_times=X.shape[-1],
                    filter_size=filter_size,
                    pool_factor=pool_factor,
                    n_outputs=Y.shape[1]
                )
                interpretation = LFCNNWInterpreter
                parametrizer = SPIRITParameters
        case 'hilbert':
            model = HilbertNet(
                n_channels=X.shape[1],
                n_latent=n_latent,
                n_times=X.shape[-1],
                filter_size=filter_size,
                pool_factor=pool_factor,
                n_outputs=Y.shape[1]
            )
            interpretation = LFCNNInterpreter
            parametrizer = LFCNNParameters
        case 'spirit':
            model = SPIRIT(
                n_channels=X.shape[1],
                n_latent=n_latent,
                n_times=X.shape[-1],
                window_size=20,
                latent_dim=10,
                filter_size=filter_size,
                pool_factor=pool_factor,
                n_outputs=Y.shape[1]
            )
            interpretation = SPIRITInterpreter
            parametrizer = SPIRITParameters
        case 'fourier':
            model = FourierSPIRIT(
                n_channels=X.shape[1],
                n_latent=n_latent,
                n_times=X.shape[-1],
                window_size=20,
                latent_dim=10,
                filter_size=filter_size,
                pool_factor=pool_factor,
                n_outputs=Y.shape[1]
            )
            interpretation = SPIRITInterpreter
            parametrizer = SPIRITParameters
        case 'canonical':
            model = CanonicalSPIRIT(
                n_channels=X.shape[1],
                n_latent=n_latent,
                n_times=X.shape[-1],
                window_size=20,
                latent_dim=10,
                filter_size=filter_size,
                pool_factor=pool_factor,
                n_outputs=Y.shape[1]
            )
            interpretation = SPIRITInterpreter
            parametrizer = SPIRITParameters
        case _:
            raise ValueError(f'Invalid model name: {model_name}')
    return model, interpretation, parametrizer
"""
rays.py
"""

from dataclasses import dataclass
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch


@dataclass(init=False)
class RayBundle:
    """
    Ray bundle class.
    """

    origins: Float[torch.Tensor, "*batch_size 3"]
    """Ray origins in the world coordinate."""
    directions: Float[torch.Tensor, "*batch_size 3"]
    """Ray directions in the world coordinate."""
    nears: Float[torch.Tensor, "*batch_size 1"]
    """Near clipping plane."""
    fars: Float[torch.Tensor, "*batch_size 1"]
    """Far clipping plane."""

    def __init__(
        self,
        origins: Float[torch.Tensor, "*batch_size 3"],
        directions: Float[torch.Tensor, "*batch_size 3"],
        nears: Float[torch.Tensor, "*batch_size 1"],
        fars: Float[torch.Tensor, "*batch_size 1"],
    ) -> None:
        """
        Initializes RayBundle.
        """
        self.origins = origins
        self.directions = directions
        self.nears = nears
        self.fars = fars

    def __len__(self) -> int:
        """
        Returns the number of rays in the bundle.
        """
        return self.origins.shape[0]

@dataclass(init=False)
class RaySamples:
    """
    Ray sample class.
    """

    ray_bundle: RayBundle
    """Ray bundle. Contains ray origin, direction, near and far bounds."""
    t_samples: Float[torch.Tensor, "num_ray num_sample"]
    """Distance values sampled along rays."""

    def __init__(
        self,
        ray_bundle: RayBundle,
        t_samples: Float[torch.Tensor, "num_ray num_sample"],
    ) -> None:
        """
        Initializes RaySample.
        """
        self.ray_bundle = ray_bundle
        self.t_samples = t_samples

    @jaxtyped
    @typechecked
    def compute_sample_coordinates(self) -> Float[torch.Tensor, "num_ray num_sample 3"]:
        """
        Computes coordinates of points sampled along rays in the ray bundle.

        Returns:
            coords: Coordinates of points sampled along rays in the ray bundle.
        """

        #origins: Float[torch.Tensor, "num_ray 3"] = self.ray_bundle.origins
        #directions: Float[torch.Tensor, "num_ray 3"] = self.ray_bundle.directions
        #t_samples: Float[torch.Tensor, "num_ray num_sample"] = self.t_samples

        # Reshape for broadcasting
        #origins = origins.unsqueeze(1)         # [num_ray, 1, 3]
        #directions = directions.unsqueeze(1)   # [num_ray, 1, 3]
        #t_samples = t_samples.unsqueeze(-1)    # [num_ray, num_sample, 1]

        # Compute sampled coordinates
        #coords = origins + t_samples * directions  # [num_ray, num_sample, 3]
        #return coords
        origins = self.ray_bundle.origins            # [num_ray, 3]
        directions = self.ray_bundle.directions      # [num_ray, 3]
        t = self.t_samples                           # [num_ray, num_sample]

        # compute the last interval per ray
        delta_last = t[..., -1] - t[..., -2]          # [num_ray]
        # new last t = last + that interval
        new_t_last = t[..., -1] + delta_last         # [num_ray]
        # append it
        t_padded = torch.cat([t, new_t_last.unsqueeze(-1)], dim=-1)  # [num_ray, num_sample+1]
        # override t_samples so compute_deltas sees this:
        self.t_samples = t_padded

        # ---- now compute coordinates ----
        # reshape for broadcasting
        o = origins.unsqueeze(1)                     # [num_ray, 1, 3]
        d = directions.unsqueeze(1)                  # [num_ray, 1, 3]
        t_exp = t_padded.unsqueeze(-1)               # [num_ray, num_sample+1, 1]

        coords = o + t_exp * d                       # [num_ray, num_sample+1, 3]
        return coords

    @jaxtyped
    @typechecked
    def compute_deltas(self, right_end: float=1e8) -> Float[torch.Tensor, "num_ray num_sample"]:
        """
        Compute differences between adjacent t's required to approximate integrals.

        Args:
            right_end: The value to be appended to the right end
                when computing 1st order difference.

        Returns:
            deltas: Differences between adjacent t's.
                When evaluating the delta for the farthest sample on a ray,
                use the value of the argument 'right_end'.
        """
        t_samples: Float[torch.Tensor, "num_ray num_sample"] = self.t_samples
        num_ray = t_samples.shape[0]
        device = t_samples.device

        deltas = torch.diff(
            t_samples,
            n=1,
            dim=-1,
            append=right_end * torch.ones((num_ray, 1), device=device),
        )

        return deltas

"""
Integrator implementing quadrature rule.
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
from torch_nerf.src.renderer.integrators.integrator_base import IntegratorBase


class QuadratureIntegrator(IntegratorBase):
    """
    Numerical integrator which approximates integral using quadrature.
    """

    @jaxtyped
    @typechecked
    def integrate_along_rays(
        self,
        sigma: Float[torch.Tensor, "num_ray num_sample"],
        radiance: Float[torch.Tensor, "num_ray num_sample 3"],
        delta: Float[torch.Tensor, "num_ray num_sample"],
    ) -> Tuple[Float[torch.Tensor, "num_ray 3"], Float[torch.Tensor, "num_ray num_sample"]]:
        """
        Computes quadrature rule to approximate integral involving in volume rendering.
        Pixel colors are computed as weighted sums of radiance values collected along rays.

        For details on the quadrature rule, refer to 'Optical models for
        direct volume rendering (IEEE Transactions on Visualization and Computer Graphics 1995)'.

        Args:
            sigma: Density values sampled along rays.
            radiance: Radiance values sampled along rays.
            delta: Distance between adjacent samples along rays.

        Returns:
            rgbs: Pixel colors computed by evaluating the volume rendering equation.
            weights: Weights used to determine the contribution of each sample to the final pixel color.
                A weight at a sample point is defined as a product of transmittance and opacity,
                where opacity (alpha) is defined as 1 - exp(-sigma * delta).
        """
        # TODO
        # HINT: Look up the documentation of 'torch.cumsum'.
        # 1) compute “optical thickness” per sample:
        #    σ_i · Δ_i
        sigma_delta = sigma * delta  # shape: [num_ray, num_sample]

        # 2) alpha_i = 1 − exp(−σ_i Δ_i)
        alpha = 1.0 - torch.exp(-sigma_delta)  # [num_ray, num_sample]

        # 3) cumulative sum of σ_j Δ_j up to (but excluding) the current i:
        #    S_i = ∑_{j=1}^{i−1} σ_j Δ_j
        #    we do this by cumsum and then shifting right by one, padding zero at front
        cum_sigma_delta = torch.cumsum(sigma_delta, dim=-1)
        # pad a column of zeros at the beginning and drop the last cum:
        zeros = torch.zeros_like(cum_sigma_delta[:, :1])
        tau = torch.cat([zeros, cum_sigma_delta[:, :-1]], dim=-1)  # [num_ray, num_sample]

        # 4) transmittance T_i = exp(−S_i)
        transmittance = torch.exp(-tau)  # [num_ray, num_sample]

        # 5) per-sample weight w_i = T_i · α_i
        weights = transmittance * alpha  # [num_ray, num_sample]

        # 6) final pixel color is weighted sum of radiance:
        #    Ĉ(r) = ∑_i w_i · c_i
        #    note: radiance has shape [num_ray, num_sample, 3]
        rgbs = torch.sum(weights.unsqueeze(-1) * radiance, dim=1)  # [num_ray, 3]

        return rgbs, weights

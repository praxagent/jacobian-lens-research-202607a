"""jlens_atlas: the layer x layer CKA map recipe behind the praxagent J-lens atlas, as a tool.

Point it at any Jacobian lens (a local jlens .pt, a Neuronpedia slug, or a file in any HF repo)
plus the model whose unembedding the lens reads out through, and it produces the same objects
the atlas note reports: the layer x layer linear-CKA map of the readout geometry D_l = U J_l,
the fixed-thirds band statistic (mid_sep), the fitted three-segmentation and its separation,
how well identified those boundaries are, the per-layer participation ratio, and a
Frobenius-matched random-transport null. Every number is written with a receipt.

Apache-2.0. Code vendored, with attribution, from the campaign repository
(jacobian-lens-research-202607a): common/cka.py, experiments/jspace_atlas/atlas_stage_a.py,
experiments/jspace_atlas/boundary_identifiability.py, experiments/jacobian_lens/cka_layers.py,
experiments/jacobian_lens/shared_vocab.py, experiments/jacobian_lens/unembed.py.
"""
__version__ = "0.1.0"
from .core import (linear_cka, cka_matrix, band_stats, fitted_seg, segmentation_scores,  # noqa: F401
                   near_optimal_spread, participation_ratio, random_transport)
from .run import run_atlas  # noqa: F401

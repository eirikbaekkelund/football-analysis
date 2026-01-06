from torchkick.sam3._mlx.backbone.utils.model import (
    get_activation_fn,
    get_clones,
    get_valid_ratio,
    gen_sineembed_for_position,
    inverse_sigmoid,
    MLP,
    Mlp,
    DropPath,
    LayerScale,
    DotProductScoring,
    MultiheadAttentionWrapper,
    TransformerWrapper,
)
from torchkick.sam3._mlx.backbone.utils.data import FindStage, interpolate

MultiHeadAttention = MultiheadAttentionWrapper
Transformer = TransformerWrapper

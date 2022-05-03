from .torch_transformer_layers import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEmbeddingLayer,
)
from .quantization import get_quant_config
from lightseq.training.ops.pytorch.builder.transformer_builder import TransformerBuilder

transformer_cuda_module = TransformerBuilder().load()

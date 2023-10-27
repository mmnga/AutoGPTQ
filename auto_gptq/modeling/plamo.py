from ._base import *
class PlamoGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "PlamoDecoderLayer"
    layers_block_name = "model.layers.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
    lm_head_name = "lm_head"

__all__ = ["PlamoGPTQForCausalLM"]

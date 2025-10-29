from .llama import LlamaForCausalLM
from .qwen2 import Qwen2ForCausalLM
from .qwen3 import Qwen3ForCausalLM

model_dict = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "Qwen2ForCausalLM": Qwen2ForCausalLM,
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
}
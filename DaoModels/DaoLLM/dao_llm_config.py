from transformers import PretrainedConfig

class DaoLLMConfig(PretrainedConfig):
    model_type = "dao_llm"

    def __init__(
        self,
        hidden_size=2048,
        num_hidden_layers=24,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        
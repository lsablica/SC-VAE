from dataclasses import dataclass
from typing import List, Optional, Union, Literal
import torch

@dataclass
class SpCauchyVAEConfig:
    """Configuration class for SpCauchyVAE hyperparameters"""
    
    # Model architecture
    input_dim: Union[int, List[int]]  # Can be scalar (flattened) or image dimensions [C,H,W]
    latent_dim: int = 20
    hidden_dims: List[int] = None
    
    # Distribution type (spcauchy or normal)
    distribution_type: Literal["spcauchy", "normal"] = "spcauchy"
    
    # Encoder/Decoder architecture type
    encoder_type: Literal["mlp", "cnn", "transformer"] = "mlp"
    decoder_type: Literal["mlp", "cnn", "transformer"] = "mlp"
    
    # CNN-specific parameters (if using CNN encoder/decoder)
    channels: List[int] = None  # Number of channels in each conv layer
    kernel_sizes: List[int] = None  # Kernel sizes for conv layers
    strides: List[int] = None  # Strides for conv layers
    
    # Transformer-specific parameters (if using transformer encoder/decoder)
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    
    vocab_size: Optional[int] = None     
    embedding_dim: Optional[int] = None   # Dimension of token embeddings
    max_seq_len: Optional[int] = None     # Maximum sequence length
    pad_token_id: Optional[int] = None    # ID of the padding token
    
    # Training parameters
    learning_rate: float = 1e-3
    beta: float = 1.0  # Weight for KL divergence term
    batch_size: int = 128
    
    # Optimization parameters
    optimizer: str = "adam"
    scheduler: Optional[str] = "cosine"
    weight_decay: float = 0.0
    
    # Device
    device: str = "cuda"
    
    # Misc
    seed: int = 42
    
    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dims,
        distribution_type="spcauchy",  # Added parameter
        encoder_type="mlp",
        decoder_type="mlp",
        is_image_input=True,
        kl_weight=1.0,
        dropout_rate=0.0,
        activation="relu",
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        vocab_size: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.distribution_type = distribution_type  # Store new parameter
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.is_image_input = is_image_input
        self.kl_weight = kl_weight
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __post_init__(self):
        # Set default values if None
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]
        
        if self.encoder_type == "cnn" and self.channels is None:
            self.channels = [32, 64, 128, 256]
            self.kernel_sizes = [3, 3, 3, 3]
            self.strides = [2, 2, 2, 2]
        

        is_transformer_for_sequence = ( (self.encoder_type == "transformer" or self.decoder_type == "transformer") and not self.is_image_input)

        if is_transformer_for_sequence:
            # Default or check embedding_dim
            if self.embedding_dim is None:
                if self.hidden_dims and len(self.hidden_dims) > 0:
                    self.embedding_dim = self.hidden_dims[0] # Default from hidden_dims
                    print(f"Info: 'embedding_dim' for transformer sequence was None, defaulting to hidden_dims[0]: {self.embedding_dim}")
                else:
                    # Or, if you prefer to always require it for sequence transformers:
                    raise ValueError("For sequence transformers, 'embedding_dim' must be provided or derivable from 'hidden_dims'.")
            
            # These are essential for sequence transformers and data-dependent, so raise error if not provided
            if self.vocab_size is None:
                raise ValueError("For sequence transformers, 'vocab_size' must be provided.")
            if self.max_seq_len is None:
                raise ValueError("For sequence transformers, 'max_seq_len' must be provided.")
            if self.pad_token_id is None:
                raise ValueError("For sequence transformers, 'pad_token_id' must be provided.")
        

        self.device = self.device if torch.cuda.is_available() and self.device == "cuda" else "cpu"
        
        # Determine if input is an image based on input_dim
        self.is_image_input = isinstance(self.input_dim, list) and len(self.input_dim) == 3
        
        # Validate distribution type
        if self.distribution_type not in ["spcauchy", "normal"]:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")

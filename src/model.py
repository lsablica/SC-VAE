import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.kl import kl_divergence_spcauchy_combined, kl_divergence_normal, kl_divergence_spcauchy

class SpCauchyVAE(nn.Module):
    """
    Hyperspherical Variational Autoencoder
    
    This VAE can use either a spherical Cauchy distribution (defined on the unit hypersphere)
    or a normal distribution for the latent space.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.is_image_input = config.is_image_input
        
        # Set distribution type (default to spherical Cauchy)
        self.distribution_type = getattr(config, 'distribution_type', 'spcauchy')
        if self.distribution_type not in ['spcauchy', 'normal']:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")
        
        # Determine input dimensions
        if self.is_image_input:
            channels, height, width = config.input_dim
            self.input_size = channels * height * width
            self.input_shape = config.input_dim
        else:
            self.input_size = config.input_dim
            self.input_shape = [config.input_dim]
        
        # Build encoder
        if config.encoder_type == "mlp":
            self.encoder = self._build_mlp_encoder()
        elif config.encoder_type == "cnn":
            self.encoder = self._build_cnn_encoder()
        elif config.encoder_type == "transformer":
            self.encoder = self._build_transformer_encoder()
        else:
            raise ValueError(f"Unsupported encoder type: {config.encoder_type}")
        
        # Latent projections
        self.fc_mu = nn.Linear(config.hidden_dims[-1], self.latent_dim)
        
        # Second parameter depends on distribution type
        if self.distribution_type == 'spcauchy':
            self.fc_second_param = nn.Linear(config.hidden_dims[-1], 1)  # rho parameter
        else:  # normal
            self.fc_second_param = nn.Linear(config.hidden_dims[-1], self.latent_dim)  # logvar parameter


        if self.distribution_type == 'spcauchy':
            with torch.no_grad():
                self.fc_second_param.bias.fill_(2.0) # start with bias that gives rho ~0.73    
        
        # Build decoder
        if config.decoder_type == "mlp":
            self.decoder = self._build_mlp_decoder()
        elif config.decoder_type == "cnn":
            self.decoder = self._build_cnn_decoder()
        elif config.decoder_type == "transformer":
            self.decoder = self._build_transformer_decoder()
        else:
            raise ValueError(f"Unsupported decoder type: {config.decoder_type}")
    
    def _build_mlp_encoder(self):
        """Build an MLP encoder network"""
        layers = []
        prev_dim = self.input_size
        
        for dim in self.config.hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            
            if self.config.activation == "relu":
                layers.append(nn.ReLU())
            elif self.config.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))
            elif self.config.activation == "tanh":
                layers.append(nn.Tanh())
                
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
                
            prev_dim = dim
        
        return nn.Sequential(*layers)
    
    def _build_cnn_encoder(self):
        """Build a CNN encoder network"""
        channels, height, width = self.input_shape
        cnn_layers = []
        
        # Layer 1
        cnn_layers.append(nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1))
        cnn_layers.append(nn.ReLU())
        height, width = height // 2, width // 2
        
        # Layer 2
        cnn_layers.append(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1))
        cnn_layers.append(nn.ReLU())
        height, width = height // 2, width // 2
        
        # Layer 3
        cnn_layers.append(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        cnn_layers.append(nn.ReLU())
        height, width = height // 2, width // 2
        
        # Flatten layer
        flatten_size = 128 * height * width
        
        # FC layers
        fc_layers = [
            nn.Flatten(),
            nn.Linear(flatten_size, self.config.hidden_dims[-1]),
            nn.ReLU()
        ]
        
        return nn.Sequential(*cnn_layers, *fc_layers)
    
    def _build_mlp_decoder(self):
        """Build an MLP decoder network"""
        layers = []
        prev_dim = self.latent_dim
        
        hidden_dims = list(reversed(self.config.hidden_dims))
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            
            if self.config.activation == "relu":
                layers.append(nn.ReLU())
            elif self.config.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))
            elif self.config.activation == "tanh":
                layers.append(nn.Tanh())
                
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
                
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, self.input_size))
        
        # For image data, add sigmoid for pixel values
        if self.is_image_input:
            layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def _build_cnn_decoder(self):
        """Build a CNN decoder network"""
        channels, height, width = self.input_shape
        

        h_start = height // 4  # 7 for MNIST
        w_start = width // 4   # 7 for MNIST
        
        # Initial FC layer to reshape
        initial_size = 128 * h_start * w_start
        
        # FC and reshape layers
        fc_layers = [
            nn.Linear(self.latent_dim, self.config.hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dims[-1], initial_size),
            nn.ReLU(),
            Reshape((-1, 128, h_start, w_start))
        ]
        
        # Transposed convolution layers with carefully chosen parameters
        conv_layers = [
            # Layer 1: 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # Layer 2: 14x14 -> 28x28
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # Layer 3: Final output layer
            nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        ]
        
        return nn.Sequential(*fc_layers, *conv_layers)
    


    def _build_transformer_encoder(self):
        """Build a transformer-style encoder for SMILES with fixed sinusoidal PEs (simplified)."""
        # Config parameters needed: vocab_size, embedding_dim, max_seq_len, pad_token_id,
        # num_heads, num_layers, dropout_rate, hidden_dims[-1]

        # Token Embedding Layer
        self.encoder_token_embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.embedding_dim,
            padding_idx=self.config.pad_token_id
        )

        target_device = torch.device(self.config.device)
        encoder_pe_tensor = get_sinusoidal_positional_encodings(
            self.config.max_seq_len,
            self.config.embedding_dim
        ).to(target_device)
        self.register_buffer('encoder_fixed_pe', encoder_pe_tensor) 

        # Transformer Encoder Block
        encoder_layer_norm = nn.TransformerEncoderLayer(
            d_model=self.config.embedding_dim,
            nhead=self.config.num_heads,
            dim_feedforward=4 * self.config.embedding_dim,
            dropout=self.config.dropout_rate,
            batch_first=True
        )
        transformer_encoder_block = nn.TransformerEncoder(
            encoder_layer_norm,
            num_layers=self.config.num_layers
        )

        # Projection Layer
        final_projection_layer = nn.Linear(
            self.config.max_seq_len * self.config.embedding_dim,
            self.config.hidden_dims[-1] 
        )
        
        dropout_module = nn.Dropout(self.config.dropout_rate)

        return TransformerEncoderWrapper( # Using the redefined wrapper
            token_embedding_layer=self.encoder_token_embedding,
            pos_embedding_param=self.encoder_fixed_pe, # Pass the buffer directly
            transformer_encoder_block=transformer_encoder_block,
            final_projection_layer=final_projection_layer,
            dropout_module=dropout_module
        )

    def _build_transformer_decoder(self):
        """Build a transformer-style decoder for SMILES with fixed sinusoidal PEs (simplified)."""
        # Config parameters needed: latent_dim, vocab_size, embedding_dim, max_seq_len,
        # num_heads, num_layers, dropout_rate

        initial_z_projection_layer = nn.Sequential(
            nn.Linear(self.latent_dim, self.config.max_seq_len * self.config.embedding_dim),
            Reshape((-1, self.config.max_seq_len, self.config.embedding_dim)) 
        )

        target_device = torch.device(self.config.device)
        decoder_pe_tensor = get_sinusoidal_positional_encodings(
            self.config.max_seq_len,
            self.config.embedding_dim
        ).to(target_device)
        self.register_buffer('decoder_fixed_pe', decoder_pe_tensor) 

        decoder_layer_norm = nn.TransformerDecoderLayer(
            d_model=self.config.embedding_dim,
            nhead=self.config.num_heads,
            dim_feedforward=4 * self.config.embedding_dim,
            dropout=self.config.dropout_rate,
            batch_first=True
        )
        transformer_decoder_block = nn.TransformerDecoder(
            decoder_layer_norm,
            num_layers=self.config.num_layers
        )

        output_to_vocab_layer = nn.Linear(
            self.config.embedding_dim,
            self.config.vocab_size
        )
        
        dropout_module = nn.Dropout(self.config.dropout_rate)

        return TransformerDecoderWrapper( # Using the redefined wrapper
            initial_z_projection_layer=initial_z_projection_layer,
            pos_embedding_param=self.decoder_fixed_pe, # Pass the buffer directly
            transformer_decoder_block=transformer_decoder_block,
            output_to_vocab_layer=output_to_vocab_layer,
            max_seq_len=self.config.max_seq_len,
            embedding_dim=self.config.embedding_dim,
            dropout_module=dropout_module
        )
    
    def encode(self, x):
        """
        Encode input to latent parameters
        For spherical Cauchy: mu (unit vector) and rho (concentration parameter)
        For normal: mu (mean) and logvar (log variance)
        """
        # Reshape input if necessary
        if self.is_image_input:
            if self.config.encoder_type == "mlp":
                x = x.view(x.size(0), -1)
            elif len(x.shape) != 4:
                x = x.view(-1, *self.input_shape)
        elif self.config.encoder_type != "transformer":
            x = x.view(x.size(0), -1)
        
        # Pass through encoder
        h = self.encoder(x)
        
        # Get latent parameters
        mu = self.fc_mu(h)
        second_param = self.fc_second_param(h)
        
        if self.distribution_type == 'spcauchy':
            # Normalize mu to unit length for spherical Cauchy
            mu = F.normalize(mu, p=2, dim=1)
            
            # Map rho to (0,1) using sigmoid
            rho = torch.sigmoid(second_param)
            return mu, rho
        else:  # normal distribution
            # No normalization needed for mu in normal distribution
            # second_param represents logvar
            return mu, second_param  # mu, logvar
    
    def reparameterize(self, mu, second_param):
        """
        Reparameterize based on distribution type
        
        For spherical Cauchy: second_param is rho (concentration)
        For normal: second_param is logvar (log variance)
        """
        if self.distribution_type == 'spcauchy':
            rho = second_param
            
            batch_size = mu.shape[0]
            
            # Sample from standard normal and normalize to get uniform on sphere
            x = torch.randn(batch_size, self.latent_dim, device=mu.device)
            x = F.normalize(x, p=2, dim=1)
            
            # Möbius transformation
            inner_prod = torch.sum(x * mu, dim=1, keepdim=True)
            rho_squared = rho**2
            one_minus_rho_squared = 1 - rho_squared
            rho_mu = rho * mu
            numerator = one_minus_rho_squared * (x + rho_mu)
            denominator = 1 + 2 * rho * inner_prod + rho_squared
            z = numerator / denominator + rho_mu
            
        else:  
            logvar = second_param
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
        return z
    
    def decode(self, z):
        """Decode latent vector z to reconstruction"""
        x_hat = self.decoder(z)
        
        if self.is_image_input and self.config.decoder_type == "mlp":
            x_hat = x_hat.view(-1, *self.input_shape)
        
        return x_hat

    def sample_prior(self, num_samples, device=None):
        """Sample latent vectors from the chosen prior distribution."""
        device = device or next(self.parameters()).device
        if self.distribution_type == 'spcauchy':
            z = torch.randn(num_samples, self.latent_dim, device=device)
            z = F.normalize(z, p=2, dim=1)
        else:
            z = torch.randn(num_samples, self.latent_dim, device=device)
        return z

    def generate_samples(self, num_samples=16, device=None):
        """Generate data samples by drawing from the prior and decoding."""
        device = device or next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            z = self.sample_prior(num_samples, device=device)
            samples = self.decode(z)
        return samples
    
    def forward(self, x):
        """Forward pass through the VAE"""
        mu, second_param = self.encode(x)
        z = self.reparameterize(mu, second_param)
        x_hat = self.decode(z)
        
        return x_hat, mu, second_param
    
    def kl_divergence(self, mu, second_param):
        """
        Compute KL divergence based on distribution type
        
        For spherical Cauchy: second_param is rho, prior is uniform on sphere
        For normal: second_param is logvar, prior is standard normal
        """
        if self.distribution_type == 'spcauchy':
            rho = second_param
            return kl_divergence_spcauchy_combined(rho, self.latent_dim)
        else:  
            logvar = second_param
            return kl_divergence_normal(mu, logvar)
    
    def loss_function(self, x, x_hat, mu, second_param):
        """Compute the ELBO loss"""
        if self.is_image_input:
            recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(0)
        else:
            effective_batch_size = x.size(0) # B
            recon_loss = F.cross_entropy(
                x_hat.reshape(-1, self.config.vocab_size),  # (B*L, V)
                x.reshape(-1),                               # (B*L)
                ignore_index=self.config.pad_token_id,   
                reduction='sum'
            ) / effective_batch_size   
        
        kl_loss = self.kl_divergence(mu, second_param).mean()

        total_loss = recon_loss + self.config.kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    
    @staticmethod
    def load_from_checkpoint(checkpoint_path, config=None, weights_only=False):
        """Load model from checkpoint.

        weights_only stays False for compatibility with existing checkpoints that
        store both config and optimizer state. Callers control the file, so this
        keeps behavior explicit.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=weights_only)
        
        if config is None and 'config' in checkpoint:
            config = checkpoint['config']
        
        model = SpCauchyVAE(config)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model


def get_sinusoidal_positional_encodings(max_seq_len, embedding_dim):

    pe = torch.zeros(max_seq_len, embedding_dim)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    if embedding_dim % 2 != 0: 
         pe[:, 1::2] = torch.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(position * div_term)   

    pe = pe.unsqueeze(0) 
    return pe

# Helper classes
class Reshape(nn.Module):
    """Reshape layer for use in Sequential"""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)

class TransformerEncoderWrapper(nn.Module):
    def __init__(self, token_embedding_layer, pos_embedding_param, 
                 transformer_encoder_block, final_projection_layer, dropout_module):
        super().__init__()
        self.token_embedding = token_embedding_layer
        self.pos_embedding = pos_embedding_param  
        self.transformer_encoder = transformer_encoder_block
        self.final_projection = final_projection_layer
        self.dropout = dropout_module 
        
        
    def forward(self, src_token_ids):
        src_padding_mask = (src_token_ids == self.token_embedding.padding_idx)
        
        token_emb = self.token_embedding(src_token_ids) 
        x = token_emb + self.pos_embedding 
        x = self.dropout(x)
        
        encoded_sequence = self.transformer_encoder(
            src=x, 
            src_key_padding_mask=src_padding_mask
        )  
        
        encoded_flat = encoded_sequence.reshape(encoded_sequence.size(0), -1) 
        projected_output = self.final_projection(encoded_flat) 
        return projected_output

class TransformerDecoderWrapper(nn.Module):
    def __init__(self, initial_z_projection_layer, pos_embedding_param,
                 transformer_decoder_block, output_to_vocab_layer, 
                 max_seq_len, embedding_dim, dropout_module): 
        super().__init__()
        self.initial_z_projection = initial_z_projection_layer
        self.pos_embedding = pos_embedding_param  
        self.transformer_decoder = transformer_decoder_block
        self.output_to_vocab_logits = output_to_vocab_layer 
        self.dropout = dropout_module 
        self.max_seq_len = max_seq_len   
        self.embedding_dim = embedding_dim 

    def forward(self, z):
        decoder_input_sequence = self.initial_z_projection(z) 
        
        x = decoder_input_sequence + self.pos_embedding
        x = self.dropout(x)
        
        batch_size = z.size(0)
        dummy_memory = torch.zeros(batch_size, self.max_seq_len, self.embedding_dim, device=z.device)
        
        output_embeddings = self.transformer_decoder(
            tgt=x, 
            memory=dummy_memory, 
            tgt_mask=None 
        )
        
        output_logits = self.output_to_vocab_logits(output_embeddings) 
        return output_logits

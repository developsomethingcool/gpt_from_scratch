import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging
import time
from torch.utils.data import DataLoader, Dataset


DEVICE = "mps"  # Options: "cpu", "cuda", "mps"


class ModelTrainer:
    """
    Model trainer for classical n-gram, neural n-gram and mini-gpt
    """
    def __init__(self, 
                 model_type: str,
                 config: Dict[str, Any],
                 experiment_dir: Optional[Path] = None):
        
        self.model_type = model_type
        self.config = config
        self.experiment_dir = experiment_dir or Path("experiments")
        self.experiment_dir.mkdir(exist_ok=True, parents=True)

        # simplified device setup with basic fallback
        self.device = self.setup_device()
        self.device_type = self.device.type
        
        # keep device optimizations for better performance
        self.setup_device_optimizations()
        self.setup_logging()
        
        self.epochs = config.get('epochs', 10)
        self.lr = config.get('learning_rate', 3e-4)
        self.batch_size = config.get('batch_size', 64)
        
        self.train_metrics = []
        self.val_metrics = []
        
        self.logger.info(f"Initialized trainer on {self.device}")

    def setup_device(self):
        """
        Simple device setup with minimal fallback logic
        """
        try:
            if DEVICE == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            elif DEVICE == "mps" and torch.backends.mps.is_available():
                return torch.device("mps") 
            else:
                print(f"Selected device {DEVICE} not available, using CPU")
                return torch.device("cpu")
        except Exception as e:
            print(f"Error setting up device: {e}, falling back to CPU")
            return torch.device("cpu")



    def log_device_info(self, message: str):
        """Log device info"""
        print(message)  
        self._device_message = message  

    def setup_device_optimizations(self):
        """
        Configure optimizations based on detected device
        """
        self.device_type = self.device.type

        if self.device_type == "mps":
            # Apple Silicon optimization
            torch.set_num_threads(8)
        elif self.device_type == "cuda":
            # NVIDIA GPU optimization
            torch.backends.cudnn.benchmark = True
        else:
            # CPU optimization - use all available cores
            torch.set_num_threads(torch.get_num_threads())

        # mixed precision setup (device-specific)
        self.use_amp = (
            self.device_type in ["cuda", "mps"] and 
            self.config.get('use_mixed_precision', False)
        )
        
        # scaler only for CUDA
        if self.use_amp and self.device_type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # efficient attention check (universal)
        self._use_efficient_attention = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        # enable optimized operations if available
        if hasattr(torch.backends, 'opt_einsum'):
            torch.backends.opt_einsum.enabled = True


    def setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.experiment_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # log the device message that was stored earlier
        if hasattr(self, '_device_message'):
            self.logger.info(self._device_message)
        
        # log optimizations
        if self.use_amp:
            self.logger.info("Mixed precision training enabled")
        if self._use_efficient_attention:
            self.logger.info("Efficient attention enabled")
        

    def train(self, model, train_data, val_data=None):
        """
        Universal training method that handles different model types
        """
        self.logger.info(f"Starting training for {self.model_type} model")
        
        if self.model_type == "ngram":
            return self._train_ngram(model, train_data, val_data)
        else:
            return self._train_neural(model, train_data, val_data)


    def _train_ngram(self, model, train_data, val_data=None):
        """Train classical n-gram model (device-independent)"""
        start_time = time.time()
        
        self.logger.info("Training classical n-gram model...")
        model.update(train_data)
        
        metrics = {}
        if val_data:
            metrics["val_perplexity"] = model.perplexity(val_data)
            self.logger.info(f"Validation perplexity: {metrics['val_perplexity']:.3f}")
        
        training_time = time.time() - start_time
        self.logger.info(f"N-gram training completed in {training_time:.2f} seconds")
        
        return model, metrics


    def _train_neural(self, model, train_data, val_data=None):
        """
        Universal neural model training 
        """
         # move model to device
        model = model.to(self.device)
        
        # model compilation if available
        if hasattr(torch, 'compile') and self.device_type in ["cuda", "mps"]:
            try:
                model = torch.compile(model)
                self.logger.info("Model compiled for better performance")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
        
        # setup optimizer
        optimizer = self._setup_optimizer(model)
        
        # create data loaders
        train_loader = self._prepare_data(train_data)
        val_loader = self._prepare_data(val_data) if val_data else None
        
        # training loop
        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}")
            
            # train one epoch
            train_metrics = self._train_epoch(model, optimizer, train_loader)
            self.train_metrics.append(train_metrics)
            
            # validate
            if val_loader:
                val_metrics = self._validate(model, val_loader)
                self.val_metrics.append(val_metrics)
                self.logger.info(
                    f"Epoch {epoch+1}: train_loss={train_metrics['loss']:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_ppl={val_metrics['perplexity']:.2f}"
                )
            else:
                self.logger.info(f"Epoch {epoch+1}: train_loss={train_metrics['loss']:.4f}")
        
        return model, {"train_metrics": self.train_metrics, "val_metrics": self.val_metrics}
    

    def _setup_optimizer(self, model):
        """Setup optimizer with device-specific settings"""
        base_params = {
            'lr': self.lr,
            'betas': (0.9, 0.95),
            'weight_decay': 0.1,
            'eps': 1e-8
        }
        
        # device-specific optimizer tweaks
        if self.device_type == "mps":
            # for Apple Silicon 
            base_params['betas'] = (0.9, 0.95)
        elif self.device_type == "cpu":
            # for CPU training 
            base_params['eps'] = 1e-7
        
        return optim.AdamW(model.parameters(), **base_params)

    def _train_epoch(self, model, optimizer, data_loader):
        """
        Universal training epoch
        """
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(data_loader):
            # prepare batch for current device
            inputs, targets = self._prepare_batch(batch)
            
            # forward and backward pass with mixed precision support
            if self.use_amp and self.device_type == "cuda":
                # CUDA mixed precision
                with torch.cuda.amp.autocast():
                    logits = model(inputs)
                    loss = self._compute_loss(logits, targets, model.vocab_size)
                
                self.scaler.scale(loss).backward()
                if self.config.get('clip_grad', True):
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config.get('max_grad_norm', 1.0)
                    )
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
            else:
                # standard training 
                logits = model(inputs)
                loss = self._compute_loss(logits, targets, model.vocab_size)
                
                optimizer.zero_grad()
                loss.backward()
                
                if self.config.get('clip_grad', True):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config.get('max_grad_norm', 1.0)
                    )
                
                optimizer.step()
            
            total_loss += loss.item()
            
            # progress logging
            if batch_idx % 100 == 0:
                self.logger.debug(f"Batch {batch_idx}/{len(data_loader)}: loss={loss.item():.4f}")
        
        return {"loss": total_loss / len(data_loader)}


    def _validate(self, model, data_loader):
        """
        Universal validation method
        """
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = self._prepare_batch(batch)
                logits = model(inputs)
                loss = self._compute_loss(logits, targets, model.vocab_size)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        return {"loss": avg_loss, "perplexity": torch.exp(torch.tensor(avg_loss)).item()}
    

    def _prepare_batch(self, batch):
        """
        Universal batch preparation 
        """
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch
        else:
            # handle different data formats
            inputs = batch[:-1]
            targets = batch[1:]
        
        # move to device efficiently
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        
        return inputs, targets

    def _compute_loss(self, logits, targets, vocab_size):
        """Universal loss computation"""
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits.view(-1, vocab_size), targets.view(-1))

    def _prepare_data(self, data):
        """Prepare data for training"""
        if self.model_type == "ngram":
            # N-gram models accept token sequences directly
            return data
        elif isinstance(data, DataLoader):
            # already prepared
            return data
        elif isinstance(data, torch.Tensor):
            # create tensor dataset with sliding window for autoregressive training
            seq_length = self.config.get("context_size", 128)  
            inputs = data[:, :-1]  # All tokens except last
            targets = data[:, 1:]   # All tokens except first
            
            dataset = torch.utils.data.TensorDataset(inputs, targets)
            return DataLoader(
                dataset,
                batch_size=self.batch_size, 
                shuffle=True,
                pin_memory=(self.device_type != "cpu")
            )
        elif isinstance(data, list):
            # convert token list to tensor
            data_tensor = torch.tensor(data, dtype=torch.long)
            # then convert to DataLoader
            return self._prepare_data(data_tensor)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

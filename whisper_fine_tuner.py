import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import os
from tqdm import tqdm

class WhisperFineTuner:
    def __init__(
        self,
        model_name="openai/whisper-medium",
        language="kannada",
        learning_rate=1e-5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=True
    ):
        """Initialize the WhisperFineTuner with model and training configurations."""
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.device = device
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision and device == "cuda" else None
        
        # Move model to device
        self.model.to(device)
        
        # Set language and task for forced decoder ids
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language,
            task="transcribe"
        )
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
    
    def train_step(self, batch, gradient_accumulation_steps=1):
        """Perform a single training step."""
        input_features = batch["input_features"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Use mixed precision training if enabled
        with torch.cuda.amp.autocast(enabled=self.mixed_precision and self.device == "cuda"):
            outputs = self.model(
                input_features=input_features,
                attention_mask=attention_mask,
                return_dict=True
            )
            loss = outputs.loss / gradient_accumulation_steps
        
        # Backward pass with gradient scaling if using mixed precision
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * gradient_accumulation_steps
    
    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        num_epochs=10,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        save_dir="checkpoints",
        log_interval=100,
        use_wandb=True
    ):
        """Train the model with the given configuration."""
        if use_wandb:
            wandb.init(project="whisper-kannada", name="fine-tuning")
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize scheduler
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_dataloader) * num_epochs
        )
        
        best_val_loss = float('inf')
        global_step = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Skip None batches (can happen due to audio loading errors)
                if batch is None:
                    continue
                    
                loss = self.train_step(batch, gradient_accumulation_steps)
                train_loss += loss
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_grad_norm
                    )
                    
                    # Optimizer step with gradient scaling if using mixed precision
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Log metrics
                    if use_wandb and global_step % log_interval == 0:
                        wandb.log({
                            "train_loss": loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "global_step": global_step
                        })
                    
                    global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'train_loss': f'{train_loss/(batch_idx+1):.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
            
            # Validation phase
            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                print(f"Validation loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(save_dir, "best_model"),
                        epoch,
                        val_loss
                    )
                
                if use_wandb:
                    wandb.log({
                        "val_loss": val_loss,
                        "epoch": epoch
                    })
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f"checkpoint-epoch-{epoch+1}"),
                    epoch,
                    val_loss if val_dataloader is not None else train_loss/(batch_idx+1)
                )
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate the model on the validation set."""
        self.model.eval()
        total_loss = 0
        valid_batches = 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Skip None batches
            if batch is None:
                continue
                
            input_features = batch["input_features"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.mixed_precision and self.device == "cuda"):
                outputs = self.model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                loss = outputs.loss
            
            total_loss += loss.item()
            valid_batches += 1
        
        return total_loss / valid_batches if valid_batches > 0 else float('inf')
    
    def save_checkpoint(self, path, epoch, loss):
        """Save a checkpoint of the model."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'processor_config': self.processor.save_pretrained(path)
        }
        
        # Save the model and processor
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        
        # Save additional training state
        torch.save(checkpoint, os.path.join(path, 'training_state.pt'))
    
    def load_checkpoint(self, path):
        """Load a checkpoint of the model."""
        # Load training state
        checkpoint = torch.load(os.path.join(path, 'training_state.pt'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['loss']
    
    @torch.no_grad()
    def transcribe(self, audio_features):
        """Transcribe audio using the fine-tuned model."""
        self.model.eval()
        input_features = audio_features.to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.mixed_precision and self.device == "cuda"):
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )
        
        return transcription

if __name__ == "__main__":
    # Example usage
    fine_tuner = WhisperFineTuner(
        model_name="openai/whisper-small",
        language="kannada",
        learning_rate=1e-5,
        mixed_precision=True
    )
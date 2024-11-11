import argparse
import torch
from transformers import WhisperProcessor
from efficient_loader import MemoryEfficientDataset, create_dataloader
from whisper_fine_tuner import WhisperFineTuner
import os
from torch.utils.data import random_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Whisper model for Kannada ASR")
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/audio",
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-small",
        help="Whisper model to use"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use Weights & Biases for logging"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Whether to use mixed precision training"
    )
    parser.add_argument(
        "--max_duration",
        type=int,
        default=30,
        help="Maximum audio duration in seconds"
    )
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Get absolute path for audio directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(project_dir, args.audio_dir)
    
    # Print directory information
    logger.info(f"Project directory: {project_dir}")
    logger.info(f"Audio directory: {audio_dir}")
    
    # Verify audio directory exists
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    # Print audio files count
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3', '.m4a'))]
    logger.info(f"\nFound {len(audio_files)} audio files in {audio_dir}")
    if audio_files:
        logger.info("First few audio files:")
        for file in audio_files[:5]:
            logger.info(f"- {file}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nUsing device: {device}")
    
    # Initialize Whisper processor
    logger.info(f"\nLoading Whisper processor: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    
    # Create dataset
    logger.info("\nCreating dataset...")
    try:
        dataset = MemoryEfficientDataset(
            audio_dir=audio_dir,  # Use absolute path
            processor=processor,
            max_duration=args.max_duration,
            sample_rate=16000,
            use_fp16=args.mixed_precision
        )
        logger.info(f"Dataset created successfully with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        raise
    
    # Split dataset
    logger.info(f"\nSplitting dataset with validation ratio: {args.val_split}")
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )
    logger.info(f"Train size: {train_size}, Validation size: {val_size}")
    
    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    logger.info("Dataloaders created successfully")
    
    # Initialize fine-tuner
    logger.info("\nInitializing WhisperFineTuner...")
    fine_tuner = WhisperFineTuner(
        model_name=args.model_name,
        language="kannada",
        learning_rate=args.learning_rate,
        device=device,
        mixed_precision=args.mixed_precision
    )
    
    # Create save directory if it doesn't exist
    save_dir = os.path.join(project_dir, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {save_dir}")
    
    # Start training
    logger.info("\nStarting training...")
    logger.info(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    try:
        fine_tuner.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=args.epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_dir=save_dir,
            use_wandb=args.use_wandb
        )
        logger.info("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user. Saving checkpoint...")
        fine_tuner.save_checkpoint(
            os.path.join(save_dir, "interrupted_checkpoint"),
            epoch=-1,
            loss=-1
        )
        logger.info("Checkpoint saved!")
    
    except Exception as e:
        logger.error(f"\nAn error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise
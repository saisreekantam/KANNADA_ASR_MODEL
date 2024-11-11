import logging
import sys
from transformers import WhisperProcessor
from efficient_loader import MemoryEfficientDataset, create_dataloader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def test_dataset():
    try:
        # Load processor
        logger.info("Loading Whisper processor...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        
        # Create dataset
        logger.info("Creating dataset...")
        dataset = MemoryEfficientDataset(
            audio_dir="data/audio",
            processor=processor,
            max_duration=30,
            sample_rate=16000,
            use_fp16=True
        )
        
        logger.info(f"Dataset created with {len(dataset)} samples")
        
        # Test loading first item
        logger.info("Testing first item load...")
        first_item = dataset[0]
        if first_item is not None:
            logger.info("Successfully loaded first item")
            logger.info(f"Input features shape: {first_item['input_features'].shape}")
            logger.info(f"Attention mask shape: {first_item['attention_mask'].shape}")
        else:
            logger.error("Failed to load first item")
        
        # Test dataloader
        logger.info("\nTesting dataloader...")
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=True)
        
        # Try loading first batch
        logger.info("Loading first batch...")
        first_batch = next(iter(dataloader))
        if first_batch is not None:
            logger.info("Successfully loaded first batch")
            logger.info(f"Batch input features shape: {first_batch['input_features'].shape}")
            logger.info(f"Batch attention mask shape: {first_batch['attention_mask'].shape}")
        else:
            logger.error("Failed to load first batch")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    test_dataset()
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_checkpoints():
    """Find and list all checkpoint directories"""
    # Check current directory
    current_dir = Path.cwd()
    logger.info(f"Current directory: {current_dir}")
    
    # Look for checkpoint directories
    checkpoint_dirs = []
    for root, dirs, files in os.walk(current_dir):
        for dir_name in dirs:
            if 'checkpoint' in dir_name.lower():
                checkpoint_path = Path(root) / dir_name
                checkpoint_dirs.append(checkpoint_path)
                
                # Check contents
                logger.info(f"\nFound checkpoint directory: {checkpoint_path}")
                logger.info("Contents:")
                for item in os.listdir(checkpoint_path):
                    size = os.path.getsize(checkpoint_path / item) / (1024 * 1024)  # Convert to MB
                    logger.info(f"- {item} ({size:.2f} MB)")

    return checkpoint_dirs

if __name__ == "__main__":
    print("\n=== Looking for Model Checkpoints ===\n")
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("\nNo checkpoint directories found!")
        print("Please ensure your model was saved during training.")
    else:
        print(f"\nFound {len(checkpoints)} checkpoint directories.")
        print("You can use these paths in the test script.")
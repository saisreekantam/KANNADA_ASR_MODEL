import os
import shutil
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def organize_audio_files():
    """Organize audio files by moving them to the correct directory"""
    try:
        # Get current directory and project paths
        current_dir = os.getcwd()
        project_dir = os.path.join(current_dir, "whisper_project")
        data_dir = os.path.join(current_dir, "data")
        target_audio_dir = os.path.join(data_dir, "audio")
        
        logger.info(f"Current directory: {current_dir}")
        
        # Create target directory if it doesn't exist
        os.makedirs(target_audio_dir, exist_ok=True)
        logger.info(f"Created/verified audio directory: {target_audio_dir}")
        
        # Find all audio files in the project
        audio_files_found = []
        audio_extensions = ('.mp3', '.wav', '.m4a')
        
        # Search in project directory
        for root, _, files in os.walk(current_dir):
            for file in files:
                if file.lower().endswith(audio_extensions):
                    source_path = os.path.join(root, file)
                    audio_files_found.append(source_path)
        
        if not audio_files_found:
            logger.error("No audio files found in the project directory!")
            return
        
        logger.info(f"\nFound {len(audio_files_found)} audio files:")
        for file in audio_files_found:
            logger.info(f"- {os.path.basename(file)}")
        
        # Move files to target directory
        moved_files = 0
        failed_files = []
        
        for source_path in audio_files_found:
            try:
                filename = os.path.basename(source_path)
                target_path = os.path.join(target_audio_dir, filename)
                
                # Check if file already exists in target
                if os.path.exists(target_path):
                    logger.info(f"File already exists in target: {filename}")
                    continue
                
                # Move file
                shutil.move(source_path, target_path)
                moved_files += 1
                logger.info(f"✅ Moved: {filename}")
                
            except Exception as e:
                logger.error(f"❌ Failed to move {filename}: {str(e)}")
                failed_files.append(filename)
        
        # Print summary
        logger.info("\n=== Organization Summary ===")
        logger.info(f"Total files found: {len(audio_files_found)}")
        logger.info(f"Files moved: {moved_files}")
        logger.info(f"Failed to move: {len(failed_files)}")
        
        if failed_files:
            logger.info("\nFailed files:")
            for file in failed_files:
                logger.info(f"- {file}")
        
        logger.info(f"\nAll audio files should now be in: {target_audio_dir}")
        
        # Verify final state
        final_files = [f for f in os.listdir(target_audio_dir) 
                      if f.lower().endswith(audio_extensions)]
        logger.info(f"\nFiles in target directory: {len(final_files)}")
        
    except Exception as e:
        logger.error(f"Error during organization: {str(e)}", exc_info=True)

if __name__ == "__main__":
    print("\n=== Starting Audio File Organization ===\n")
    organize_audio_files()
    print("\nOrganization complete!")

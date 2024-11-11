import torch
import torchaudio
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioVerifier:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.required_sample_rate = 16000
        
    def verify_file(self, file_path):
        """Verify a single audio file"""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(file_path)
            duration = waveform.shape[1] / sample_rate
            
            return {
                "valid": True,
                "sample_rate": sample_rate,
                "channels": waveform.shape[0],
                "duration": duration,
                "samples": waveform.shape[1],
                "needs_conversion": sample_rate != self.required_sample_rate or waveform.shape[0] > 1
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }

def verify_audio_files():
    """Verify all audio files in the directory"""
    try:
        # Setup paths
        audio_dir = os.path.join(os.getcwd(), "data", "audio")
        logger.info(f"Checking audio files in: {audio_dir}")
        
        # Initialize verifier
        verifier = AudioVerifier()
        
        # Get all audio files
        audio_files = [f for f in os.listdir(audio_dir) 
                      if f.endswith(('.mp3', '.wav', '.m4a'))]
        
        if not audio_files:
            logger.error("No audio files found!")
            return
        
        # Verify each file
        results = {
            "total": len(audio_files),
            "valid": 0,
            "invalid": 0,
            "needs_conversion": 0,
            "durations": [],
            "issues": []
        }
        
        logger.info(f"\nVerifying {len(audio_files)} files...")
        
        for file in audio_files:
            file_path = os.path.join(audio_dir, file)
            verification = verifier.verify_file(file_path)
            
            if verification["valid"]:
                results["valid"] += 1
                results["durations"].append(verification["duration"])
                
                if verification["needs_conversion"]:
                    results["needs_conversion"] += 1
                    logger.info(
                        f"File needs conversion - {file}:\n"
                        f"  Sample rate: {verification['sample_rate']} Hz\n"
                        f"  Channels: {verification['channels']}"
                    )
            else:
                results["invalid"] += 1
                results["issues"].append(f"{file}: {verification['error']}")
        
        # Calculate statistics
        if results["durations"]:
            avg_duration = sum(results["durations"]) / len(results["durations"])
            max_duration = max(results["durations"])
            min_duration = min(results["durations"])
            total_duration = sum(results["durations"])
        
        # Print summary
        logger.info("\n=== Audio Verification Summary ===")
        logger.info(f"Total files: {results['total']}")
        logger.info(f"Valid files: {results['valid']}")
        logger.info(f"Invalid files: {results['invalid']}")
        logger.info(f"Files needing conversion: {results['needs_conversion']}")
        
        if results["durations"]:
            logger.info(f"\nDuration Statistics:")
            logger.info(f"Average duration: {avg_duration:.2f} seconds")
            logger.info(f"Maximum duration: {max_duration:.2f} seconds")
            logger.info(f"Minimum duration: {min_duration:.2f} seconds")
            logger.info(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        
        if results["issues"]:
            logger.info("\nIssues found:")
            for issue in results["issues"]:
                logger.info(f"- {issue}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}", exc_info=True)

if __name__ == "__main__":
    print("\n=== Starting Audio Verification ===\n")
    verify_audio_files()
    print("\nVerification complete!")
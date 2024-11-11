import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperMediumTester:
    def __init__(self):
        """Initialize Whisper Medium model"""
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Use medium model
        model_name = "openai/whisper-medium"
        logger.info(f"Loading model: {model_name}")
        
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def prepare_audio(self, audio_path):
        """Prepare audio with proper attention mask"""
        try:
            # Load audio
            logger.info(f"Loading audio file: {audio_path}")
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.info("Converted stereo to mono")
            
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                logger.info("Resampled to 16kHz")
            
            # Convert to numpy array
            audio_array = waveform.squeeze().numpy()
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error preparing audio: {str(e)}")
            return None
    
    @torch.no_grad()
    def transcribe(self, audio_path):
        """Transcribe audio using medium model"""
        try:
            # Process audio
            audio_array = self.prepare_audio(audio_path)
            if audio_array is None:
                return None
            
            # Prepare inputs
            logger.info("Processing with Whisper...")
            inputs = self.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate transcription with medium model settings
            generated = self.model.generate(
                inputs.input_features,
                forced_decoder_ids=self.processor.get_decoder_prompt_ids(
                    language="kn",
                    task="transcribe"
                ),
                max_length=448,  # Increased for medium model
                num_beams=5,     # More beams for better quality
                temperature=0.7,  # Balanced temperature
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                return_dict_in_generate=True
            )
            
            # Decode transcription
            transcription = self.processor.batch_decode(
                generated.sequences, 
                skip_special_tokens=True
            )[0].strip()
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing: {str(e)}")
            return None

def test_model():
    """Test the medium model"""
    try:
        # Initialize tester
        tester = WhisperMediumTester()
        
        # Setup paths
        audio_dir = os.path.join("data", "audio")
        results_dir = "transcription_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Get test files
        test_files = [f for f in os.listdir(audio_dir) 
                     if f.endswith(('.mp3', '.wav', '.m4a'))][:5]
        
        if not test_files:
            logger.error("No audio files found!")
            return
        
        # Process files
        logger.info(f"\nProcessing {len(test_files)} files with medium model...")
        
        with open(os.path.join(results_dir, "medium_model_results.txt"), "w", encoding="utf-8") as f:
            f.write("Whisper Medium Model - Kannada Transcriptions\n")
            f.write("=" * 50 + "\n\n")
            
            for audio_file in test_files:
                logger.info(f"\nProcessing: {audio_file}")
                audio_path = os.path.join(audio_dir, audio_file)
                
                transcription = tester.transcribe(audio_path)
                
                if transcription:
                    # Check if output is in Kannada
                    has_kannada = any('\u0C80' <= c <= '\u0CFF' for c in transcription)
                    
                    logger.info("Transcription:")
                    logger.info(transcription)
                    logger.info(f"Contains Kannada: {has_kannada}")
                    
                    # Save results
                    f.write(f"File: {audio_file}\n")
                    f.write("Transcription:\n")
                    f.write(f"{transcription}\n")
                    f.write(f"Contains Kannada: {has_kannada}\n")
                    f.write("-" * 50 + "\n")
                else:
                    logger.error(f"Failed to transcribe {audio_file}")
        
        logger.info(f"\nResults saved to {results_dir}/medium_model_results.txt")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}", exc_info=True)

if __name__ == "__main__":
    print("\n=== Testing Whisper Medium Model ===\n")
    test_model()
    print("\nTesting complete!")
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import logging
import json
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_sizes = ["tiny", "base", "small", "medium"]
        self.results = {}
    
    def evaluate_model(self, model_size, audio_path):
        """Evaluate a specific model size"""
        try:
            logger.info(f"\nEvaluating {model_size} model...")
            
            # Load model
            model_name = f"openai/whisper-{model_size}"
            processor = WhisperProcessor.from_pretrained(model_name)
            model = WhisperForConditionalGeneration.from_pretrained(model_name)
            model.to(self.device)
            
            # Process audio
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Prepare input
            inputs = processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                language="kn",
                task="transcribe"
            ).to(self.device)
            
            # Time transcription
            start_time = time.time()
            
            with torch.no_grad():
                predicted_ids = model.generate(
                    inputs.input_features,
                    language="kn",
                    task="transcribe"
                )
            
            transcription = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            processing_time = time.time() - start_time
            
            return {
                "transcription": transcription,
                "processing_time": processing_time,
                "characters": len(transcription),
                "words": len(transcription.split())
            }
            
        except Exception as e:
            logger.error(f"Error evaluating {model_size} model: {str(e)}")
            return None
    
    def evaluate_all_models(self, test_file):
        """Evaluate all model sizes on a test file"""
        results = {}
        
        for size in self.model_sizes:
            result = self.evaluate_model(size, test_file)
            if result:
                results[size] = result
                logger.info(f"\n{size.upper()} Model Results:")
                logger.info(f"Processing time: {result['processing_time']:.2f} seconds")
                logger.info(f"Characters: {result['characters']}")
                logger.info(f"Words: {result['words']}")
                logger.info("Transcription:")
                logger.info(result['transcription'])
        
        return results

def main():
    """Evaluate different Whisper models"""
    try:
        # Setup
        audio_dir = os.path.join("data", "audio")
        results_dir = "model_evaluation_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Get test file (use first audio file)
        test_files = [f for f in os.listdir(audio_dir) 
                     if f.endswith(('.mp3', '.wav', '.m4a'))][:1]
        
        if not test_files:
            logger.error("No audio files found!")
            return
        
        test_file = os.path.join(audio_dir, test_files[0])
        logger.info(f"Using test file: {test_files[0]}")
        
        # Evaluate models
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_all_models(test_file)
        
        # Save results
        output_file = os.path.join(results_dir, "model_comparison.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Whisper Model Comparison\n{'='*50}\n\n")
            f.write(f"Test file: {test_files[0]}\n\n")
            
            for size, result in results.items():
                f.write(f"{size.upper()} Model\n{'-'*20}\n")
                f.write(f"Processing time: {result['processing_time']:.2f} seconds\n")
                f.write(f"Characters: {result['characters']}\n")
                f.write(f"Words: {result['words']}\n")
                f.write("Transcription:\n")
                f.write(f"{result['transcription']}\n\n")
        
        logger.info(f"\nResults saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)

if __name__ == "__main__":
    print("\n=== Evaluating Whisper Models ===\n")
    main()
    print("\nEvaluation complete!")
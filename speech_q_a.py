import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechQASystem:
    def __init__(self, base_model: str = "openai/whisper-small"):
        """Initialize Speech QA System"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load base model
        logger.info(f"Loading base model: {base_model}")
        self.processor = WhisperProcessor.from_pretrained(base_model)
        self.model = WhisperForConditionalGeneration.from_pretrained(base_model)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize text matching
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.corpus_data = {}
        self.corpus_vectors = None
        
        # Load cached transcriptions if available
        self.cache_file = "transcriptions_cache.json"
        self.load_cache()
    
    def load_cache(self):
        """Load cached transcriptions"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.corpus_data = json.load(f)
                logger.info(f"Loaded {len(self.corpus_data)} cached transcriptions")
        except Exception as e:
            logger.warning(f"Could not load cache: {str(e)}")
    
    def save_cache(self):
        """Save transcriptions to cache"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.corpus_data, f, ensure_ascii=False, indent=2)
            logger.info("Saved transcriptions to cache")
        except Exception as e:
            logger.warning(f"Could not save cache: {str(e)}")
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        try:
            # Check file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=16000
                )
                waveform = resampler(waveform)
            
            return waveform
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            raise
    
    def transcribe_audio(self, waveform: torch.Tensor) -> str:
        """Transcribe audio using Whisper"""
        try:
            # Process audio
            inputs = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_features,
                    language="kn",
                    task="transcribe",
                    temperature=0.3,
                    num_beams=5
                )
            
            transcription = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )[0].strip()
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return ""
    
    def find_audio_files(self, directory: str) -> list:
        """Find all audio files in directory and subdirectories"""
        audio_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.m4a')):
                    audio_files.append(os.path.join(root, file))
        return audio_files
    
    def process_corpus(self, corpus_path: str):
        """Process audio corpus"""
        try:
            logger.info(f"Processing corpus from: {corpus_path}")
            
            # Find all audio files
            audio_files = self.find_audio_files(corpus_path)
            
            if not audio_files:
                raise ValueError(f"No audio files found in {corpus_path}")
            
            logger.info(f"Found {len(audio_files)} audio files")
            
            # Process each file
            transcriptions = []
            for audio_file in tqdm(audio_files, desc="Processing corpus"):
                try:
                    file_id = Path(audio_file).stem
                    
                    # Use cache if available
                    if file_id in self.corpus_data:
                        transcription = self.corpus_data[file_id]["transcription"]
                    else:
                        # Preprocess and transcribe
                        waveform = self.preprocess_audio(audio_file)
                        transcription = self.transcribe_audio(waveform)
                        
                        # Add to cache
                        self.corpus_data[file_id] = {
                            "path": audio_file,
                            "transcription": transcription
                        }
                    
                    if transcription:
                        transcriptions.append(transcription)
                    
                except Exception as e:
                    logger.error(f"Error processing {audio_file}: {str(e)}")
            
            # Create TF-IDF vectors
            if transcriptions:
                self.corpus_vectors = self.vectorizer.fit_transform(transcriptions)
                logger.info(f"Successfully processed {len(transcriptions)} files")
                self.save_cache()
            else:
                raise ValueError("No files were successfully processed")
            
        except Exception as e:
            logger.error(f"Error processing corpus: {str(e)}")
            raise
    
    def find_answer(self, question_path: str, top_k: int = 3) -> list:
        """Find answer for audio question"""
        try:
            # Process question
            logger.info(f"Processing question from: {question_path}")
            waveform = self.preprocess_audio(question_path)
            question_text = self.transcribe_audio(waveform)
            
            if not question_text:
                raise ValueError("Failed to transcribe question")
            
            logger.info(f"Transcribed question: {question_text}")
            
            if not self.corpus_vectors is not None:
                raise ValueError("No corpus data available. Process corpus first.")
            
            # Get question vector
            question_vector = self.vectorizer.transform([question_text])
            
            # Calculate similarities
            similarities = cosine_similarity(
                question_vector,
                self.corpus_vectors
            ).flatten()
            
            # Get top matches
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Prepare results
            results = []
            for idx in top_indices:
                file_id = list(self.corpus_data.keys())[idx]
                data = self.corpus_data[file_id]
                
                results.append({
                    "file": data["path"],
                    "transcription": data["transcription"],
                    "similarity": float(similarities[idx])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding answer: {str(e)}")
            return []

def main():
    """Main execution"""
    try:
        # Initialize system
        qa_system = SpeechQASystem()
        
        # Set paths based on project root
        project_root = os.getcwd()  # This should be your ml_fiesta directory
        corpus_path = os.path.join(project_root, "data")  # Search all data directory
        questions_dir = os.path.join(project_root, "questions")
        
        # List all paths for debugging
        logger.info(f"Project root: {project_root}")
        logger.info(f"Corpus path: {corpus_path}")
        logger.info(f"Questions directory: {questions_dir}")
        
        # Check question directory
        if os.path.exists(questions_dir):
            question_files = [f for f in os.listdir(questions_dir)
                            if f.endswith(('.mp3', '.wav', '.m4a'))]
            
            if question_files:
                logger.info("\nAvailable question files:")
                for f in question_files:
                    logger.info(f"- {f}")
                
                # Process corpus once
                logger.info("\nProcessing corpus...")
                qa_system.process_corpus(corpus_path)
                
                while True:
                    question_file = input("\nEnter question filename (or 'exit' to quit): ")
                    if question_file.lower() == 'exit':
                        break
                    
                    question_path = os.path.join(questions_dir, question_file)
                    if not os.path.exists(question_path):
                        logger.error(f"File not found: {question_path}")
                        continue
                    
                    # Find answers
                    results = qa_system.find_answer(question_path)
                    
                    # Display results
                    if results:
                        logger.info("\n=== Found Answers ===")
                        for i, result in enumerate(results, 1):
                            logger.info(f"\nMatch {i} (similarity: {result['similarity']:.2f})")
                            logger.info(f"File: {Path(result['file']).name}")
                            logger.info(f"Transcription: {result['transcription']}")
                    else:
                        logger.info("\nNo matching answers found!")
            else:
                logger.error(f"No question files found in {questions_dir}")
        else:
            logger.error(f"Questions directory not found: {questions_dir}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    print("\n=== Starting Speech QA System ===\n")
    main()

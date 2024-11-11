import streamlit as st
import os
from pathlib import Path
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import tempfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import librosa  # Alternative to torchaudio

class SpeechQA:
    def __init__(self, base_model: str = "openai/whisper-small"):
        """Initialize Speech QA System"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Using device: {self.device}")
        
        # Load base model
        self.processor = WhisperProcessor.from_pretrained(base_model)
        self.model = WhisperForConditionalGeneration.from_pretrained(base_model)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize text matching
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.corpus_data = {}
        self.corpus_vectors = None
    
    def load_audio(self, audio_path: str):
        """Load audio using librosa"""
        try:
            # Load and resample audio
            waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Convert to torch tensor
            waveform = torch.from_numpy(waveform).float()
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
                
            return waveform
            
        except Exception as e:
            st.error(f"Error loading audio: {str(e)}")
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
            st.error(f"Error transcribing audio: {str(e)}")
            return ""
    
    def process_corpus(self, corpus_path: str):
        """Process audio corpus"""
        try:
            st.info(f"Processing corpus from: {corpus_path}")
            
            # Find all audio files
            audio_files = []
            for ext in ['.mp3', '.wav', '.m4a']:
                audio_files.extend(Path(corpus_path).glob(f"**/*{ext}"))
            
            if not audio_files:
                raise ValueError(f"No audio files found in {corpus_path}")
            
            st.info(f"Found {len(audio_files)} audio files")
            
            # Process each file
            transcriptions = []
            progress_bar = st.progress(0)
            
            for i, audio_file in enumerate(audio_files):
                try:
                    # Load and transcribe
                    waveform = self.load_audio(str(audio_file))
                    transcription = self.transcribe_audio(waveform)
                    
                    if transcription:
                        self.corpus_data[audio_file.stem] = {
                            "path": str(audio_file),
                            "transcription": transcription
                        }
                        transcriptions.append(transcription)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(audio_files))
                    
                except Exception as e:
                    st.warning(f"Error processing {audio_file.name}: {str(e)}")
            
            # Create TF-IDF vectors
            if transcriptions:
                self.corpus_vectors = self.vectorizer.fit_transform(transcriptions)
                st.success(f"Successfully processed {len(transcriptions)} files")
            else:
                raise ValueError("No files were successfully processed")
            
        except Exception as e:
            st.error(f"Error processing corpus: {str(e)}")
            raise
    
    def find_answer(self, question_path: str, top_k: int = 3) -> list:
        """Find answer for audio question"""
        try:
            # Process question
            waveform = self.load_audio(question_path)
            question_text = self.transcribe_audio(waveform)
            
            if not question_text:
                raise ValueError("Failed to transcribe question")
            
            st.info(f"Transcribed question: {question_text}")
            
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
            st.error(f"Error finding answer: {str(e)}")
            return []

def record_audio(duration=10, sample_rate=16000):
    """Record audio from microphone"""
    try:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1
        )
        sd.wait()
        return recording, sample_rate
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        raise

def save_recording(audio_data, sample_rate):
    """Save recording to questions directory"""
    save_dir = Path("questions")
    save_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = save_dir / f"question_{timestamp}.mp3"
    
    sf.write(filename, audio_data, sample_rate)
    return str(filename)

def main():
    st.title("🎤 Sandalwood Knowledge QA System")
    st.markdown("Ask questions about Sandalwood cultivation in Kannada!")
    
    # Initialize QA system
    qa_system = SpeechQA()
    
    # Create tabs
    record_tab, upload_tab = st.tabs(["Record Question", "Upload Question"])
    
    with record_tab:
        st.subheader("🎙️ Record Your Question")
        duration = st.slider("Recording duration (seconds)", 5, 30, 10)
        
        if st.button("Start Recording"):
            try:
                # Countdown
                for i in range(3, 0, -1):
                    st.markdown(f"Starting in {i}...")
                    time.sleep(1)
                
                st.markdown("🔴 Recording... Speak now!")
                audio_data, sample_rate = record_audio(duration)
                question_path = save_recording(audio_data, sample_rate)
                st.success("Recording complete!")
                
                # Process question
                if not qa_system.corpus_vectors:
                    with st.spinner("Processing corpus..."):
                        qa_system.process_corpus("data")
                
                results = qa_system.find_answer(question_path)
                display_results(results)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with upload_tab:
        st.subheader("📁 Upload Your Question")
        uploaded_file = st.file_uploader(
            "Upload audio (MP3, WAV, M4A)",
            type=['mp3', 'wav', 'm4a']
        )
        
        if uploaded_file:
            try:
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    
                    # Process corpus if needed
                    if not qa_system.corpus_vectors:
                        with st.spinner("Processing corpus..."):
                            qa_system.process_corpus("data")
                    
                    # Process question
                    results = qa_system.find_answer(tmp_file.name)
                    display_results(results)
                    
                    # Cleanup
                    os.unlink(tmp_file.name)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

def display_results(results):
    """Display search results"""
    if results:
        st.subheader("🎯 Found Answers")
        for i, result in enumerate(results, 1):
            with st.container():
                st.markdown(f"""
                    <div style='padding:1rem;border-radius:0.5rem;background-color:#f0f2f6;margin-bottom:1rem;'>
                        <h4>Match {i}</h4>
                        <p style='color:#4CAF50;font-weight:bold;'>
                            Confidence: {result['similarity']:.2%}
                        </p>
                        <p>Source: {Path(result['file']).name}</p>
                        <p style='font-style:italic;color:#666;'>
                            {result['transcription']}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                if os.path.exists(result['file']):
                    audio_file = open(result['file'], 'rb')
                    st.audio(audio_file)
    else:
        st.warning("No matching answers found!")

if __name__ == "__main__":
    main()

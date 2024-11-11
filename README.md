# KANNADA_ASR_MODEL
# ML Fiesta - Sandalwood Knowledge Assistant

A speech-based Question & Answer system for Sandalwood cultivation knowledge in Kannada language, using fine-tuned Whisper ASR model.

## Project Structure

```
ml_fiesta/
├── whisper_project/
│   ├── app.py                     # Streamlit frontend application
│   ├── check_audio.py             # Audio file verification utility
│   ├── efficient_loader.py        # Memory-efficient data loading
│   ├── find_checkpoints.py        # Checkpoint management utility
│   ├── main_train.py             # Main training script
│   ├── model_eval.py             # Model evaluation script
│   ├── organize_files.py         # File organization utility
│   ├── questions.py              # Question handling
│   ├── speech_q_a.py            # Speech QA implementation
│   ├── test_audio.py            # Audio testing utility
│   ├── test_loader.py           # DataLoader testing
│   ├── test_whisper.py          # Whisper model testing
│   ├── train.py                 # Training implementation
│   └── whisper_fine_tuner.py    # Whisper model fine-tuning
├── data/
│   └── audio/                    # Audio dataset directory
├── checkpoints/                  # Model checkpoints
├── model_evaluation_results/     # Evaluation results
├── questions/                    # Question audio files
├── transcription_results/        # Transcription outputs
└── whisper_env/                 # Python virtual environment
```

## Setup and Installation

1. Create and activate virtual environment:
```bash
python -m venv whisper_env
source whisper_env/bin/activate  # Linux/Mac
# or
whisper_env\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Features

- **Fine-tuned Whisper ASR**: Customized for Kannada language
- **Memory-Efficient Data Loading**: Optimized for large audio datasets
- **Streamlit UI**: User-friendly interface for asking questions
- **Real-time Audio Recording**: Direct question recording capability
- **Audio Corpus Management**: Organized audio data handling
- **Model Evaluation**: Comprehensive testing and evaluation tools

## Usage

### Training the Model
```bash
python whisper_project/main_train.py --audio_dir data/audio --model_name "openai/whisper-small"
```

### Starting the QA System
```bash
streamlit run whisper_project/app.py
```

### Testing Audio Files
```bash
python whisper_project/check_audio.py
```

## Testing and Evaluation

1. Check audio files:
```bash
python whisper_project/check_audio.py
```

2. Evaluate model:
```bash
python whisper_project/model_eval.py
```

3. Test data loading:
```bash
python whisper_project/test_loader.py
```

## Code Structure

- `app.py`: Streamlit frontend interface
- `whisper_fine_tuner.py`: Core fine-tuning implementation
- `efficient_loader.py`: Memory-efficient data loading
- `model_eval.py`: Model evaluation utilities
- `test_*.py`: Various testing utilities
- `organize_files.py`: File organization tools

## Dependencies

- PyTorch
- Transformers
- Streamlit
- torchaudio
- sounddevice
- soundfile
- librosa
- scikit-learn
- numpy
- tqdm

## Model Details

- Base Model: OpenAI Whisper Small
- Fine-tuned on: Kannada audio dataset
- Task: Speech Recognition and Question-Answering
- Domain: Sandalwood Cultivation Knowledge

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



## Acknowledgments

- OpenAI's Whisper model
- MLFiesta project team
- Sandalwood cultivation domain experts
- Audio dataset contributors


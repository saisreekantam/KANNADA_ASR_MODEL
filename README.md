# ML Fiesta - Sandalwood Knowledge Assistant

A speech-based Question & Answer system for Sandalwood cultivation knowledge in Kannada language. This project uses OpenAI's Whisper ASR model fine-tuned on Kannada language data to provide accurate answers to voice queries about sandalwood cultivation.

## Project Overview

This system allows users to ask questions about sandalwood cultivation in Kannada language either through voice recording or audio file upload. The system processes these questions using a fine-tuned Whisper model and finds relevant answers from a corpus of audio knowledge.

## Project Structure and File Descriptions

```
ml_fiesta/
├── whisper_project/
│   ├── app.py                    # Streamlit frontend interface for user interaction
│   ├── speech_qa.py             # Core QA system implementation
│   ├── efficient_loader.py       # Memory-efficient data loading utilities
│   ├── whisper_fine_tuner.py    # Whisper model fine-tuning implementation
│   ├── main_train.py            # Main training script orchestration
│   ├── model_eval.py            # Model evaluation and metrics calculation
│   ├── check_audio.py           # Audio file verification and preprocessing
│   ├── organize_files.py        # File organization and management
│   ├── test_audio.py            # Audio processing testing utilities
│   ├── test_loader.py           # DataLoader testing and validation
│   ├── test_whisper.py          # Whisper model testing suite
│   └── find_checkpoints.py      # Checkpoint management utilities
├── data/
│   └── audio/                   # Audio dataset directory
├── checkpoints/                 # Model checkpoints storage
├── questions/                   # Question audio files
└── requirements.txt             # Project dependencies
```

### Key Components

#### Core Files
- **app.py**
  - Streamlit-based web interface
  - Handles audio recording and file uploads
  - Displays results with audio playback
  - Provides user-friendly interaction

- **speech_qa.py**
  - Implements the core Question-Answering logic
  - Manages audio transcription
  - Performs semantic search for answers
  - Handles result ranking and selection

- **whisper_fine_tuner.py**
  - Implements Whisper model fine-tuning
  - Customizes model for Kannada language
  - Manages training loops and optimization
  - Handles model saving and loading

#### Training and Evaluation
- **main_train.py**
  - Orchestrates the training process
  - Handles command-line arguments
  - Manages dataset splitting
  - Controls training parameters

- **model_eval.py**
  - Implements evaluation metrics
  - Tests model performance
  - Generates performance reports
  - Compares different model versions

#### Data Management
- **efficient_loader.py**
  - Implements memory-efficient data loading
  - Handles batch processing
  - Manages audio file streaming
  - Optimizes memory usage for large datasets

- **organize_files.py**
  - Manages file organization
  - Handles dataset structure
  - Maintains file naming conventions
  - Ensures proper directory structure

#### Testing Utilities
- **check_audio.py**
  - Verifies audio file integrity
  - Checks format compatibility
  - Validates sample rates
  - Ensures audio quality standards

- **test_loader.py**
  - Tests data loading functionality
  - Validates batch processing
  - Ensures proper data handling
  - Verifies memory efficiency

- **test_whisper.py**
  - Tests Whisper model functionality
  - Validates transcription quality
  - Checks model performance
  - Ensures reliable output

- **test_audio.py**
  - Tests audio processing functions
  - Validates preprocessing steps
  - Ensures format conversions
  - Checks audio quality

#### Support Utilities
- **find_checkpoints.py**
  - Manages model checkpoints
  - Handles checkpoint loading/saving
  - Tracks training progress
  - Maintains version control

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/saisreekatam/KANNADA_ASR_MODEL.git
cd ml_fiesta
```

2. Create and activate virtual environment:
```bash
python -m venv whisper_env
source whisper_env/bin/activate  # Linux/Mac
# or
whisper_env\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Instructions

### Training the Model
```bash
python whisper_project/main_train.py \
    --audio_dir data/audio \
    --model_name "openai/whisper-small" \
    --batch_size 8 \
    --epochs 10 \
    --learning_rate 1e-5
```

### Running the QA System
```bash
streamlit run whisper_project/app.py
```

### Testing Components
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

## Dependencies

Main dependencies include:
- PyTorch: Deep learning framework
- Transformers: For Whisper model
- Streamlit: Web interface
- torchaudio: Audio processing
- sounddevice: Audio recording
- soundfile: Audio file handling
- librosa: Audio processing
- scikit-learn: Machine learning utilities
- numpy: Numerical computations
- tqdm: Progress bars

## Model Architecture

The system uses a fine-tuned version of OpenAI's Whisper ASR model:
- Base Model: whisper-small
- Fine-tuning: Custom Kannada dataset
- Task: Speech Recognition and QA
- Modifications:
  - Custom tokenizer for Kannada
  - Modified attention mechanisms
  - Optimized for domain-specific vocabulary

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the Whisper model
- MLFiesta project team
- Audio dataset contributors
- Sandalwood cultivation experts

## Contact

For questions and support, please open an issue in the GitHub repository.

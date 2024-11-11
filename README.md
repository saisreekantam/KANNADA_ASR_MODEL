# KANNADA_ASR_MODEL
# Sandalwood Knowledge QA System

A speech-based Question Answering system for Sandalwood cultivation knowledge in Kannada language. The system uses OpenAI's Whisper ASR model fine-tuned on Kannada language for transcription and implements a semantic search to find relevant answers from an audio corpus.

## Features

- Real-time audio question recording
- Support for uploading audio questions
- Fine-tuned Whisper ASR for Kannada language
- Semantic search for finding relevant answers
- User-friendly Streamlit interface
- Audio playback for answers

## Project Structure

```
sandalwood-qa/
├── app.py                  # Streamlit frontend
├── speech_qa.py           # Core QA system implementation
├── data/
│   └── audio/
│       └── corpus/        # Audio corpus directory
├── questions/             # Recorded questions directory
├── checkpoints/           # Model checkpoints
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sandalwood-qa.git
cd sandalwood-qa
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download model checkpoints (if using fine-tuned model):
```bash
# Add instructions for downloading checkpoints
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Use the system:
   - Record questions directly through the microphone
   - Upload pre-recorded questions
   - View transcriptions and matching answers
   - Play audio segments of answers

## Tech Stack

- Python 3.8+
- PyTorch
- Transformers (Whisper)
- Streamlit
- scikit-learn
- librosa
- sounddevice
- soundfile

## Dataset

The system uses a corpus of audio files related to Sandalwood cultivation in Kannada language. The dataset includes:
- Source: YouTube scraping
- Language: Kannada
- Content: Sandalwood cultivation techniques and knowledge
- Format: MP3/WAV audio files

## Model

The system uses OpenAI's Whisper ASR model fine-tuned on Kannada language data:
- Base model: whisper-small
- Fine-tuning: Custom Kannada dataset
- Task: Speech recognition and transcription

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Thanks to OpenAI for the Whisper model
- Dataset contributors


## Contact


Project Link: https://github.com/saisreekantam/KANNADA_ASR_MODEL


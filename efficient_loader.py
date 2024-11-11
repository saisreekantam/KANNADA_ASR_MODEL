import torch
import torchaudio
from transformers import WhisperProcessor
from torch.utils.data import Dataset
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryEfficientDataset(Dataset):
    def __init__(
        self,
        audio_dir: str,
        processor: WhisperProcessor,
        max_duration: int = 30,
        sample_rate: int = 16000,
        use_fp16: bool = True
    ):
        self.audio_dir = audio_dir
        self.processor = processor
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.use_fp16 = use_fp16
        
        # Get all audio files
        self.audio_files = [
            f for f in os.listdir(audio_dir)
            if f.endswith(('.wav', '.mp3', '.m4a'))
        ]
        logger.info(f"Found {len(self.audio_files)} audio files")
        
        # Set dtype for memory efficiency
        self.dtype = torch.float16 if use_fp16 else torch.float32
    
    def __len__(self):
        return len(self.audio_files)
    
    def load_audio(self, audio_path: str):
        """Load and preprocess audio file"""
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.sample_rate
                )
                waveform = resampler(waveform)
            
            # Convert to efficient dtype
            waveform = waveform.to(self.dtype)
            
            return waveform
            
        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {str(e)}")
            return None
    
    def __getitem__(self, idx):
        """Get a single item"""
        try:
            audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
            
            # Load audio
            waveform = self.load_audio(audio_path)
            if waveform is None:
                return None
            
            # Process with Whisper processor
            inputs = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            
            return {
                "input_features": inputs.input_features.squeeze(),
                "attention_mask": inputs.attention_mask.squeeze()
            }
            
        except Exception as e:
            logger.error(f"Error processing {self.audio_files[idx]}: {str(e)}")
            return None

def create_dataloader(
    dataset,
    batch_size: int = 4,
    shuffle: bool = True
):
    """Create memory efficient dataloader"""
    
    def collate_fn(batch):
        # Remove None values
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        
        # Stack features
        input_features = torch.stack([item["input_features"] for item in batch])
        attention_masks = torch.stack([item["attention_mask"] for item in batch])
        
        return {
            "input_features": input_features,
            "attention_mask": attention_masks
        }
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=False
    )
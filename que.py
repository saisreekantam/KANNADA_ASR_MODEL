import sounddevice as sd
import soundfile as sf
import numpy as np
import os
from pathlib import Path

def record_question(duration=10, sample_rate=16000):
    """
    Record audio question from microphone
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Audio sample rate
    """
    # Create questions directory if it doesn't exist
    save_dir = Path("questions")
    save_dir.mkdir(exist_ok=True)
    
    print(f"\nRecording will start in 3 seconds...")
    print("Please ask your question in Kannada about sandalwood cultivation")
    print("Example questions:")
    print("1. ಗಂಧದ ಮರ ಬೆಳೆಸಲು ಎಷ್ಟು ನೀರು ಬೇಕು?")
    print("2. ಗಂಧದ ಮರ ಬೆಳೆಸಲು ಯಾವ ಮಣ್ಣು ಉತ್ತಮ?")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"{i}...")
        sd.sleep(1000)
    
    print("\nRecording started... Speak now!")
    
    # Record audio
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1
    )
    
    # Wait for recording to complete
    sd.wait()
    
    print("Recording complete!")
    
    # Generate filename with timestamp
    filename = save_dir / "question.mp3"
    
    # Save recording
    sf.write(filename, recording, sample_rate)
    print(f"\nQuestion saved to: {filename}")
    
    return str(filename)

if __name__ == "__main__":
    try:
        # Record for 10 seconds
        question_file = record_question(duration=10)
        print("\nRecording successful!")
        print("You can now use this file with the QA system")
        
    except Exception as e:
        print(f"Error recording audio: {str(e)}")

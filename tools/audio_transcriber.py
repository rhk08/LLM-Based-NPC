import whisper

class AudioTranscriber:
    def __init__(self, model_name="small.en"):
        # Load the Whisper model
        self.model = whisper.load_model(model_name)

    def transcribe_audio(self, audio_file):
        print(f"Transcribing {audio_file}...")
        result = self.model.transcribe(audio_file)
        print("Transcription completed.")
        print(result["text"])
        return result["text"]
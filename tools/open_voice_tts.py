import os
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

class OpenVoiceTTS:
    def __init__(self, base_speaker_ckpt='OpenVoice/checkpoints/base_speakers/EN', converter_ckpt='OpenVoice/checkpoints/converter', device=None, output_dir='OpenVoice/outputs'):
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        
        # Load the base speaker TTS model
        self.base_speaker_tts = BaseSpeakerTTS(f'{base_speaker_ckpt}/config.json', device=self.device)
        self.base_speaker_tts.load_ckpt(f'{base_speaker_ckpt}/checkpoint.pth')

        # Load the tone color converter model
        self.tone_color_converter = ToneColorConverter(f'{converter_ckpt}/config.json', device=self.device)
        self.tone_color_converter.load_ckpt(f'{converter_ckpt}/checkpoint.pth')

        # Create output directory if not exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Load the source speaker embedding
        self.source_se = torch.load(f'{base_speaker_ckpt}/en_default_se.pth').to(self.device)

    def extract_target_speaker(self, reference_speaker_path, processed_dir='processed', vad=True):
        """
        Extracts the speaker embedding from a reference speaker.
        """
        target_se, audio_name = se_extractor.get_se(reference_speaker_path, self.tone_color_converter, target_dir=processed_dir, vad=vad)
        return target_se, audio_name

    def generate_audio(self, text, target_se, speaker='default', language='English', speed=1.0, encode_message="@MyShell"):
        """
        Generates audio from the input text using the base speaker TTS model, then applies tone color conversion.
        """
        # Generate temporary output path for base speaker audio
        temp_audio_path = f'{self.output_dir}/tmp.wav'
        
        # Generate base speaker TTS audio
        self.base_speaker_tts.tts(text, temp_audio_path, speaker=speaker, language=language, speed=speed)
        
        # Define the final output path
        final_output_path = f'{self.output_dir}/output_en_default.wav'
        
        # Apply tone color conversion
        self.tone_color_converter.convert(
            audio_src_path=temp_audio_path, 
            src_se=self.source_se, 
            tgt_se=target_se, 
            output_path=final_output_path,
            message=encode_message)
        
        print(f"Audio generated and saved to {final_output_path}")
        return final_output_path

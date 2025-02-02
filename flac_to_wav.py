import os
import librosa
import soundfile as sf

def convert_audio_to_wav(input_dir, output_dir):
    """
    Convert FLAC and MP3 files to WAV format using Librosa.
    Args:
        input_dir (str): Path to the directory containing FLAC/MP3 files.
        output_dir (str): Path to the directory where WAV files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            input_path = os.path.join(root, file)
            filename, ext = os.path.splitext(file)
            if ext.lower() in [".flac", ".mp3"]:
                output_path = os.path.join(output_dir, f"{filename}.wav")
                try:
                    # Load audio using Librosa
                    data, samplerate = librosa.load(input_path, sr=None)
                    # Save the audio as WAV
                    sf.write(output_path, data, samplerate)
                    print(f"Converted {ext.upper()} to WAV: {output_path}")
                except Exception as e:
                    print(f"Error converting {input_path} to WAV: {e}")

if __name__ == "__main__":
    input_directory = r"C:\Users\Legion 5I 72IN\Desktop\SUMMER internship\SSL\asv19_flac"
    output_directory = r"C:\Users\Legion 5I 72IN\Desktop\SUMMER internship\SSL\asv19_wav"
    convert_audio_to_wav(input_directory, output_directory)



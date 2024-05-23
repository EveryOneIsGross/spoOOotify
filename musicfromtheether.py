import os
import random
import re
from textblob import TextBlob
import hashlib
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import numpy as np
from pydub import AudioSegment


# Expanded mood dictionary
mood_dict = {
'positive': [
"Uplifting", "Joyful", "Energetic", "Hopeful", "Happy", "Cheerful", "Optimistic", "Ecstatic",
"Euphoric", "Blissful", "Exhilarated", "Inspired", "Motivated", "Triumphant", "Adventurous",
"Carefree", "Playful", "Whimsical", "Humorous", "Silly", "Quirky", "Manic", "Hyper",
"Crazed","Spoken Word", "Celebratory", "Victorious", "Passionate",
"Amused", "Delighted", "Enchanted", "Fascinated", "Awestruck", "Grateful", "Blessed",
"Thrilled", "Elated", "Enthusiastic", "Zestful", "Vibrant", "Radiant", "Bubbly", "Lively"
],
'negative': [
"Melancholic", "Sad", "Gloomy", "Pensive", "Sorrowful", "Mournful", "Grieving", "Despairing",
"Hopeless", "Depressed", "Anguished", "Heartbroken", "Lonely", "Abandoned", "Forlorn",
"Bleak", "Dismal", "Pessimistic", "Cynical", "Bitter", "Angry", "Furious", "Enraged",
"Irritated", "Frustrated", "Anxious", "Stressed", "Nervous", "Tense", "Uneasy", "Worried",
"Paranoid", "Suspicious", "Fearful", "Terrified", "Panicked", "Hysterical", "Goth", "Satanic",
"Spoken Word", "Disappointed", "Betrayed", "Hurt", "Offended", "Disgusted", "Ashamed",
"Guilty", "Regretful", "Remorseful", "Miserable", "Wretched", "Troubled", "Disturbed"
],
'neutral': [
"Calm", "Balanced", "Serene", "Reflective", "Mellow", "Stoic", "Detached", "Indifferent",
"Corporate", "Bland", "Boring", "Mundane", "Vaporwave", "Ambient", "Piano", "Spoken Word",
"Contemplative", "Meditative", "Introspective", "Pensive", "Thoughtful", "Zen", "Neutral",
"Impartial", "Unbiased", "Objective", "Dispassionate", "Aloof", "Distant", "Reserved",
"Formal", "Professional", "Businesslike", "Serious", "Sober", "Solemn", "Grave", "Heavy",
"Monotonous", "Repetitive", "Droning", "Hypnotic", "Trance-like", "Ethereal", "Spacey",
"Floaty", "Dreamy", "Surreal", "Liminal", "Uncanny", "Eerie", "Haunting", "Mysterious"
]
}
# Specify the directory containing the corpus files
corpus_directory = "sysPROMPTcorpus"

# Specify the guidance text file
guidance_file = "guidance.txt"


# Get the list of corpus files
corpus_files = [file for file in os.listdir(corpus_directory) if file.endswith(".txt")]

# Load the guidance text
with open(guidance_file, "r") as file:
    guidance_text = file.read().split("\n")

# Load the corpus from multiple text files
corpus = []
for file_name in corpus_files:
    file_path = os.path.join(corpus_directory, file_name)
    with open(file_path, "r") as file:
        corpus.extend(file.read().split("\n"))

def clean_text(text):
    # Allow hyphens by adding '-' inside the allowed character set
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s-]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


corpus = [clean_text(prompt) for prompt in corpus if clean_text(prompt)]
guidance_text = [clean_text(prompt) for prompt in guidance_text if clean_text(prompt)]

# Tokenize the corpus and create a vocabulary
vocabulary = set()
tokenized_corpus = []
bpm_values = []

for prompt in corpus:
    tokens = prompt.split()
    tokenized_corpus.append(tokens)
    vocabulary.update(tokens)

    # Extract BPM value from the prompt
    bpm_match = re.search(r'\b\d+\s*BPM\b', prompt)
    if bpm_match:
        bpm = int(bpm_match.group().split()[0])
        bpm_values.append(bpm)

vocabulary = list(vocabulary)

# Generate training data
num_samples = 256
max_length = 16

generated_prompts = []
def assign_mood(polarity):
    if polarity > 0.2:
        mood_keywords = random.sample(mood_dict['positive'], 1)
    elif polarity < -0.2:
        mood_keywords = random.sample(mood_dict['negative'], 2)
    else:
        mood_keywords = random.sample(mood_dict['neutral'], 2)
    return ", ".join(mood_keywords)

for _ in range(num_samples):
    prompt = []
    prompt_length = random.randint(5, max_length)

    # Determine the minimum number of words from the corpus
    min_corpus_words = int(prompt_length * 0.3)  # 60% of the prompt length

    # Add words from the corpus to the prompt
    while len(prompt) < min_corpus_words:
        token = random.choice(vocabulary)
        prompt.append(token)

    # Add words from the guidance text to the prompt
    while len(prompt) < prompt_length:
        token = random.choice(guidance_text).split()
        token = random.choice(token)
        prompt.append(token)

    # Remove duplicate words, empty strings, and words with less than 3 characters
    prompt = list(set(prompt))
    prompt = [word for word in prompt if word.strip() and len(word) > 2]

    # ' BPM' 'BPM ', 'BPM' -> ''
    prompt = [re.sub(r'\s*BPM\s*', '', word) for word in prompt]

    # Clean numbers from alphabetic characters if no space between them
    prompt = [re.sub(r'\b(\d+)([a-zA-Z]+)\b', r'\1 \2', word) for word in prompt]

    # Perform sentiment analysis on the prompt
    prompt_text = " ".join(prompt)
    sentiment = TextBlob(prompt_text).sentiment
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity
    feelings = assign_mood(sentiment.polarity)

    # Capitalize the first word in the prompt
    prompt[0] = prompt[0].capitalize()

    # Clean and join the prompt
    prompt = " ".join(prompt)

    # Generate a random BPM value
    if bpm_values and random.random() < 0.5:
        bpm = random.choice(bpm_values)
    else:
        bpm = random.randint(70, 180)

    # Append BPM and feelings to the prompt
    prompt = f"{prompt}, {bpm} BPM, {feelings}"

    generated_prompts.append(prompt)
# Generate a hash for the generated prompts
hash_object = hashlib.md5("\n".join(generated_prompts).encode())
hash_value = hash_object.hexdigest()
short_hash = hash_value[:8]

prompt_file_name = f"generated_prompts_{short_hash}.txt"

# Save the generated prompts to a file
with open(prompt_file_name, "w") as file:
    file.write("\n".join(generated_prompts))

print(f"Generated prompts saved to {prompt_file_name}")

# Specify the model repository
model_repository = "facebook/musicgen-stereo-small"

# Create a cache directory for the downloaded model and processor
cache_dir = "cache"
os.makedirs(cache_dir, exist_ok=True)

# Check if the model and processor files are already cached
model_path = os.path.join(cache_dir, model_repository)
processor_path = os.path.join(cache_dir, model_repository)

try:
    # Try to load the cached model and processor
    processor = AutoProcessor.from_pretrained(processor_path)
    model = MusicgenForConditionalGeneration.from_pretrained(model_path)
    print(f"Loaded cached model and processor from {model_path}")
except OSError:
    # If the cached model and processor are not found, download them from the model repository
    processor = AutoProcessor.from_pretrained(model_repository, cache_dir=cache_dir)
    model = MusicgenForConditionalGeneration.from_pretrained(model_repository, cache_dir=cache_dir)
    print(f"Downloaded and cached model and processor from {model_repository}")

def generate_music(prompt, max_new_tokens, original_sampling_rate, target_sampling_rate):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    print(f"Moved inputs to: {device}")

    model.to(device)
    print(f"Moved model to: {device}")

    audio_values = model.generate(**inputs, do_sample=True, max_new_tokens=(random.randint(1000, 1500)))

    print("Generated audio values")

    audio_values_cpu = audio_values[0, 0].cpu()
    print("Moved audio values to: CPU")

    return audio_values_cpu.numpy()

def normalize_audio(audio, target_peak=0.95):
    peak = np.max(np.abs(audio))
    scaling_factor = target_peak / peak
    normalized_audio = audio * scaling_factor
    normalized_audio = np.float32(normalized_audio)  # Change data type to float32
    return normalized_audio

# Generate audio tracks for each prompt
used_prompts = set()
# Create a directory to store the generated audio files
output_directory = f"generated_audio_{short_hash}"
os.makedirs(output_directory, exist_ok=True)

# Generate audio tracks for each prompt
for i, prompt in enumerate(generated_prompts):
    print(f"Generating audio for prompt {i+1}/{len(generated_prompts)}: {prompt}")
    
    # Extract BPM from the prompt
    bpm_match = re.search(r'\b\d+\s*BPM\b', prompt)
    if bpm_match:
        bpm = int(bpm_match.group().split()[0])
    else:
        bpm = 120  # Default BPM if not found in the prompt
    
    # Generate music for the prompt
    
    section_audio = generate_music(prompt, None, 8000, 48000) # 8000, 16000, 22050, 44100, 48000 ( prompt, max_new_tokens, original_sampling_rate, target_sampling_rate)
    
    # Normalize the section audio
    normalized_section_audio = normalize_audio(section_audio)
        
    # Create a filename based on the first 64 characters of the prompt
    prompt_filename = re.sub(r'[<>:"/\\|?*]', '', prompt[:128]).strip()
    track_name = f"{prompt_filename}.wav"
    track_path = os.path.join(output_directory, track_name)

    # Save the audio file
    scipy.io.wavfile.write(track_path, rate=16000, data=normalized_section_audio)
    print(f"Audio saved as {track_path}")

    # Load the WAV file using pydub
    audio_segment = AudioSegment.from_wav(track_path)

    # Apply fade out to the audio segment
    fade_out_duration = 5000  # Fade out duration in milliseconds (5 seconds)
    faded_audio_segment = audio_segment.fade_out(fade_out_duration)

    # Save the faded audio as WAV file
    faded_track_name = f"{prompt_filename}_faded.wav"
    faded_track_path = os.path.join(output_directory, faded_track_name)
    faded_audio_segment.export(faded_track_path, format="wav")
    print(f"Faded audio saved as {faded_track_path}")
    
    # Save the faded audio as MP3 file with metadata
    faded_mp3_track_name = f"{prompt_filename}.mp3"
    faded_mp3_track_path = os.path.join(output_directory, faded_mp3_track_name)
    
    # Split the prompt into artist and track name
    prompt_parts = prompt.split(', ')
    artist = ', '.join(prompt_parts[:len(prompt_parts)//2])
    track_name = ', '.join(prompt_parts[len(prompt_parts)//2:])
    
    # Select a random word from the prompt as the genre
    genre = random.choice(prompt.split())
    
    # Add metadata to the MP3 file
    faded_audio_segment.export(faded_mp3_track_path, format="mp3", tags={
        "artist": artist,
        "title": track_name,
        "genre": genre
    })
    
    print(f"Faded audio saved as {faded_mp3_track_path}")

print("Audio generation completed.")

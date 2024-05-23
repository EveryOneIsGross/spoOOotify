from pydub import AudioSegment
from pydub.playback import play
import threading
import time
import keyboard
import pandas as pd
import os
import random
import pickle
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import eyed3
import argparse

# Configuration and Initialization
PICKLE_FILE = "my_ituneslibrary2024.pkl"  # Where the database is stored

df = None
current_track = None
played_tracks = set()
playback_status = "stopped"
playback_threads = []
stop_playback_event = threading.Event()

def scan_directory(directory):
    tracks = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp3'):
                path = os.path.join(root, file)
                try:
                    audio = eyed3.load(path)
                    if audio and audio.tag:
                        artist = audio.tag.artist if audio.tag.artist else "Unknown Artist"
                        album = audio.tag.album if audio.tag.album else "Unknown Album"
                        title = audio.tag.title if audio.tag.title else "Unknown Title"
                    else:
                        raise ValueError("Missing tag info")
                except:
                    tokens = file.replace('.mp3', '').replace('_', ' ').replace('-', ' ').split()
                    artist = "Unknown Artist"
                    if not tokens:
                        title = "Unknown Title"
                        album = "Unknown Album"
                    else:
                        # Randomly assign words from the filename to title and album
                        num_title_words = random.randint(1, len(tokens))
                        title_words = random.sample(tokens, num_title_words)
                        title = ' '.join(title_words)
                        album_words = [word for word in tokens if word not in title_words]
                        if album_words:
                            album = ' '.join(album_words)
                        else:
                            album = "Unknown Album"
                tracks.append((path, file, artist, album, title))
    return tracks

def tokenize(text):
    return nltk.word_tokenize(text)

def sentiment_score(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def extract_topics_sentiments(filename, artist, album, title):
    filename_sentiment = sentiment_score(filename)
    metadata_sentiment = sentiment_score(" ".join([artist, album, title]))
    avg_sentiment = (filename_sentiment + metadata_sentiment) / 2
    if -0.1 < avg_sentiment < 0.1:
        sentiment_label = "Neutral"
    elif avg_sentiment >= 0.1:
        sentiment_label = "Positive"
    else:
        sentiment_label = "Negative"
    topics = " ".join(tokenize(" ".join([filename, artist, album, title])))
    return topics, sentiment_label

def create_dataframe(tracks):
    data = []
    for path, filename, artist, album, title in tracks:
        topics, sentiment = extract_topics_sentiments(filename, artist, album, title)
        data.append([path, filename, artist, album, title, topics, sentiment])
    return pd.DataFrame(data, columns=["path", "filename", "artist", "album", "title", "topics", "sentiment"])

def save_to_pickle(df):
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(df, f)

def load_or_create_db(directory):
    global df
    if os.path.exists(PICKLE_FILE):
        with open(PICKLE_FILE, 'rb') as f:
            df = pickle.load(f)
    else:
        tracks = scan_directory(directory)
        df = create_dataframe(tracks)
        save_to_pickle(df)
    print(df)

def search_next_track(current_track_text, remaining_tracks, context):
    tfidf = TfidfVectorizer()
    track_matrix = tfidf.fit_transform(remaining_tracks['topics'].tolist() + [current_track_text])
    context_matrix = tfidf.transform([context])
    
    track_cosine_similarities = cosine_similarity(track_matrix[-1], track_matrix[:-1])
    context_cosine_similarities = cosine_similarity(context_matrix, track_matrix[:-1])
    
    combined_similarities = track_cosine_similarities[0] + context_cosine_similarities[0]
    next_track_index = combined_similarities.argsort()[-1]
    
    return remaining_tracks.iloc[next_track_index]

def play_track(track, context):
    global current_track, played_tracks, playback_status

    try:
        print(f"Now playing: {track['sentiment']} - {track['filename']} - {track['artist']} - {track['title']}")
        audio_file = track['path']

        # Using pydub to play the audio
        audio = AudioSegment.from_mp3(audio_file)
        play_thread = threading.Thread(target=play, args=(audio,))
        play_thread.start()

        track_length = len(audio) / 1000
        time.sleep(track_length - 5)  # Adjust this value based on overlap

        remaining_tracks = df[~df['path'].isin(played_tracks)]
        if not remaining_tracks.empty:
            next_track = search_next_track(track['topics'], remaining_tracks, context)
            if next_track is not None and playback_status == "playing":
                played_tracks.add(next_track['path'])
                play_track(next_track, context)
            else:
                playback_status = "stopped"
        else:
            # Reset played tracks and start over
            played_tracks.clear()
            remaining_tracks = df
            current_track = remaining_tracks.sample(n=1).iloc[0]
            played_tracks.add(current_track['path'])
            if playback_status == "playing":
                play_track(current_track, context)
    except Exception as e:
        print(f"Error during playback: {e}")
        playback_status = "stopped"

def start_playback(context):
    global current_track, playback_status
    if playback_status == "stopped":
        remaining_tracks = df[~df['path'].isin(played_tracks)]
        if not remaining_tracks.empty:
            current_track = remaining_tracks.sample(n=1).iloc[0]
            played_tracks.add(current_track['path'])
            playback_thread = threading.Thread(target=play_track, args=(current_track, context), daemon=True)
            playback_thread.start()
            playback_status = "playing"
        else:
            print("No remaining tracks to play.")

def stop_playback():
    global playback_status
    playback_status = "stopped"
    # Stop all playback threads
    stop_playback_event.set()

def main(directory, context):
    load_or_create_db(directory)

    print("Auto Semantic Jukebox")

    # Start playback automatically when the program starts
    start_playback(context)

    print("Playback started. Press 'q' to quit.")

    try:
        while True:
            if keyboard.is_pressed('q'):
                stop_playback()
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_playback()
        print("Exiting...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Semantic Jukebox")
    parser.add_argument("directory", help="Path to the directory containing MP3 files")
    parser.add_argument("context", help="Search context prompt as a string")
    args = parser.parse_args()

    main(args.directory, args.context)

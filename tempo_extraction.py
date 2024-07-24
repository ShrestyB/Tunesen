import os
import librosa
import csv

print("Extracting tempo segments...")
count = 0  

def extract_tempo_segments(filename):
    try:
        y, sr = librosa.load(filename)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return [], []
    
    segment_tempo = []
    timestamps = []
    
    song_length = len(y) / sr 
    
    if song_length < 10:
        timestamps = [0]
    else:
        for i in range(0, len(y), sr * 10): 
            segment = y[i:i + sr * 10]
            tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
            segment_tempo.append(tempo)
            if i == 0:
                timestamps.append(0)
            else:
                timestamps.append(timestamps[-1] + 10) 
    
    return timestamps, segment_tempo

def save_tempo_to_csv(filename, song_name, timestamps, tempo_segments):
    global count 
    save_path = os.path.join('Extracted_Tempos', filename)
    with open(save_path, 'a', newline='') as file:
        writer = csv.writer(file)
        
        if os.path.getsize(save_path) == 0:
            writer.writerow(["Name of song", "Start timestamp", "Tempo"])
        
        for i in range(len(tempo_segments)):
            writer.writerow([song_name, timestamps[i], tempo_segments[i]])
            count += 1
            print(f"Wrote {count} segment")

music_folder = '../Ludwig Music Dataset (Moods and Subgenres)/mp3/mp3'

if not os.path.exists('Extracted_Tempos'):
    os.makedirs('Extracted_Tempos')

for filename in os.listdir(music_folder):
    print(filename)
    if filename.endswith('.mp3'):
        filepath = os.path.join(music_folder, filename)
        timestamps, tempo_segments = extract_tempo_segments(filepath)
        
        if timestamps and tempo_segments:  # Check if lists are not empty
            save_tempo_to_csv('all_songs_tempo_segments.csv', filename, timestamps, tempo_segments)

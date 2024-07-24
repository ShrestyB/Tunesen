import numpy as np
import librosa
import tensorflow as tf
import soundfile as sf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os

print("Loading tempo data...")
tempo_data = pd.read_csv('Extracted_Tempos\\tempo_data.csv')

def extract_clips(file_name, start_time, duration):
    try:
        clip, _ = librosa.load(f'A:\\Class Material\\Artificial Intelligence Lab\\PROJECT\\Ludwig Music Dataset (Moods and Subgenres)\\mp3\\mp3\\{file_name}', sr=None, offset=start_time, duration=duration)
        return clip
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return np.array([])

def map_tempo_to_mood(tempo):
    if 160 <= tempo <= 200:
        return 'Happy'
    elif 120 <= tempo < 160:
        return 'Exuberant'
    elif 90 <= tempo < 120:
        return 'Energetic'
    elif 70 <= tempo < 90:
        return 'Frantic'
    elif 50 <= tempo < 70:
        return 'Sad'
    else:
        return 'Unknown'

print("Mapping tempo to mood...")
tempo_data['Mood'] = tempo_data['Tempo'].apply(map_tempo_to_mood)

bce_loss = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real, fake):
    real_loss = bce_loss(tf.ones_like(real), real)
    fake_loss = bce_loss(tf.zeros_like(fake), fake)
    total_loss = real_loss + fake_loss
    return total_loss
   
def generator_loss(preds):
    return bce_loss(tf.ones_like(preds), preds)
    
d_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

print("Building generator...")
def build_generator(input_shape):
    model = Sequential()
    model.add(Dense(256, input_shape=input_shape, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(input_shape[0], activation='tanh'))
    return model

print("Building discriminator...")
def build_discriminator(input_shape):
    model = Sequential()
    model.add(Dense(1024, input_shape=input_shape, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

print("Building GAN...")
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

input_shape = (100,)

generator = build_generator(input_shape)
generator.summary()
discriminator = build_discriminator(input_shape)
discriminator.summary()
discriminator.compile(optimizer=d_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

gan = build_gan(generator, discriminator)
gan.compile(optimizer=g_optimizer, loss='binary_crossentropy')

def generate_music(mood, generator, tempo_data, duration):
    print("Generating music...")
    mood_data = tempo_data[tempo_data['Mood'] == mood]
    
    selected_clips_data = mood_data.sample(n=3)
    selected_clips = selected_clips_data.apply(lambda x: extract_clips(x['Name of song'], x['Start timestamp'], duration), axis=1).tolist()
    
    generated_clips = generator.predict(np.random.normal(0, 1, (3, 100)))
    
    mixed_clips = [((generated_clip / 2).flatten()) for generated_clip in generated_clips]
    
    new_song = np.concatenate([np.concatenate(generated) for generated in zip(selected_clips, mixed_clips)])
    
    return new_song

def save_new_song(new_song, mood):
    output_file = f'Generated_clips\\{mood}_song.wav'

    if os.path.exists(output_file):
        os.remove(output_file)
    sf.write(output_file, new_song, 44100, 'PCM_24')
    print(f"New song for {mood} mood saved to {output_file}")

def train_gan(generator, discriminator, gan, tempo_data, duration_seconds, epochs, batch_size=32, sample_rate=44100):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for _ in range(batch_size):
            
            real_clips_data = tempo_data.sample(batch_size)
            real_clips = real_clips_data.apply(lambda x: extract_clips(x['Name of song'], x['Start timestamp'], duration_seconds), axis=1).tolist()

            max_length = int(duration_seconds * sample_rate)
            real_clips = [librosa.util.fix_length(clip, size=max_length) for clip in real_clips]
            real_clips = [clip.reshape((-1,))[:100] for clip in real_clips] 

            noise = np.random.normal(0, 1, (batch_size, 100))
            fake_clips = generator.predict(noise)

            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            d_loss_real = discriminator.train_on_batch(np.array(real_clips), real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_clips, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, 100))
            valid_labels = np.ones((batch_size, 1))

            g_loss = gan.train_on_batch(noise, valid_labels)

        print(f"Epoch {epoch + 1}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

epochs = 10
batch_size = 5
duration_seconds = 1200

train_gan(generator, discriminator, gan, tempo_data, duration_seconds, epochs, batch_size)

generator.summary()
discriminator.summary()

print('Moods : Happy, Exuberant, Energetic, Frantic, Sad')
mood = str(input("Mood : "))
new_song = generate_music(mood, generator, tempo_data, duration_seconds)
save_new_song(new_song, mood)

print("Done!")

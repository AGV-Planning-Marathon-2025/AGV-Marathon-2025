import moviepy.editor as mp

# ==== FILE PATHS ====
main_video_path = "document_6172644345460562886.mp4"
music_path = "Sport Rock Energy Racing by Infraction [No Copyright Music]  Powerlifting.mp3"
output_path = "final_youtube_video.mp4"

# ==== INTRO SETTINGS ====
intro_duration = 6  # seconds (between 5–7)
W, H = 1280, 720    # output resolution

# Background (techy dark gradient)
intro_bg = mp.ColorClip(size=(W, H), color=(10, 10, 30), duration=intro_duration)

# Text with glowing/techy effect
txt = mp.TextClip(
    "AGV Planning Subteam",
    method="caption",
    size=(W - 100, None),
    color="cyan",
    font="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    fontsize=70
)
txt = txt.set_position("center").set_duration(intro_duration).crossfadein(1).crossfadeout(1)

# Combine text and background
intro = mp.CompositeVideoClip([intro_bg, txt])

# ==== MAIN VIDEO ====
main_video = mp.VideoFileClip(main_video_path).resize((W, H)).without_audio()

# ==== CONCATENATE INTRO + MAIN VIDEO ====
final_video = mp.concatenate_videoclips([intro, main_video], method="compose")

# ==== AUDIO ====
music = mp.AudioFileClip(music_path)
from moviepy.audio.fx.all import audio_loop
music = audio_loop(music, duration=final_video.duration)

# Fade volume: normal in intro, reduced in main video
music_intro = music.subclip(0, intro_duration).volumex(1.0)
music_main = music.subclip(intro_duration, final_video.duration).volumex(0.3)

# Concatenate audio segments for the proper length
from moviepy.audio.AudioClip import concatenate_audioclips
final_audio = concatenate_audioclips([music_intro, music_main])

# Set the composite audio to final video
final_video = final_video.set_audio(final_audio)

# ==== EXPORT ====
final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=30)

print("✅ Final video exported:", output_path)

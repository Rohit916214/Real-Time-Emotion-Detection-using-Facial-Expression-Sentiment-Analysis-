# from flask import Flask, render_template, Response, jsonify
# import cv2
# from deepface import DeepFace
# import whisper as whisper
# from flask import jsonify
# from transformers import pipeline
# import sounddevice as sd
# import numpy as np
# import soundfile as sf 

# app = Flask(__name__)

# # Load models
# whisper_model = whisper.load_model("base")
# sentiment_pipeline = pipeline("sentiment-analysis")

# # Global variables
# current_emotion = "neutral"
# current_sentiment = {"label": "NEUTRAL", "score": 0.5}

# def analyze_frame(frame):
#     global current_emotion
#     try:
#         result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
#         current_emotion = result[0]['dominant_emotion']
#     except:
#         pass

# def analyze_audio():
#     global current_sentiment
#     fs = 44100  # Sample rate
#     duration = 3  # Record for 3 seconds
#     print("Recording...")
#     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
#     sd.wait()
#     audio = np.squeeze(audio)
#     audio_path = "temp_audio.wav"
#     sf.write(audio_path, audio, fs)
#     text = whisper_model.transcribe(audio_path)["text"]
#     if text.strip():
#         current_sentiment = sentiment_pipeline(text)[0]
#     return text


# def gen_frames():
#     cap = cv2.VideoCapture(0)
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         analyze_frame(frame)
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_sentiment')
# def get_sentiment():
#     text = analyze_audio()
#     return jsonify({
#         "emotion": current_emotion,
#         "text": text,
#         "sentiment": current_sentiment
#     })

# @app.route('/')
# def index():
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)






# -------------------------------------------------------------------------







# from flask import Flask, render_template, Response, jsonify, request
# import cv2
# from deepface import DeepFace
# import whisper
# from transformers import pipeline
# import sounddevice as sd
# import numpy as np
# import soundfile as sf

# app = Flask(__name__)

# # Global variables
# current_emotion = "Neutral"
# current_sentiment = {"label": "NEUTRAL", "score": 0.0}
# camera_running = True  # Track camera status

# # Lazy load Whisper model
# def get_whisper_model():
#     if not hasattr(get_whisper_model, "model"):
#         get_whisper_model.model = whisper.load_model("base")
#     return get_whisper_model.model

# # Lazy load Sentiment model
# def get_sentiment_pipeline():
#     if not hasattr(get_sentiment_pipeline, "pipe"):
#         get_sentiment_pipeline.pipe = pipeline("sentiment-analysis")
#     return get_sentiment_pipeline.pipe

# # Analyze a video frame for facial emotion
# def analyze_frame(frame):
#     global current_emotion
#     try:
#         result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
#         current_emotion = result[0]['dominant_emotion']
#     except Exception as e:
#         print("Error analyzing frame:", e)
#         current_emotion = "Neutral"

# # Analyze live microphone audio for sentiment
# def analyze_microphone_audio():
#     global current_sentiment
#     fs = 44100  # Sample rate
#     duration = 3  # seconds
#     print("Recording audio...")
#     try:
#         audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
#         sd.wait()
#         audio = np.squeeze(audio)
#         audio_path = "temp_audio.wav"
#         sf.write(audio_path, audio, fs)

#         whisper_model = get_whisper_model()
#         result = whisper_model.transcribe(audio_path)
#         text = result["text"]

#         print(f"Transcribed text: {text}")

#         if text.strip():
#             sentiment_pipe = get_sentiment_pipeline()
#             sentiment_result = sentiment_pipe(text)[0]
#             current_sentiment = sentiment_result
#         else:
#             current_sentiment = {"label": "NEUTRAL", "score": 0.0}

#         return text

#     except Exception as e:
#         print("Error analyzing microphone audio:", e)
#         current_sentiment = {"label": "ERROR", "score": 0.0}
#         return ""

# # Video streaming generator
# def gen_frames():
#     global camera_running
#     cap = cv2.VideoCapture(0)
#     while camera_running:
#         success, frame = cap.read()
#         if not success:
#             break
#         analyze_frame(frame)
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#     cap.release()

# # Routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     global camera_running
#     camera_running = True
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_sentiment', methods=['GET', 'POST'])
# def get_sentiment():
#     try:
#         text = analyze_microphone_audio()
#         return jsonify({
#             'emotion': current_emotion,
#             'sentiment': current_sentiment,
#             'text': text
#         })
#     except Exception as e:
#         print("Error in /get_sentiment:", e)
#         return jsonify({
#             'emotion': "Error",
#             'sentiment': {'label': 'Error', 'score': 0.0},
#             'text': ""
#         })

# @app.route('/stop_camera')
# def stop_camera():
#     global camera_running
#     camera_running = False
#     return "Camera stopped."

# @app.route('/start_camera')
# def start_camera():
#     global camera_running
#     camera_running = True
#     return "Camera started."

# @app.route('/shutdown', methods=['POST'])
# def shutdown():
#     func = request.environ.get('werkzeug.server.shutdown')
#     if func is None:
#         raise RuntimeError('Not running with the Werkzeug Server')
#     func()
#     return 'Server shutting down...'

# # Analyze uploaded audio file (new separate route)
# # @app.route('/analyze_audio', methods=['POST'])
# # def analyze_uploaded_audio():
# #     if 'file' not in request.files:
# #         return jsonify({'error': 'No file uploaded'}), 400

# #     file = request.files['file']

# #     try:
# #         # Save the uploaded file temporarily
# #         temp_audio_path = "uploaded_temp_audio.wav"
# #         file.save(temp_audio_path)

# #         # Transcribe using Whisper
# #         whisper_model = get_whisper_model()
# #         result = whisper_model.transcribe(temp_audio_path)
# #         transcribed_text = result["text"]

# #         # Analyze the text sentiment
# #         sentiment_pipe = get_sentiment_pipeline()
# #         sentiment = sentiment_pipe(transcribed_text)[0]

# #         return jsonify({'text': transcribed_text, 'sentiment': sentiment})
# #     except Exception as e:
# #         print("Error in /analyze_audio:", e)
# #         return jsonify({'error': str(e)}), 500

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)














# --------------------------------------------------------

from flask import Flask, render_template, Response, jsonify, request
import cv2
from deepface import DeepFace

app = Flask(__name__)

# Global variables
current_emotion = "Neutral"
camera_running = True  # Track camera status

# Analyze a video frame for facial emotion
def analyze_frame(frame):
    global current_emotion
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        current_emotion = result[0]['dominant_emotion']
    except Exception as e:
        print("Error analyzing frame:", e)
        current_emotion = "Neutral"

# Video streaming generator
def gen_frames():
    global camera_running
    cap = cv2.VideoCapture(0)
    while camera_running:
        success, frame = cap.read()
        if not success:
            break
        analyze_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global camera_running
    camera_running = True
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion', methods=['GET'])
def get_emotion():
    return jsonify({'emotion': current_emotion})

@app.route('/stop_camera')
def stop_camera():
    global camera_running
    camera_running = False
    return "Camera stopped."

@app.route('/start_camera')
def start_camera():
    global camera_running
    camera_running = True
    return "Camera started."

@app.route('/shutdown', methods=['POST'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

if __name__ == '__main__':
    app.run(debug=True)

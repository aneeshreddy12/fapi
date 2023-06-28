from flask import Flask,request,jsonify
import tensorflow as tf
import librosa
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('cnn_mfcc_10')

list=["TB Negative","TB Positive"]
def func(filename):
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features = np.repeat(mfccs_scaled_features, 1, axis=0)

    mfccs_scaled_features=mfccs_scaled_features.reshape(1, 8, 5, 1)


    predicted_label=np.argmax(model.predict(mfccs_scaled_features), axis=-1)
    # print(predicted_label)
    return list[predicted_label[0]]

@app.route('/predict',methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return 'No file provided', 400

    audio_file = request.files['audio']
    if not audio_file.filename.lower().endswith('.wav'):
        return 'Invalid file type, must be .wav', 400
    preditction = func(audio_file)
    print(preditction)
    return preditction

if __name__ == '__main__':
    app.run(debug=True)
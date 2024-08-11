from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import librosa
import yaml
from model import RawNet
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def pad(x, max_len=96000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def load_sample(sample_path, max_len=96000):
    y_list = []
    y, sr = librosa.load(sample_path, sr=None)
    if sr != 24000:
        y = librosa.resample(y, orig_sr=sr, target_sr=24000)
    if len(y) <= 96000:
        return [Tensor(pad(y, max_len))]
    for i in range(int(len(y) / 96000)):
        if (i + 1) == range(int(len(y) / 96000)):
            y_seg = y[i * 96000:]
        else:
            y_seg = y[i * 96000:(i + 1) * 96000]
        y_pad = pad(y_seg, max_len)
        y_inp = Tensor(y_pad)
        y_list.append(y_inp)
    return y_list

# Load model configuration
dir_yaml = 'model_config_RawNet.yaml'
with open(dir_yaml, 'r') as f_yaml:
    model_config = yaml.safe_load(f_yaml)

# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = RawNet(model_config['model'], device).to(device)
model_path = 'librifake_pretrained_lambda0.5_epoch_25.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        file_path = temp.name
        file.save(file_path)

    # Process the file
    try:
        out_list_multi = []
        out_list_binary = []
        for m_batch in load_sample(file_path):
            m_batch = m_batch.to(device=device, dtype=torch.float).unsqueeze(0)
            logits, multi_logits = model(m_batch)
            probs = F.softmax(logits, dim=-1)
            probs_multi = F.softmax(multi_logits, dim=-1)
            out_list_multi.append(probs_multi.tolist()[0])
            out_list_binary.append(probs.tolist()[0])

        result_multi = np.average(out_list_multi, axis=0).tolist()
        result_binary = np.average(out_list_binary, axis=0).tolist()
    finally:
        os.remove(file_path)  # Clean up the temporary file

    return jsonify({
        'multi_classification': {
            'gt': result_multi[0],
            'wavegrad': result_multi[1],
            'diffwave': result_multi[2],
            'parallel_wave_gan': result_multi[3],
            'wavernn': result_multi[4],
            'wavenet': result_multi[5],
            'melgan': result_multi[6]
        },
        'binary_classification': {
            'fake': result_binary[0],
            'real': result_binary[1]
        }
    })

if __name__ == '__main__':
    app.run(debug=True)

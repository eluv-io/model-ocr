import os
import shutil
import subprocess

def load_caption():
    res = subprocess.run(['bash', 'pull-models'], stdout=None, stderr=None)
    if res.returncode != 0:
        raise RuntimeError("Failed to pull models")

    print('Cleaning up models directory')
    for model in os.listdir('models'):
        if model != 'ocr':
            shutil.rmtree(f'models/{model}')

if not os.path.exists('models/ocr'):
    load_caption()
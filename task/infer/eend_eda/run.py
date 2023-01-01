from pathlib import Path
import torch
from librosa import load
import soundfile as sf

from infer_scr.eend_eda_model import EENDEDAModel

def main():

	WAVPATH = ''
	MODELPATH = 'valid.si_snr_loss.best_old.pth'	
	model = EENDEDAModel()

	model.load_state_dict(torch.load(MODELPATH))
	wavfile, sample_rate = load(WAVPATH, sr = 8000)
	
	wavfile = torch.from_numpy(wavfile).unsqueeze(0)	
	diar_pred = model(wavfile)
	
if __name__ == '__main__':

	main()
import os

import torch
import torch.nn as nn

from .content_encoder import ContentEncoder
from .pitch_estimator import PitchEstimator
from .decoder import Decoder
from .audio import PerceptualLoudness
from .common import match_features, compute_f0
from .excitation import generate_excitation


# for realtime inferencing
class Convertor(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_encoder = ContentEncoder().eval()
        self.pitch_estimator = PitchEstimator().eval()
        self.decoder = Decoder().eval()
        self.frame_size = self.decoder.frame_size
        if self.content_encoder.n_fft != self.pitch_estimator.n_fft:
            raise ValueError("content and pitch analysis windows must match")
        self.lookahead_samples = (
            self.content_encoder.n_fft // 2
            + self.frame_size
            + self.decoder.lookahead_samples
        )
        self.loudness_frame_size = self.decoder.loudness_frame_size
        self.sample_rate = self.decoder.sample_rate
        self.loudness_extractor = PerceptualLoudness(
            self.sample_rate,
            self.loudness_frame_size,
            self.decoder.loudness_n_fft,
        )

    def load(self, path='./models', device='cpu'):
        self.pitch_estimator.load_state_dict(torch.load(os.path.join(path, 'pitch_estimator.pt'), map_location=device, weights_only=True))
        self.content_encoder.load_state_dict(torch.load(os.path.join(path, 'content_encoder.pt'), map_location=device, weights_only=True))
        self.decoder.load_state_dict(torch.load(os.path.join(path, 'decoder.pt'), map_location=device, weights_only=True))

    @torch.inference_mode()
    def encode_target(self, wave, stride=4):
        tgt = self.content_encoder.encode(wave)
        return tgt[:, :, ::stride]

    # convert single waveform without buffering
    @torch.inference_mode()
    def convert(self, wave, tgt, pitch_shift=0, k=4, alpha=0, pitch_estimation_algorithm='default'):
        z = self.content_encoder.encode(wave)

        z = match_features(z, tgt, k, alpha)
        l = self.loudness_extractor(wave)
        if pitch_estimation_algorithm != 'default':
            p = compute_f0(wave, algorithm=pitch_estimation_algorithm)
        else:
            p = self.pitch_estimator.estimate(wave)
        scale = 12 * torch.log2(p / 440)
        scale += pitch_shift
        p = 440 * 2 ** (scale / 12)
        return self.decoder.synthesize(z, p, l)

    # initialize buffer for realtime inferencing
    @torch.inference_mode()
    def init_buffer(self, buffer_size, device='cpu'):
        if buffer_size % self.frame_size != 0:
            raise ValueError("buffer size must be a multiple of the decoder frame size")
        if buffer_size < self.lookahead_samples:
            raise ValueError("buffer size must cover the analysis lookahead")
        audio_buffer = torch.zeros(1, buffer_size, device=device)
        source_buffer = torch.zeros(1, 1, buffer_size, device=device)
        phase_buffer = torch.zeros(1, 1, 1, device=device)
        return audio_buffer, source_buffer, phase_buffer
    
    # convert voice with buffer for realtime inferencing
    @torch.inference_mode()
    def convert_rt(self, chunk, buffer, tgt, pitch_shift, k=4, alpha=0, pitch_estimation='default'):
        k = int(k)

        # extpand buffer variables
        audio_buffer, source_buffer, phase_buffer = buffer

        # buffer size and chunk size
        buffer_size = audio_buffer.shape[1]
        chunk_size = chunk.shape[1]
        if chunk_size % self.frame_size != 0:
            raise ValueError("chunk size must be a multiple of the decoder frame size")
        # concateante audio buffer and chunk
        x = torch.cat([audio_buffer, chunk], dim=1)

        # encode content, estimate loudness, estimate pitch
        z = self.content_encoder.encode(x)
        if pitch_estimation == 'default':
            p = self.pitch_estimator.estimate(x)
        else:
            p = compute_f0(x, algorithm=pitch_estimation)
        e = self.loudness_extractor(x)

        # convert style
        z = match_features(z, tgt, k, alpha)

        # pitch shift
        scale = 12 * torch.log2(p / 440)
        scale += pitch_shift
        p = 440 * 2 ** (scale / 12)

        current_frame_count = chunk_size // self.frame_size
        current_source, new_phase_buffer = generate_excitation(
                p[:, :, -current_frame_count:],
                phase_buffer,
                self.frame_size,
                self.sample_rate)
        source_signal = torch.cat([source_buffer, current_source], dim=2)
        
        # synthesize new voice
        y = self.decoder(z, e, source_signal)

        # return new voice and shift left
        audio_out = y[:, buffer_size-self.lookahead_samples:-self.lookahead_samples]
        new_audio_buffer = x[:, -buffer_size:]
        new_source_buffer = source_signal[:, :, -buffer_size:]

        return audio_out, (new_audio_buffer, new_source_buffer, new_phase_buffer)

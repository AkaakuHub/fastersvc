import unittest

import torch

from module.convertor import Convertor


class StreamingBufferTest(unittest.TestCase):
    def test_target_encoding_does_not_retain_training_graphs(self):
        convertor = Convertor()
        waveform = torch.randn(1, 1920, requires_grad=True)

        target = convertor.encode_target(waveform)

        self.assertFalse(target.requires_grad)

    def test_keeps_audio_source_and_phase_state(self):
        convertor = Convertor()

        audio, source, phase = convertor.init_buffer(7680)

        self.assertEqual(audio.shape, (1, 7680))
        self.assertEqual(source.shape, (1, 1, 7680))
        self.assertEqual(phase.shape, (1, 1, 1))

    def test_rejects_buffer_not_aligned_to_decoder_frames(self):
        convertor = Convertor()
        with self.assertRaises(ValueError):
            convertor.init_buffer(1000)

    def test_rejects_buffer_shorter_than_analysis_lookahead(self):
        convertor = Convertor()
        self.assertEqual(convertor.lookahead_samples, 5681)
        with self.assertRaises(ValueError):
            convertor.init_buffer(1920)


if __name__ == "__main__":
    unittest.main()

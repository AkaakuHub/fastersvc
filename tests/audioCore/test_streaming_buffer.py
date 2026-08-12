import unittest

from module.convertor import Convertor


class StreamingBufferTest(unittest.TestCase):
    def test_keeps_audio_source_and_phase_state(self):
        convertor = Convertor()

        audio, source, phase = convertor.init_buffer(1920)

        self.assertEqual(audio.shape, (1, 1920))
        self.assertEqual(source.shape, (1, 1, 1920))
        self.assertEqual(phase.shape, (1, 1, 1))

    def test_rejects_buffer_not_aligned_to_decoder_frames(self):
        convertor = Convertor()
        with self.assertRaises(ValueError):
            convertor.init_buffer(1000)


if __name__ == "__main__":
    unittest.main()

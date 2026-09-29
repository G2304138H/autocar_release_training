import unittest

from src.modules.sparse_limits import check_spconv_feature_size


class SpconvFeatureLimitTests(unittest.TestCase):
    def test_reported_single_view_case_exceeds_decoder_limit(self):
        with self.assertRaisesRegex(RuntimeError, "feature_bytes=2213404160"):
            check_spconv_feature_size(4_323_055, 128, 4, stage="decoder")

    def test_fp32_limit_is_strict(self):
        check_spconv_feature_size(4_194_303, 128, 4, stage="decoder")
        with self.assertRaisesRegex(RuntimeError, "int32 range"):
            check_spconv_feature_size(4_194_304, 128, 4, stage="decoder")

    def test_check_uses_actual_channel_width_and_dtype_size(self):
        check_spconv_feature_size(4_323_055, 96, 4, stage="decoder output")
        check_spconv_feature_size(4_323_055, 128, 2, stage="half precision")
        with self.assertRaisesRegex(RuntimeError, "int32 range"):
            check_spconv_feature_size(1_048_576, 512, 4, stage="coarse skip")


if __name__ == "__main__":
    unittest.main()

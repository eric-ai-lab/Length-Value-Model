import types
import unittest

# Import by file path to avoid importing the whole `sglang` package (which may require
# optional deps like numpy/torch in minimal environments).
import importlib.util
from pathlib import Path


_UTILS_PATH = (
    Path(__file__).resolve().parents[1] / "srt" / "lvm" / "lvm_value_utils.py"
)
_spec = importlib.util.spec_from_file_location("lvm_value_utils", _UTILS_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

force_eos_value_zero = _mod.force_eos_value_zero


class TestLvmGuidedSamplingEosValue(unittest.TestCase):
    def test_force_eos_value_zero_prefers_req_eos_token_ids(self):
        req = types.SimpleNamespace(
            eos_token_ids={2, 3},
            tokenizer=types.SimpleNamespace(eos_token_id=999),
        )
        token_ids = [10, 2, 11, 3]
        token_values = [0.9, 0.8, 0.7, 0.6]

        force_eos_value_zero(token_ids, token_values, req)
        self.assertEqual(token_values, [0.9, 0.0, 0.7, 0.0])

    def test_force_eos_value_zero_falls_back_to_tokenizer_eos_token_id(self):
        req = types.SimpleNamespace(
            eos_token_ids=None,
            tokenizer=types.SimpleNamespace(eos_token_id=2),
        )
        token_ids = [2, 5]
        token_values = [0.42, 0.11]

        force_eos_value_zero(token_ids, token_values, req)
        self.assertEqual(token_values, [0.0, 0.11])


if __name__ == "__main__":
    unittest.main()


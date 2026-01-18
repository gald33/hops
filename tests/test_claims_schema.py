import unittest

from hops.claims_schema import ClaimsSchemaError, load_claims_block


class TestClaimsSchema(unittest.TestCase):
    def test_duplicate_ids(self) -> None:
        block = {
            "claims": [
                {"id": "C1", "scope": "package", "metrics": [{"name": "final_welfare"}]},
                {"id": "C1", "scope": "package", "metrics": [{"name": "final_welfare"}]},
            ]
        }
        with self.assertRaises(ClaimsSchemaError):
            load_claims_block(block, base_dir=".")

    def test_unknown_key(self) -> None:
        block = {"unknown": 1}
        with self.assertRaises(ClaimsSchemaError):
            load_claims_block(block, base_dir=".")


if __name__ == "__main__":
    unittest.main()

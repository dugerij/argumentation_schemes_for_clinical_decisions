import unittest

from helpers.config import parse_optional_int


class ConfigParsingTests(unittest.TestCase):
    def test_parse_optional_int_accepts_all_aliases(self):
        for value in ("all", "none", "null", "unlimited"):
            self.assertIsNone(parse_optional_int(value, default=25))

    def test_parse_optional_int_accepts_integers_and_defaults(self):
        self.assertEqual(parse_optional_int("42", default=25), 42)
        self.assertEqual(parse_optional_int("", default=25), 25)


if __name__ == "__main__":
    unittest.main()

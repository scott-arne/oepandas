import unittest


class TestVersion(unittest.TestCase):
    def test_version(self):
        """
        Example test
        """
        from oepandas import __version__
        self.assertIsNotNone(__version__)

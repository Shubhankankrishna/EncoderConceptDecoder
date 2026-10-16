import unittest
from .clean_math import clean_math_field


class MathNormalizationTests(unittest.TestCase):
    def test_native_and_serialized_fields(self):
        for value in (b'10\n','10\n',"b'10\\n'",'b"10\\n"'):
            self.assertEqual(clean_math_field(value),'10')
        self.assertEqual(clean_math_field("b'What is -2 + 3?\\n'"),'What is -2 + 3?')
        self.assertEqual(clean_math_field('a - b'),'a - b')
        self.assertEqual(clean_math_field("b\"What's 2 + 2?\\n\""),"What's 2 + 2?")
    def test_invalid_input(self):
        for value in (None,12,'',"b'broken",b'\xff'):
            with self.assertRaises((ValueError,TypeError,SyntaxError,UnicodeDecodeError)):clean_math_field(value)


if __name__=='__main__':unittest.main()

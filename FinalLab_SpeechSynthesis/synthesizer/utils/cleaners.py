"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You"ll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""
import re
from unidecode import unidecode
from synthesizer.utils.numbers import normalize_numbers


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1]) for x in [
    ("mrs", "misess"),
    ("mr", "mister"),
    ("dr", "doctor"),
    ("st", "saint"),
    ("co", "company"),
    ("jr", "junior"),
    ("maj", "major"),
    ("gen", "general"),
    ("drs", "doctors"),
    ("rev", "reverend"),
    ("lt", "lieutenant"),
    ("hon", "honorable"),
    ("sgt", "sergeant"),
    ("capt", "captain"),
    ("esq", "esquire"),
    ("ltd", "limited"),
    ("col", "colonel"),
    ("ft", "fort"),
]]

#  扩展文本中的缩写词
def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

# 将文本中的数字转换为文字表述
def expand_numbers(text):
    return normalize_numbers(text)

# 将文本转换为小写
def lowercase(text):
    """lowercase input tokens."""
    return text.lower()

# 合并文本中的多个连续空白字符为一个空格
def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)

# 将文本转换为 ASCII 编码，主要用于去除特殊字符或重音符号
def convert_to_ascii(text):
    return unidecode(text)

# 基本的文本清洗流程，包括转换小写和合并空白
def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

# 用于非英语文本的清洗流程，包括转换为 ASCII、转换小写和合并空白
def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

# 英语文本的清洗流程，包括转换为 ASCII、扩展数字和缩写、转换小写和合并空白。
def english_cleaners(text):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text

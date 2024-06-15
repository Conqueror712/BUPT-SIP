from synthesizer.utils.symbols import symbols
from synthesizer.utils import cleaners
import re


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


def text_to_sequence(text, cleaner_names):
    """
    将文本字符串转换为对应符号的ID序列。

    参数:
        text (str): 需要转换的文本字符串。
        cleaner_names (list): 要应用的文本清洗函数名称列表。

    返回:
        list: 对应文本中符号的ID序列。

    描述:
        文本中可以包含用花括号括起来的ARPAbet序列，例如 "Turn left on {HH AW1 S S T AH0 N} Street."。
        该函数首先检查文本中的ARPAbet序列，并适当转换，然后应用指定的清洗函数，最后转换为符号ID序列。
    """
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    # Append EOS token
    sequence.append(_symbol_to_id["~"])
    return sequence


def sequence_to_text(sequence):
    """
    将符号ID序列转换回文本字符串。

    参数:
        sequence (list): 符号ID序列。

    返回:
        str: 从序列转换回的文本字符串。

    描述:
        该函数将ID序列转换回文本，如果符号表示ARPAbet，则将其放回花括号中。
    """
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")

# 根据指定的清洗函数列表清洗文本
def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text

# 将符号列表转换为ID序列
def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]

# 将ARPAbet文本转换为ID序列
def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])

# 检查是否应该保留符号
def _should_keep_symbol(s):
    return s in _symbol_to_id and s not in ("_", "~")

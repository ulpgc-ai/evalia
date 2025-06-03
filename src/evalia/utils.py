from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

def clean_illegal_chars(text):
    """
    Cleans illegal characters from a given text string.

    :param text: the string to clean
    :return: the cleaned string with illegal characters removed
    """
    if isinstance(text, str):
        return ILLEGAL_CHARACTERS_RE.sub("", text)
    return text
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

def clean_illegal_chars(text):
    if isinstance(text, str):
        return ILLEGAL_CHARACTERS_RE.sub("", text)
    return text
BLOCKERS = [
    '_START_ARTICLE_',
    '_START_SECTION_',
]

START = '_START_PARAGRAPH_'
NEWLINE = '_NEWLINE_'

def preprocess_text(s):
    lines = s.splitlines()
    keep = False
    for line in lines:
        if line in BLOCKERS:
            keep = False
        elif line == START:
            keep = True
        elif keep:
            yield line.replace(NEWLINE, "\n")

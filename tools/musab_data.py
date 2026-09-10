PAD, BOS, EOS, BYTE_OFFSET = 0, 1, 2, 3
MAX_SEQ_LEN, PROMPT_BUDGET, ANSWER_BUDGET = 256, 192, 64

def encode_prompt(q):
    ids = [BOS] + [b + BYTE_OFFSET for b in ('Question: ' + q + '\nAnswer: ').encode('utf-8')]
    if len(ids) > PROMPT_BUDGET:
        raise ValueError('Question exceeds prompt budget')
    return ids

def decode(ids):
    return bytes(i-BYTE_OFFSET for i in ids if BYTE_OFFSET <= i < 259).decode('utf-8', errors='replace')

def pack(q,a):
    raise RuntimeError('Training is not part of this Android project')

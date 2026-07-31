def number2string(i):
    """Convert a number to a string

    Input: long or integer
    Output: string (big-endian)
    """
    s=hex(i)[2:].rstrip('L')
    if len(s) % 2:
        s = '0' + s
    return bytes.fromhex(s).decode('latin1')

def number2string_N(i, N):
    """Convert a number to a string of fixed size

    i: long or integer
    N: length of string
    Output: string (big-endian)
    """
    s = '%0*x' % (N*2, i)
    return bytes.fromhex(s).decode('latin1')

def string2number(i):
    """ Convert a string to a number

    Input: string (big-endian)
    Output: long or integer
    """
    if isinstance(i, bytes):
        raw = i
    else:
        raw = i.encode('latin1')
    return int.from_bytes(raw, 'big')

def xorstring(a,b):
    """XOR two strings of same length

    For more complex cases, see CryptoPlus.Cipher.XOR"""
    assert len(a) == len(b)
    return number2string_N(string2number(a)^string2number(b), len(a))

class Counter(str):
    #found here: http://www.lag.net/pipermail/paramiko/2008-February.txt
    """Necessary for CTR chaining mode

    Initializing a counter object (ctr = Counter('xxx'), gives a value to the counter object.
    Everytime the object is called ( ctr() ) it returns the current value and increments it by 1.
    Input/output is a raw string.

    Counter value is big endian"""
    def __init__(self, initial_ctr):
        if not isinstance(initial_ctr, str):
            raise TypeError("nonce must be str")
        self.c = string2number(initial_ctr)
    def __call__(self):
        # This might be slow, but it works as a demonstration
        ctr = bytes.fromhex("%032x" % (self.c,)).decode('latin1')
        self.c += 1
        return ctr

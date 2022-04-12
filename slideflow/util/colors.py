'''Text coloring.'''


def dim(text):
    return '\033[2m' + str(text) + '\033[0m'


def yellow(text):
    return '\033[93m' + str(text) + '\033[0m'


def cyan(text):
    return '\033[96m' + str(text) + '\033[0m'


def blue(text):
    return '\033[94m' + str(text) + '\033[0m'


def green(text):
    return '\033[92m' + str(text) + '\033[0m'


def red(text):
    return '\033[91m' + str(text) + '\033[0m'


def bold(text):
    return '\033[1m' + str(text) + '\033[0m'


def underline(text):
    return '\033[4m' + str(text) + '\033[0m'


def purple(text):
    return '\033[38;5;5m' + str(text) + '\033[0m'

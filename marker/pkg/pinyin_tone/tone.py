#! /usr/bin/python
# -*- encoding: utf-8 -*-

vowels = {
    u'ɑ': ('a', ''),
    u'ɡ': ('g', ''),
    u'ā': ('a', '1'),
    u'á': ('a', '2'),
    u'ǎ': ('a', '3'),
    u'à': ('a', '4'),
    u'ē': ('e', '1'),
    u'é': ('e', '2'),
    u'ě': ('e', '3'),
    u'è': ('e', '4'),
    u'ō': ('o', '1'),
    u'ó': ('o', '2'),
    u'ǒ': ('o', '3'),
    u'ò': ('o', '4'),
    u'ī': ('i', '1'),
    u'í': ('i', '2'),
    u'ǐ': ('i', '3'),
    u'ì': ('i', '4'),
    u'ū': ('u', '1'),
    u'ú': ('u', '2'),
    u'ǔ': ('u', '3'),
    u'ù': ('u', '4'),
    u'ü': ('v', ''),
    u'ǖ': ('v', '1'),
    u'ǘ': ('v', '2'),
    u'ǚ': ('v', '3'),
    u'ǜ': ('v', '4'),
    u'ń': ('n', '2'),
    u'ň': ('n', '3'),
    u'': ('m', '2'),
    u'ḿ': ('m', '2'),
}


def tone_symbol_to_number(word):
    cword = ''
    tone = ''
    for i in word:
        try:
            cword += vowels[i][0]
            tone = vowels[i][1]
        except KeyError:
            if i >= u'a' and i <= u'z':
                cword += i
    return cword + tone

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_word", default="", help="converting word")
    flags = parser.parse_args()

    ret = tone_symbol_to_number(flags.input_word.decode('utf-8'))
    print(1, ret)

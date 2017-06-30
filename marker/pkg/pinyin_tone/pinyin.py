#! /usr/bin/python
# -*- encoding: utf-8 -*-
#
# PinyinMarker Lib
#
# Usuage:
#       marker = simple_mandarin_marker([YOUR_MARKING_FILES])
#       where marking.dat is marking file.
#       marker.mark(chinese_string)
#
# =============================================================================

import os

from . import actrie as actrie


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def is_light_tone(pinyin):
    tone_set = set("0123456789")
    for l in pinyin:
        if l in tone_set:
            return False
    return True


class PinyinUnit(object):
    def __init__(self, pinyin, freq):
        self._pinyin = pinyin
        self._freq = freq
        self._length = len(pinyin)

    @property
    def pinyin(self):
        return self._pinyin

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, value):
        self._freq = value

    @property
    def length(self):
        return self._length


class MarkUnit:
    def __init__(self, chinese, pinyin_units):
        self._chinese = chinese
        self._pinyin_units = dict([(' '.join(item.pinyin), item)
                                   for item in pinyin_units])

    def merge_pinyin(self, pinyin_units):
        for pinyin_unit in pinyin_units:
            key = ' '.join(pinyin_unit.pinyin)
            if key in self._pinyin_units:
                self._pinyin_units[key].freq += pinyin_unit.freq
            else:
                self._pinyin_units[key] = pinyin_unit

    def merge_mark_unit(self, markunit):
        if self.chinese != markunit.chinese:
            return
        self._pinyin_units = self.merge(markunit.pinyin_units)

    @property
    def chinese(self):
        return self._chinese

    @property
    def pinyin_units(self):
        return self._pinyin_units


class PinyinMarker:
    UNK_LABEL = '<unk>'

    def __init__(self, mark_dict=dict(), marking_files=[],
                 poly_dict=dict(), poly_files=[],
                 replace_unknown_to_tag=True):
        self._dict = {}
        self._dict_size = 0

        self._mark_dict = mark_dict
        self.merge_dat_file(marking_files)

        for chinese in self._mark_dict:
            self.add_seq_to_dict(chinese)

        self._poly_dict = poly_dict
        self.merge_polyphone_dict(poly_files)

        self._sing_dict = {}
        self.create_sing_dict()

        self._actrie_head = actrie.ACTrieNode()
        for key, value in self._mark_dict.items():
            max_freq_pinyin = max(value.pinyin_units,
                                  key=lambda k: value.pinyin_units[k].freq)
            node = self._actrie_head.walk_build_trie(key)
            node.set_value(value.pinyin_units[max_freq_pinyin])
        actrie.build_fail_links(self._actrie_head)

        self._replace_unknown_to_tag = replace_unknown_to_tag

    def create_sing_dict(self):
        for ch in self._dict:
            if ch not in self.poly_dict and ch in self.mark_dict:
                best_pinyin = None
                for pinyin_unit in self.mark_dict[ch].pinyin_units.values():
                    if best_pinyin is None or \
                                    pinyin_unit.freq > best_pinyin.freq:
                        best_pinyin = pinyin_unit
                self._sing_dict[ch] = best_pinyin

    def add_seq_to_dict(self, seq):
        for ch in seq:
            if ch not in self._dict:
                self._dict[ch] = self._dict_size
                self._dict_size += 1

    def add_pinyin_units(self, chinese, pinyin_units):
        if chinese in self._mark_dict:
            self._mark_dict[chinese].merge_pinyin(pinyin_units)
        else:
            self._mark_dict[chinese] = MarkUnit(chinese, pinyin_units)

    def add_mark_unit(self, markunit):
        if markunit.chinese in self._mark_dict:
            self._mark_dict[markunit.chinese].merge_mark_unit(markunit)
        else:
            self._mark_dict[markunit.chinese] = \
                MarkUnit(markunit.chinese, markunit.pinyin_units)

    def merge_dict(self, mark_dict):
        for mark_unit in mark_dict.values():
            self.add_mark_unit(mark_unit)

    def merge_dat_file(self, marking_files):
        for marking_file in marking_files:
            with open(marking_file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    units = line.strip().replace(u'・', '').split('|')
                    freq = 1 if len(units) <= 2 else int(units[2])
                    self.add_pinyin_units(units[0],
                                          [PinyinUnit(units[1].split(' '),
                                                      freq)])

    def merge_polyphone_dict(self, poly_files):
        for poly_file in poly_files:
            with open(poly_file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    units = line.strip().replace(u'・', '').split('|')
                    self.poly_dict[units[0]] = True

    def mark_only_unicode(self, tokens):
        """Given a list of character, mark them to a list of pinyin.

        Consider each character from left to right, its pinyin is determined by
        the longest word (choose the most frequent if there is more than one
        longest word) in the input sequence. All words are defined by
        dictionary.
        """
        sentence_length = len(tokens)

        match_word_length = [-1] * sentence_length
        match_word_freq = [0] * sentence_length
        match_word_pinyin = [t for t in tokens]
        is_poly = [True] * sentence_length

        for pos in range(sentence_length):
            if tokens[pos] in self.sing_dict:
                pinyin_unit = self.sing_dict[tokens[pos]]
                match_word_pinyin[pos] = pinyin_unit.pinyin[0]
                match_word_freq[pos] = pinyin_unit.freq
                match_word_length[pos] = 1
                is_poly[pos] = False

        match_pattern = []
        actrie_node = self._actrie_head
        for pos in range(sentence_length):
            actrie_node = actrie_node.move(tokens[pos])
            if actrie_node is None:
                actrie_node = self._actrie_head
            else:
                for pinyin in actrie_node.generate_all_suffix_nodes_values():
                    match_pattern.append((pinyin, pos))
                    break

        left_most_pos = sentence_length
        for pinyin, pos in reversed(match_pattern):
            if pos - pinyin.length + 1 >= left_most_pos:
                continue
            left_most_pos = pos - pinyin.length + 1
            for left_delta in range(pinyin.length):
                word_p = pos - left_delta
                if not is_poly[word_p]:
                    continue
                if (pinyin.length, pinyin.freq) > (
                        match_word_length[word_p],
                        match_word_freq[word_p]):
                    match_word_length[word_p], match_word_freq[word_p] = \
                        pinyin.length, pinyin.freq
                    match_word_pinyin[word_p] = pinyin.pinyin[
                        pinyin.length - left_delta - 1]

        if self._replace_unknown_to_tag:
            match_word_pinyin = \
                [PinyinMarker.UNK_LABEL
                 if match_word_length[pos] < 0
                 else match_word_pinyin[pos]
                 for pos in range(sentence_length)]
        return match_word_pinyin

    def mark(self, tokens):
        ret = []
        ll = 0
        for i in range(len(tokens)):
            if tokens[i] == u'<eos>' or tokens[i] == u'<sos>':
                ret.extend(self.mark_only_unicode(tokens[ll:i]))
                ret.append('<null>')
                ll = i + 1
        if ll != len(tokens):
            ret.extend(self.mark_only_unicode(tokens[ll:len(tokens)]))
        return ret

    @property
    def mark_dict(self):
        return self._mark_dict

    @property
    def poly_dict(self):
        return self._poly_dict

    @property
    def sing_dict(self):
        return self._sing_dict


class TonedMarker(PinyinMarker):
    def __init__(self, replace_unknown_to_tag=True):
        file_path = os.path.dirname(os.path.abspath(__file__))
        PinyinMarker.__init__(self, marking_files=[
            os.path.join(file_path, 'hot_fix_words.dat'),
            os.path.join(file_path, 'phrase.dat'),
            os.path.join(file_path, 'single_word.dat')],
                              poly_files=[
                                  os.path.join(file_path, 'polyphone.dat')],
                              replace_unknown_to_tag=replace_unknown_to_tag)


class XiaoyunMarker(TonedMarker):
    def __init__(self):
        from pkg.reg_dict.initial_final_without_light_tone import \
            initials, blocks, independents
        self._initials = initials
        self._blocks = set(blocks)
        self._independents = set(independents)
        TonedMarker.__init__(self)

    def mark(self, tokens):
        pinyin_tokens = TonedMarker.mark(self, tokens)
        iftokens = []
        for pinyin_token in pinyin_tokens:
            if pinyin_token in self._blocks:
                iftokens.append(pinyin_token)
            elif 'g_' + pinyin_token in self._independents:
                iftokens.append('g_' + pinyin_token)
            else:
                for initial in self._initials:
                    if pinyin_token.startswith(initial):
                        iftokens.append(initial)
                        iftokens.append(pinyin_token.lstrip(initial))
                        break
        return iftokens


class ZhaoxiongMarker(TonedMarker):
    def __init__(self):
        from pkg.reg_dict.initial_final_without_light_tone import \
            initials, blocks, independents
        self._initials = initials
        self._blocks = set(blocks)
        self._independents = set(independents)
        TonedMarker.__init__(self)

    def mark(self, tokens):
        pinyin_tokens = TonedMarker.mark(self, tokens)
        iftokens = []
        for pinyin_token in pinyin_tokens:
            if pinyin_token in self._blocks:
                iftokens.append(pinyin_token)
            elif 'g_' + pinyin_token in self._independents:
                iftokens.extend(['<b>', pinyin_token])
            else:
                for initial in self._initials:
                    if pinyin_token.startswith(initial):
                        iftokens.append(initial)
                        iftokens.append(pinyin_token.lstrip(initial))
                        break
        return iftokens


class EngTonedMarker(TonedMarker):
    def __init__(self):
        super().__init__(replace_unknown_to_tag=False)


if __name__ == "__main__":
    marker = TonedMarker(replace_unknown_to_tag=False)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_sentence", default="",
                        help="converting sentence")
    parser.add_argument("-f", "--file", default="",
                        help="Input file")
    flags = parser.parse_args()

    if flags.input_sentence:
        ret = marker.mark(flags.input_sentence.split())
        print(ret)

    if flags.file:
        with open(flags.file, 'r', encoding='utf-8') as f:
            for line in f:
                ret = marker.mark(line.strip().split())
                print(ret)


def pinyin_segment(pinyin):
    # -- extract initial and final for pinyin_tone --
    # example: 'zhong1 -> zh, ong, 1'
    non_initial_set = set(["a", "e", "o"])
    py_initial_end = 2 if pinyin[1] == "h" else 1
    py_initial_end = 0 if pinyin[0] in non_initial_set \
                       else py_initial_end

    py_initial = pinyin[0: py_initial_end]
    py_final = pinyin[py_initial_end: -1]
    py_tone = pinyin[-1]

    return py_initial, py_final, py_tone

if __name__ == "__main__":
    from pkg.reg_dict.pinyin_without_light_tone import py_list
    for py in py_list:
        py_initial, py_final, py_tone = pinyin_segment(py)
        print("{:<7} -> {:<2} {:<4} {}".format(py, py_initial, py_final, py_tone))


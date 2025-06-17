from phylotrackpy import systematics as syst
import sys

filename = sys.argv[1]

translation_dict_test = {
    "a": "nop-A",         # a
    "b": "nop-B",         # b
    "c": "nop-C",         # c
    "d": "if-n-equ",      # d
    "e": "if-less",       # e
    "f": "if-label",      # f
    "g": "mov-head",      # g
    "h": "jmp-head",      # h
    "i": "get-head",      # i
    "j": "set-flow",      # j
    "k": "shift-r",       # k
    "l": "shift-l",       # l
    "m": "inc",           # m
    "n": "dec",           # n
    "o": "push",          # o
    "p": "pop",           # p
    "q": "swap-stk",      # q
    "r": "swap",          # r 
    "s": "add",           # s
    "t": "sub",           # t
    "u": "nand",          # u
    "v": "h-copy",        # v
    "w": "h-alloc",       # w
    "x": "h-divide",      # x
    "y": "IO",            # y
    "z": "h-search",      # z
    "A": "rotate-l",
    "B": "read-faced-cell-org-id",
    "C": "get-cell-x",
    "D": "get-cell-y",
    "E": "get-cell-xy",
    "F": "get-north-offset"
}

translation_dict_control = {
    "a": "nop-A",         # a
    "b": "nop-B",         # b
    "c": "nop-C",         # c
    "d": "if-n-equ",      # d
    "e": "if-less",       # e
    "f": "if-label",      # f
    "g": "mov-head",      # g
    "h": "jmp-head",      # h
    "i": "get-head",      # i
    "j": "set-flow",      # j
    "k": "shift-r",       # k
    "l": "shift-l",       # l
    "m": "inc",           # m
    "n": "dec",           # n
    "o": "push",          # o
    "p": "pop",           # p
    "q": "swap-stk",      # q
    "r": "swap",          # r 
    "s": "add",           # s
    "t": "sub",           # t
    "u": "nand",          # u
    "v": "h-copy",        # v
    "w": "h-alloc",       # w
    "x": "h-divide",      # x
    "y": "IO",            # y
    "z": "h-search",      # z
    "A": "read-faced-cell-org-id",
    "B": "get-cell-x",
    "C": "get-cell-y",
    "D": "get-cell-xy",
    "E": "get-north-offset"
}

translation_dict_switch = {
    "a": "nop-A",         # a
    "b": "nop-B",         # b
    "c": "nop-C",         # c
    "d": "if-n-equ",      # d
    "e": "if-less",       # e
    "f": "if-label",      # f
    "g": "mov-head",      # g
    "h": "jmp-head",      # h
    "i": "get-head",      # i
    "j": "set-flow",      # j
    "k": "shift-r",       # k
    "l": "shift-l",       # l
    "m": "inc",           # m
    "n": "dec",           # n
    "o": "push",          # o
    "p": "pop",           # p
    "q": "swap-stk",      # q
    "r": "swap",          # r 
    "s": "add",           # s
    "t": "sub",           # t
    "u": "nand",          # u
    "v": "h-copy",        # v
    "w": "h-alloc",       # w
    "x": "h-divide",      # x
    "y": "IO",            # y
    "z": "h-search",      # z
    "A": "nop-A",
    "B": "read-faced-cell-org-id",
    "C": "get-cell-x",
    "D": "get-cell-y",
    "E": "get-cell-xy",
    "F": "get-north-offset"
}

if sys.argv[3] == "test":
    translation_dict = translation_dict_test
elif sys.argv[3] == "control":
    translation_dict = translation_dict_control
elif sys.argv[3] == "switch":
    translation_dict = translation_dict_switch


s = syst.Systematics()
s.load_from_file(filename, info_col="sequence")
t = s.get_mrca()
if t is None:
    for tax in s.get_ancestor_taxa():
        if tax.get_origination_time() >= 50000:
            if t is None or tax.get_total_offspring() > t.get_total_offspring():
                t = tax        
    
seq = t.get_info()

with open(sys.argv[2], "w") as outfile:
    outfile.write("#inst_set heads_default\n#hw_type 0\n\n")
    for el in seq:
        outfile.write(translation_dict[el] + "\n")

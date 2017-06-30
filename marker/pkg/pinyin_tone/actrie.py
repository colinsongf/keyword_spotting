class TrieNode(object):
    ''' Node in a trie tree
    '''

    def __init__(self, character="", children=None):
        self._c = character
        if children is not None:
            self._nexts = children
        else:
            self._nexts = {}
        self._value = None

    def move(self, character):
        if character in self.nexts:
            return self.nexts[character]
        else:
            return None

    def add_child(self, character):
        if character not in self.nexts:
            self.nexts[character] = TrieNode()

    def set_value(self, value):
        self._value = value

    def walk_build_trie(self, seq):
        """building the trie during walking."""
        root = self
        for ch in seq:
            root.add_child(ch)
            root = root.move(ch)
        return root

    @property
    def c(self):
        return self.c

    @property
    def nexts(self):
        return self._nexts

    @property
    def value(self):
        return self._value


class ACTrieNode(TrieNode):
    def __init__(self, character="", children=None):
        super(ACTrieNode, self).__init__(character, children)
        self._f = None

    def set_failure(self, node):
        self._f = node

    def add_child(self, character):
        if character not in self.nexts:
            self.nexts[character] = ACTrieNode()

    def move(self, character):
        """Take a walk in the automaton. Walking fail link until hit None."""
        if character in self.nexts:
            return self.nexts[character]
        else:
            suffix_match_node = self.f
            while suffix_match_node and \
                    character not in suffix_match_node.nexts:
                suffix_match_node = suffix_match_node.f
            if suffix_match_node:
                return suffix_match_node.nexts[character]
            else:
                return None

    def generate_all_suffix_nodes_values(self):
        """It generates data of all suffix node from bottom to top."""
        node = self
        while node:
            if node.value:
                yield node.value
            node = node.f

    @property
    def f(self):
        return self._f


def build_fail_links(root):
    """build fail link."""
    import queue
    q = queue.Queue()
    q.put(root)
    while not q.empty():
        cur = q.get()
        for ch, child in cur.nexts.items():
            if cur is root:
                child.set_failure(root)
            else:
                suffix_match_node = cur.f
                while suffix_match_node and ch not in suffix_match_node.nexts:
                    suffix_match_node = suffix_match_node.f
                if suffix_match_node:
                    child.set_failure(suffix_match_node.nexts[ch])
                else:
                    child.set_failure(root)
            q.put(child)

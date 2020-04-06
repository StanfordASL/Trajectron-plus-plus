class NodeType(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if type(other) == str and self.name == other:
            return True
        else:
            return isinstance(other, self.__class__) and self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def __add__(self, other):
        return self.name + other


class NodeTypeEnum(list):
    def __init__(self, node_type_list):
        self.node_type_list = node_type_list
        node_types = [NodeType(name, node_type_list.index(name) + 1) for name in node_type_list]
        super().__init__(node_types)

    def __getattr__(self, name):
        if not name.startswith('_') and name in object.__getattribute__(self, "node_type_list"):
            return self[object.__getattribute__(self, "node_type_list").index(name)]
        else:
            return object.__getattribute__(self, name)

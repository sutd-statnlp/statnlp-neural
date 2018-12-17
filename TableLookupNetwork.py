from Network import Network


class TableLookupNetwork(Network):

    def __init__(self, network_id, inst, nodes, children, param, compiler):
        super().__init__(network_id, inst, param)
        self.nodes = nodes
        self.children = children

    def get_node(self, k):
        return self.nodes[k]

    def get_children(self, k):
        return self.children[k]

    def get_all_nodes(self):
        return self.nodes

    def get_all_children(self):
        return self.children

    def count_nodes(self):
        return len(self.nodes)

    def is_removed(self, k):
        return False

    def is_root(self, k):
        return self.count_nodes() - 1 == k


from .networkframe import NetworkFrame


class MultilayerNetworkFrame:
    def __init__(self):
        pass

    def add_layer(self, nodes, edges, layer_key):
        pass

    def add_layer_link(self, layer_mapping):
        pass

    def _get_layer_nodes(self, layer_key):
        pass

    def _get_layer_edges(self, layer_key):
        pass

    def layer(self, layer_key) -> NetworkFrame:
        layer_nodes = self._get_layer_nodes(layer_key)
        layer_edges = self._get_layer_edges(layer_key)
        return NetworkFrame(layer_nodes, layer_edges)

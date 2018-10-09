# -*- coding: utf-8 -*-


class BaseNNModel(object):
    def __init__(self, config):
        self.config = config
        self.graph = None

    def var(self, key):
        return self.graph.get_operation_by_name(key).outputs[0]

    def set_graph(self, graph):
        self.graph = graph

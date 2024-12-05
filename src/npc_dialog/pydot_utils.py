import textwrap

import pydot

WRAPPER = lambda text: textwrap.fill(text, width=50)


def create_node(graph, id, label, shape='oval', wrapper=WRAPPER, **kwargs):
    node = pydot.Node(str(id), label=wrapper(label), shape=shape, **kwargs)
    graph.add_node(node)
    return node


def create_edge(graph, node_parent, node_child, **kwargs):
    link = pydot.Edge(src=node_parent, dst=node_child, **kwargs)
    graph.add_edge(link)
    return link

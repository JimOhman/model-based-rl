import matplotlib.pyplot as plt
import numpy as np
import pydot
import os

def write_mcts_as_png(search_paths, max_depth=float('inf'), use_value=True, path_to_file=''):
  graph = pydot.Dot(graph_type='graph')

  unique_nodes = []
  for path in search_paths:
    for node in path:
      if node not in unique_nodes:
        unique_nodes.append(node)

  seen = []
  chosen_nodes = []
  for path in search_paths:
    depth = 0
    for i, node in enumerate(path):
      if i == 0:
        root_id = str(node)
        root_label = str(node.visit_count) if not use_value else str(round(node.value(), 2))
        graph.add_node(pydot.Node(root_id, label=root_label, penwidth=4, color='green', fontsize=15, fontname="times bold"))
        chosen_nodes.append(node)

      if node not in seen:
        if any([child in unique_nodes for child in node.children]):
          parent_id = str(node)
          chosen_action = np.argmax([child.visit_count for child in node.children])
          for action, child in enumerate(node.children):
            child_id = str(child)
            node_value = "c:{}, v:{}".format(child.visit_count, round(child.value(), 2)) if child.visit_count else "NV"
            child_label = '{}'.format(node_value)

            node_style = ''
            border_width = 1
            fontname = 'times'
            arrow_width = 1
            if child.visit_count > 0:
              node_color = 'green'
              arrow = 'forward'
              edge_style = 'filled'
            else:
              node_color = 'red'
              arrow = ''
              edge_style = 'dotted'

            if node in chosen_nodes and action == chosen_action:
              fontname = 'times bold'
              node_style = 'bold'
              edge_style = 'bold'
              border_width = 4
              arrow_width = 2
              chosen_nodes.append(node.children[chosen_action])

            graph.add_node(pydot.Node(child_id, label=child_label, penwidth=border_width, style=node_style, color=node_color, fontsize=15, fontname=fontname))
            edge_label = 'p:{}'.format(round(child.prior, 2))
            if child.visit_count:
              edge_label += ', r:{}'.format(round(child.reward, 2))
            edge = pydot.Edge(parent_id, child_id, label=edge_label, dir=arrow, style=edge_style, penwidth=arrow_width, fontsize=10, fontname=fontname)
            graph.add_edge(edge)

        seen.append(node)

      depth += 1
      if depth > max_depth:
        break

  image = graph.write_png(path_to_file)

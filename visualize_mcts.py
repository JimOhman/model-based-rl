import numpy as np
import pydot


def write_mcts_as_png(search_paths, max_search_depth=float('inf'), path_to_file=''):
  graph = pydot.Dot(graph_type='graph')

  args = {'label': 'default',
          'penwidth': 1,
          'style': '',
          'color': 'green',
          'fontsize': 15,
          'fontname': 'times',
          'dir': '',
          'arrow': ''}

  unique_nodes = []
  for path in search_paths:
    for node in path:
      if node not in unique_nodes:
        unique_nodes.append(node)

  seen = []
  chosen_nodes = []
  for path in search_paths:
    search_depth = 0
    for i, node in enumerate(path):

      if i == 0:
        root_args = args.copy()
        c, v = node.visit_count, round(node.value(), 2)
        root_args['label'] = "c:{}, v:{}".format(c, v)
        root_args['penwidth'] = 4
        graph.add_node(pydot.Node(str(node), **root_args))
        chosen_nodes.append(node)

      if node not in seen:
        if any([child in unique_nodes for child in node.children]):
          parent_id = str(node)
          chosen_action = np.argmax([child.visit_count for child in node.children])
          for action, child in enumerate(node.children):
            child_id = str(child)

            node_args = args.copy()
            edge_args = args.copy()

            node_args['label'] = 'nv'
            edge_args['label'] = 'p:{}'.format(round(child.prior, 2))
            edge_args['fontsize'] = 10

            if child.visit_count > 0:
              c, v = node.visit_count, round(node.value(), 2)
              node_args['label'] = "c:{}, v:{}".format(c, v)
              edge_args['label'] += ', r:{}'.format(round(child.reward, 2))
              node_args['arrow'] = 'forward'
              edge_args['style'] = 'filled'
            else:
              node_args['color'] = 'red'
              edge_args['color'] = 'red'
              edge_args['style'] = 'dotted'

            if node in chosen_nodes and action == chosen_action:
              node_args['fontname'] = 'times bold'
              edge_args['fontname'] = 'times bold'
              node_args['style'] = 'bold'
              edge_args['style'] = 'bold'
              node_args['penwidth'] = 4
              edge_args['penwidth'] = 2
              chosen_nodes.append(node.children[chosen_action])

            graph.add_node(pydot.Node(child_id, **node_args))
            graph.add_edge(pydot.Edge(parent_id, child_id, **edge_args))

        seen.append(node)

      search_depth += 1
      if search_depth > max_search_depth:
        break

  image = graph.write_png(path_to_file)

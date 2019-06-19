#!/usr/bin/python3

# Generate dot file graph for the specified model.
# Uses the graphviz package
# Inspired by previous version (MKC + AP)
# Redeveloped by PTH

# @todo - better exceptions.
# @todo - remove UISS specialisation to merge back.
# @todo - probably better to build a graph structure in python, then traverse it to produce the graphviz? 
# @todo - tidy
# @todo - group for vertical alignment. Keratinocyte is an example. Should use the same group as the ancestor state, unless there are 2 childs of that state.
# @todo - order is right to left not left to right?
# @todo - globalConditions - keratinocyte as example
# @todo - localConditions - stable marriage as example
# @todo - death
# @todo - agent creation
# @todo - hostLauyer functions
# @todo - skip linear state relationships (when no conditionals)
# @todo - handle terminating paths. I.e. Keratinocyte migrate? / resolve, where currenlty a real-link is drawn even though it is not necesary. I.e. resolve state all immediately feeds through output_location into default
# @todo - fix wiggles
# @todo - stable marriage as good example of conditional.
# @todo - key
# @todo - positioning of init functions etc.
# @todo - iteration loop


import argparse
import sys
import os

import xml.etree.ElementTree as ET

import graphviz

DEBUG_COLORS = False


MESSAGE_COLOR = "green4"
MESSAGE_SHAPE = "box"

FUNCTION_COLOR = "black"
FUNCTION_SHAPE = "box"


STATE_COLOR = "#aaaaaa"
STATE_SHAPE = "circle"


HIDDEN_COLOR = "#ffffff"
HIDDEN_STYLE = "invis"
HIDDEN_SHAPE = "point"

if DEBUG_COLORS:
  HIDDEN_COLOR = "#dddddd"
  HIDDEN_STYLE = None


# Dictionary of the expected XML namespaces, making search easier
NAMESPACES = {
  "xmml": "http://www.dcs.shef.ac.uk/~paul/XMML",
  "gpu":  "http://www.dcs.shef.ac.uk/~paul/XMMLGPU"
}

def verbose_log(args, msg):
  if args.verbose:
    print("Log: " + msg)

def parse_arguments():
  # Get arguments using argparse
  parser = argparse.ArgumentParser(
    description="Generate a dot file from an xml model file."
  )
  parser.add_argument(
    "xmlmodelfile",
    type=str,
    help="Path to xml model file"
  )
  parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="Path to output location"
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Render the graphviz dotfile"
    )
  parser.add_argument(
    "--init-functions",
    action="store_true",
    help="render init functions"
    )
  parser.add_argument(
    "--step-functions",
    action="store_true",
    help="render step functions"
    )
  parser.add_argument(
    "--exit-functions",
    action="store_true",
    help="render exit functions"
    )
  parser.add_argument(
    "--all",
    action="store_true",
    help="include everything"
    )
  parser.add_argument(
    "-f",
    "--force",
    action="store_true",
    help="Overwrite output files."
    )
  parser.add_argument(
    "--verbose",
    action="store_true",
    help="Verbose output"
    )
  # This seems evil.
  parser.add_argument(
    "-c",
    "--cycles",
    action="store_true",
    help="Can have direct cycles"
    )

  args = parser.parse_args()
  return args

def validate_arguments(args):
  verbose_log(args, "Validating Arguments")

  # Validate the path exists.
  if os.path.exists(args.xmlmodelfile):
    if not os.path.isfile(args.xmlmodelfile):
      raise ValueError("xmlmodelfile {:} is not a file.".format(args.xmlmodelfile))
  else:
    raise ValueError("xmlmodelfile {:} does not exist.".format(args.xmlmodelfile))

  # Validate the (optional) output file. 
  if args.output is not None:
    if os.path.exists(args.output):
      if os.path.isfile(args.output) and not args.force:
        raise ValueError("Output path {:} exists. Use --force to overwrite.".format(args.output))
      elif os.path.isdir(args.output):
        raise ValueError("Output path {:} is a directory.".format(args.output))


def render_init_functions(args):
  return args.init_functions or args.all
def render_step_functions(args):
  return args.step_functions or args.all
def render_exit_functions(args):
  return args.exit_functions or args.all

def load_xmlmodelfile(args):
  verbose_log(args, "Loading XML Model File")

  # Attempt to parse the file. 
  tree = ET.parse(args.xmlmodelfile)

  return tree


def get_model_name(args, xmlroot):
  namexml = xmlroot.find('xmml:name', NAMESPACES)
  return namexml.text

def get_init_functions(args, xmlroot):
  data = []
  for x in xmlroot.findall('gpu:environment/gpu:initFunctions/gpu:initFunction/gpu:name', NAMESPACES):
    data.append(x.text)
  return data


def get_step_functions(args, xmlroot):
  data = []
  for x in xmlroot.findall('gpu:environment/gpu:stepFunctions/gpu:stepFunction/gpu:name', NAMESPACES):
    data.append(x.text)
  return data

def get_exit_functions(args, xmlroot):
  data = []
  for x in xmlroot.findall('gpu:environment/gpu:exitFunctions/gpu:exitFunction/gpu:name', NAMESPACES):
    data.append(x.text)
  return data

def get_host_layer_functions(args, xmlroot):
  data = []
  for x in xmlroot.findall('gpu:environment/gpu:hostLayerFunctions/gpu:hostLayerFunction/gpu:name', NAMESPACES):
    data.append(x.text)
  return data

def get_agent_names(args, xmlroot):
  data = []
  for x in xmlroot.findall('xmml:xagents/gpu:xagent/xmml:name', NAMESPACES):
    data.append(x.text)
  return data

  # data = []
  # xagents = xmlroot.find('xmml:xagents', NAMESPACES)
  # if xagents is not None:
  #   for xagent in xagents.findall('gpu:xagent', NAMESPACES):
  #     name = xagent.find('xmml:name', NAMESPACES)
  #     if name is not None:
  #       data.append(name.text)
  # return data

def get_agent_functions(args, xmlroot):
  data = {}
  for xagent in xmlroot.findall('xmml:xagents/gpu:xagent', NAMESPACES):
    xagent_name = xagent.find('xmml:name', NAMESPACES)
    for function in xagent.findall('xmml:functions/gpu:function', NAMESPACES):
      function_name = function.find('xmml:name', NAMESPACES)
      if function_name is not None and xagent_name is not None:
        currentState = function.find('xmml:currentState', NAMESPACES)
        nextState = function.find('xmml:nextState', NAMESPACES)
        data[function_name.text] = {
          "agent": xagent_name.text,
          "currentState": currentState.text,
          "nextState": nextState.text,
          "inputs": [],
          "outputs": [],
        }
        # inputs
        for msg in function.findall('xmml:inputs/gpu:input', NAMESPACES):
          msg_name = msg.find('xmml:messageName', NAMESPACES)
          data[function_name.text]["inputs"].append({
            "name": msg_name.text,
          })
        # outputs
        for msg in function.findall('xmml:outputs/gpu:output', NAMESPACES):
          msg_name = msg.find('xmml:messageName', NAMESPACES)
          msg_type = msg.find('gpu:type', NAMESPACES)
          data[function_name.text]["outputs"].append({
            "name": msg_name.text,
            "type": msg_type.text,
          })

  return data

def get_agent_states(args, xmlroot):
  data = {}
  for xagent in xmlroot.findall('xmml:xagents/gpu:xagent', NAMESPACES):
    xagent_name = xagent.find('xmml:name', NAMESPACES)
    for state in xagent.findall('xmml:states/gpu:state', NAMESPACES):
      state_name = state.find('xmml:name', NAMESPACES)
      if state_name is not None and xagent_name is not None:
        if xagent_name.text not in data:
          data[xagent_name.text] = []
        data[xagent_name.text].append(state_name.text)
  return data

def get_message_names(args, xmlroot):
  pass



def get_function_layers(args, xmlroot):
  data = []
  for layer in xmlroot.findall('xmml:layers/xmml:layer', NAMESPACES):
    row = []
    for layerFunctionName in layer.findall('gpu:layerFunction/xmml:name', NAMESPACES):
      row.append(layerFunctionName.text)
    data.append(row)
  return data




def generate_graphviz(args, xml):
  verbose_log(args, "Generating Graphviz Data")
  
  
  # Get the root element from the XML
  xmlroot = xml.getroot()

  model_name = get_model_name(args, xmlroot)
  # print("Model: " + model_name)


  # Get a bunch of data from the xml file. 
  init_functions = get_init_functions(args, xmlroot)
  step_functions = get_step_functions(args, xmlroot)
  exit_functions = get_exit_functions(args, xmlroot)
  host_layer_functions = get_host_layer_functions(args, xmlroot)
  agent_names = get_agent_names(args, xmlroot)
  agent_functions = get_agent_functions(args, xmlroot)
  function_layers = get_function_layers(args, xmlroot)
  agent_states = get_agent_states(args, xmlroot)
  
  # print(init_functions)
  # print(step_functions)
  # print(exit_functions)
  


  # Create the digraph
  dot = graphviz.Digraph(name=model_name, comment="FLAME GPU State Diagram")


  # Add some global settings.
  dot.body.append("\tnewrank=true;")
  dot.body.append("\tcompound=true;")
  dot.body.append("\tsplines=ortho;")
  dot.body.append("\trankdir=ortho;")
  dot.body.append("\tordering=out;")
  # dot.body.append("\tSTART [style=invisible];");
  # dot.body.append("\tMID [style=invisible];");
  # dot.body.append("\tEND [style=invisible];");




  # Populate the digraph.

  # If there are any init functions, add a subgraph.
  # if init_functions and len(init_functions) > 0:
  if render_init_functions(args):
    ifg = graphviz.Digraph(
      name="cluster_initFunctions", 
    )
    ifg.attr(label="initFunctions")
    ifg.attr(color="blue")
    ifg.attr(penwidth="3")

    # Add node per func
    for func in init_functions:
      ifg.node(func, shape=FUNCTION_SHAPE)
      # @todo edges.

    # Add edge between subsequent nodes, if needed.
    if len(init_functions) > 1:
      for a, b in zip(init_functions, init_functions[1:]):
        ifg.edge(a, b)

    # Add an invisible node.
    ifg.node("invisible_initFunctions", shape="point", style="invis")

    # Add to the main graph
    dot.subgraph(ifg)




  # If there are any step functions, add a subgraph.
  # if step_functions and len(step_functions) > 0:
  if render_step_functions(args):
    sfg = graphviz.Digraph(
      name="cluster_stepFunctions", 
    )
    sfg.attr(label="stepFunctions")
    sfg.attr(color="brown")
    sfg.attr(penwidth="3")

    for func in step_functions:
      sfg.node(func, shape=FUNCTION_SHAPE)
    
    # Add edge between subsequent nodes, if needed.
    if len(step_functions) > 1:
      for a, b in zip(step_functions, step_functions[1:]):
        sfg.edge(a, b)

    sfg.node("invisible_stepFunctions", shape="point", style="invis")

    # Add to the main graph
    dot.subgraph(sfg)



  # If there are any exit functions, add a subgraph.
  # if exit_functions and len(exit_functions) > 0:
  if render_exit_functions(args):
    efg = graphviz.Digraph(
      name="cluster_exitFunctions", 
    )
    efg.attr(label="exitFunctions")
    efg.attr(color="red")
    efg.attr(penwidth="3")

    for func in exit_functions:
      efg.node(func, label=func, shape=FUNCTION_SHAPE)
    
    # Add edge between subsequent nodes, if needed.
    if len(exit_functions) > 1:
      for a, b in zip(exit_functions, exit_functions[1:]):
        efg.edge(a, b)

    efg.node("invisible_exitFunctions", shape="point", style="invis")


    # Add to the main graph
    dot.subgraph(efg)



  # Create a subgraph for the agent functions.
  afg= graphviz.Digraph(
    name="cluster_agentFunctions"
  )
  afg.attr(label="Function Layers")
  afg.attr(color="cyan")
  afg.attr(penwidth="3")
  afg.attr(rankdir="RL")

  afg.node("invisible_agentFunctions", shape="point", style="invis")




  layer_count = len(function_layers)
  # print("{:} layers".format(layer_count))


  # Create a dict of one subgraph per agent
  # For each agent
  agent_subgraphs = {}
  agent_state_connections = {}

  message_info = {}

  for agent_name in agent_names:
    agent_subgraphs[agent_name] = graphviz.Digraph(
      name="cluster_agent_{:}".format(agent_name)
    )
    agent_subgraphs[agent_name].attr(label=agent_name)
    # Also create a data structure to show if a state/function is used or not. 
    agent_state_connections[agent_name] = {}


    # Add each state once per layer + 1
    for state in agent_states[agent_name]:
      agent_state_connections[agent_name][state] = [False]* layer_count
      for i in range(layer_count + 1):
        state_id = "{:}_{:}".format(state, i)
        agent_subgraphs[agent_name].node(
          state_id, 
          label=state, 
          color=STATE_COLOR, 
          fontcolor=STATE_COLOR,
          group=state,
        )


  # For each function in each layer

  for layerIndex, function_layer in enumerate(function_layers):
    for col, function_name in enumerate(function_layer):
      # get the function data
      if function_name in agent_functions:
        function_obj = agent_functions[function_name]
        # Get the agent data
        agent_name = function_obj["agent"]
        state_before = "{:}_{:}".format(function_obj["currentState"], layerIndex)
        state_after = "{:}_{:}".format(function_obj["nextState"], layerIndex + 1)
        # print("{:}->{:}, {:}:{:}, {:}=>{:}".format(agent_name, function_name, layerIndex, col,state_before, state_after))

        # Add a node for the function.
        agent_subgraphs[agent_name].node(
          function_name,
          shape=FUNCTION_SHAPE,
          rank=str(layerIndex)
        )


        # Mark off the connection if staying int eh same state
        if function_obj["currentState"] == function_obj["nextState"]:
          agent_state_connections[agent_name][function_obj["currentState"]][layerIndex] = True

        # Add a link between the state_before and the function node, 
        agent_subgraphs[agent_name].edge(state_before, function_name)
        # And a link from the function node to teh after state.
        agent_subgraphs[agent_name].edge(function_name, state_after)


        # Store message related information in a global list for later insertion.
        if function_obj["inputs"]:
          for msg_input in function_obj["inputs"]:
            msg_name = msg_input["name"]
            if msg_name not in message_info:
              message_info[msg_name] = {
                "output_by": [],
                "input_by": [],
              }
            message_info[msg_name]["input_by"].append({
              "agent": agent_name,
              "function": function_name,
              "layer": layerIndex,
            })

        if function_obj["outputs"]:
          for msg_output in function_obj["outputs"]:
            msg_name = msg_output["name"]
            if msg_name not in message_info:
              message_info[msg_name] = {
                "output_by": [],
                "input_by": [],
              }
            message_info[msg_name]["output_by"].append({
              "agent": agent_name,
              "function": function_name,
              "layer": layerIndex,
            })
      elif function_name in host_layer_functions:
        print("@todo - host layer functions.")




  # Add missing links
  for agent_name in agent_names:
    for state in agent_states[agent_name]:
      for startLayer, connected in enumerate(agent_state_connections[agent_name][state]):
        if not connected:
          state_before = "{:}_{:}".format(state, startLayer)
          state_after = "{:}_{:}".format(state, startLayer+1)
          
          # Add the edge
          agent_subgraphs[agent_name].edge(state_before, state_after)

          # Add an invisible node to get vertical alignment across agents
          invisible_node_key = "invisible_{:}_{:}".format(state, startLayer)
          agent_subgraphs[agent_name].node(invisible_node_key, shape=HIDDEN_SHAPE, label='', style=HIDDEN_STYLE)
          # And invisible edges.
          agent_subgraphs[agent_name].edge(state_before, invisible_node_key, color=HIDDEN_COLOR, style=HIDDEN_STYLE)
          agent_subgraphs[agent_name].edge(invisible_node_key, state_after, color=HIDDEN_COLOR, style=HIDDEN_STYLE)


          # Mark as done
          agent_state_connections[agent_name][state][startLayer] = True



  # Add each agent_subgraph to the agent functions subgraph.
  for agent_subgraph in agent_subgraphs.values():
    afg.subgraph(agent_subgraph)

   # try adding an invisble link between all 0 states, as a weird hack to get vertical alignment?

  # heirarchys are annyoing...
  # flat_states = [item for agent, sublist in agent_states.items() for item in sublist]
  sg = graphviz.Digraph("state_alignment")
  sg.attr(rank="same")
  ordered_state_invis_nodes = []
  for agent_name in agent_names:
    for state in agent_states[agent_name]:
      key = "{:}_-1".format(state)
      zero = "{:}_0".format(state)
      sg.node(key, shape="point",color=HIDDEN_COLOR, rank="same", style=HIDDEN_STYLE)
      sg.edge(key, zero, color=HIDDEN_COLOR, style=HIDDEN_STYLE)
      ordered_state_invis_nodes.append(key)

  for a, b in zip(ordered_state_invis_nodes, ordered_state_invis_nodes[1:]):
    # Add an invisble edge.
    sg.edge(a, b, color=HIDDEN_STYLE, style=HIDDEN_STYLE)

  afg.subgraph(sg)

  # Add messages nodes
  for message_name in message_info:
    message_node_key = "message_{:}".format(message_name)
    afg.node(
      message_node_key, 
      label=message_name,
      shape=MESSAGE_SHAPE,
      fontcolor=MESSAGE_COLOR,
      color=MESSAGE_COLOR,
      penwidth="3",
    )

    # Add message edges.
    # out
    for output_by in message_info[message_name]["output_by"]:
      function_key = "{:}".format(output_by["function"])

      afg.edge(
        function_key,
        message_node_key,
        color=MESSAGE_COLOR,
        penwidth="3",
      )

    # in
    for input_by in message_info[message_name]["input_by"]:
      function_key = "{:}".format(input_by["function"])

      afg.edge(
        message_node_key,
        function_key,
        color=MESSAGE_COLOR,
        penwidth="3",
      )


  dot.subgraph(afg)


  # Connect the clusters.
  if render_init_functions(args) and render_step_functions(args):
    dot.edge(
      "invisible_initFunctions",
      "invisible_stepFunctions", 
      ltail="cluster_initFunctions", 
      lhead="cluster_stepFunctions"
    )
  if render_init_functions(args) and not render_step_functions(args):
    dot.edge(
      "invisible_initFunctions",
      "invisible_agentFunctions", 
      ltail="cluster_initFunctions", 
      lhead="cluster_agentFunctions"
    )
  if render_step_functions(args):
    dot.edge(
      "invisible_stepFunctions",
      "invisible_agentFunctions", 
      ltail="cluster_stepFunctions", 
      lhead="cluster_agentFunctions"
    )
  if render_exit_functions(args):
    dot.edge(
      "invisible_agentFunctions",
      "invisible_exitFunctions", 
      ltail="cluster_agentFunctions", 
      lhead="cluster_exitFunctions"
    )

  # Finally fix some ranks.

  # dot.attr(rank="same")

  return dot


def output_graphviz(args, graphviz_data):
  verbose_log(args, "Outputting Graphviz Data")

  if graphviz_data is None:
    raise ValueError("Invalid Graphviz Data")

  if args.output:
    graphviz_data.render(args.output, view=args.render)  
  else:
    print("@todo - render without saving to disk?")

def main():

  # Process command line args
  args = parse_arguments()
  
  # Validate arguments
  try:
    validate_arguments(args)
  except ValueError as err:
      print(err)
      return

  # Generate the dotfile
  try:
    xml = load_xmlmodelfile(args)
    graphviz_data = generate_graphviz(args, xml)
    output_graphviz(args, graphviz_data)
  except ValueError as err:
      print(err)
      return


if __name__ == "__main__":
  main()

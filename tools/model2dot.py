#!/usr/bin/python3

# Generate dot file graph for the specified model.
# Uses the graphviz package
# Inspired by previous version (MKC + AP)
# Redeveloped by PTH

# @todo - better exceptions.
# @todo - remove UISS specialisation to merge back.

import argparse
import sys
import os

import xml.etree.ElementTree as ET

import graphviz

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
        if xagent_name.text not in data:
          data[xagent_name.text] = {}
        data[xagent_name.text][function_name.text] = {}
  return data

def get_message_names(args, xmlroot):
  pass

def get_agent_states(args, xmlroot):
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
  dot.body.append("\tSTART [style=invisible];");
  dot.body.append("\tMID [style=invisible];");
  dot.body.append("\tEND [style=invisible];");




  # Populate the digraph.

  # If there are any init functions, add a subgraph.
  # if init_functions and len(init_functions) > 0:
  if True:
    ifg = graphviz.Digraph(
      name="cluster_initFunctions", 
    )
    ifg.attr(label="initFunctions")
    ifg.attr(color="blue")
    ifg.attr(penwidth="3")

    # Add node per func
    for func in init_functions:
      ifg.node(func, shape="box")
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
  if True:
    sfg = graphviz.Digraph(
      name="cluster_stepFunctions", 
    )
    sfg.attr(label="stepFunctions")
    sfg.attr(color="brown")
    sfg.attr(penwidth="3")

    for func in step_functions:
      sfg.node(func, shape="box")
    
    # Add edge between subsequent nodes, if needed.
    if len(step_functions) > 1:
      for a, b in zip(step_functions, step_functions[1:]):
        sfg.edge(a, b)

    sfg.node("invisible_stepFunctions", shape="point", style="invis")

    # Add to the main graph
    dot.subgraph(sfg)



  # If there are any exit functions, add a subgraph.
  # if exit_functions and len(exit_functions) > 0:
  if True:
    efg = graphviz.Digraph(
      name="cluster_exitFunctions", 
    )
    efg.attr(label="exitFunctions")
    efg.attr(color="red")
    efg.attr(penwidth="3")

    for func in exit_functions:
      efg.node(func, label=func, shape="box")
    
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



  # For each function layer
  for layerIndex, function_layer in enumerate(function_layers):

    lfg = graphviz.Digraph(
      name="cluster_layer_{:}".format(layerIndex)
    )
    lfg.attr(label="Layer {:}".format(layerIndex))
    lfg.attr(color="pink") # @todo rotate colours
    lfg.attr(penwidth="3")
    lfg.attr(rank="same")

    lfg.node("invisble_layer_{:}".format(layerIndex), shape="point", style="invis")

    for col, function in enumerate(function_layer):
      print(function)
      lfg.node(
        function,
        shape="box",
        group=str(col),
      )
    afg.subgraph(lfg)

  for i in range(1, len(function_layers)):
    afg.edge(
      "invisble_layer_{:}".format(i-1),
      "invisble_layer_{:}".format(i),
      ltail="cluster_layer_{:}".format(i-1),
      lhead="cluster_layer_{:}".format(i),
    )
    print(i-1, i)



  dot.subgraph(afg)


  # Connect the clusters.
  dot.edge(
    "invisible_initFunctions",
    "invisible_stepFunctions", 
    ltail="cluster_initFunctions", 
    lhead="cluster_stepFunctions"
  )
  dot.edge(
    "invisible_stepFunctions",
    "invisible_agentFunctions", 
    ltail="cluster_stepFunctions", 
    lhead="cluster_agentFunctions"
  )
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

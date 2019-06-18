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
  # file.write("  newrank=true;\ncompound=true; \n splines=ortho;\n")
  # file.write("  START [style=invisible];\n");
  # file.write("  MID [style=invisible];\n");
  # file.write("  END [style=invisible];\n");


  # Populate the digraph.
  # @todo



  # For each function layer
  for layerIndex, function_layer in enumerate(function_layers):
    print("{:}:".format(layerIndex))
    for function in function_layer:
      print("  " + function)

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

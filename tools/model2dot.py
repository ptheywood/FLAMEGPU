#!/usr/bin/python3

# Generate dot file graph for the specified model.
# Uses the graphviz package
# Inspired by previous version (MKC + AP)
# Redeveloped by PTH

# @todo - better exceptions.
# @todo - remove UISS specialisation to merge back.
# @todo - probably better to build a graph structure in python, then traverse it to produce the graphviz? 
# @todo - tidy
# @todo - Improve vertical alignment, though its reasonable now.
# @todo - globalConditions - add a decision box indicating if the func should be used or skipped
# @todo - localConditions - add a decision box indicating the condition.
# @todo - terminating paths - highlight impossible sections of the graph
# @todo - fix wiggles
# @todo - key


import argparse
import sys
import os

import xml.etree.ElementTree as ET

import graphviz
from collections import OrderedDict
import tempfile

# Config options
DEBUG_COLORS = False
DEBUG_LABLES = True
STATE_FOLDING = True
USE_ORTHO_SPLINES = True
USE_PORTS = False
ITERATION_LOOP_LINK = False # Doesn't get along with ortho really.

# Constants.

# Choose from "dot", "neato", "fdp". "dot" is the only sane option.
GRAPHVIZ_ENGINE = "dot"

# Output formats. Others are supported by graphviz but are less suitable
AVAILABLE_OUTPUT_FORMATS = ["pdf", "svg", "png"]
DEFAULT_OUTPUT_FORMAT = AVAILABLE_OUTPUT_FORMATS[0]



GV_STYLE_SOLID="solid"
GV_STYLE_DASHED="dashed"
GV_STYLE_DOTTED="dotted"
GV_STYLE_BOLD="bold"
GV_STYLE_ROUNDED="bold"
GV_STYLE_FILLED="filled"
GV_STYLE_STRIPED="striped"
GV_STYLE_INVIS="invis"

VERT_EDGE_WEIGHT = "100"

GV_PORT_N = "n"
GV_PORT_E = "e"
GV_PORT_S = "s"
GV_PORT_W = "w"


RNG_LABEL = "RNG = true"
REALLOCATE_LABEL = "reallocate = true"


MESSAGE_COLOR = "green4"
MESSAGE_SHAPE = "parallelogram"

FUNCTION_COLOR = "black"
FUNCTION_SHAPE = "box"
FUNCTION_LINK_COLOR = "#000000"


EDGE_COLOR = "#000000"
EDGE_FONTCOLOR = "#000000"
CONDITION_COLOR = "#0000dd"
CONDITION_FONTCOLOR = "#0000dd"

GLOBAL_CONDITION_COLOR = "#0000ff"
GLOBAL_CONDITION_SHAPE = "box"

STATE_COLOR = "#aaaaaa"
STATE_SHAPE = "circle"
STATE_STATE_LINK_COLOR = "#000000"


START_KEY = "START"
START_LABEL = "START"
START_SHAPE = "oval"
START_COLOR = "darkgreen"

STOP_KEY = "STOP"
STOP_LABEL = "STOP"
STOP_SHAPE = "octagon"
STOP_COLOR = "red"

ITERATION_START_KEY = "iteration_graph_start"
ITERATION_START_LABEL = "Iteration\nStart"
ITERATION_START_COLOR = "#00cc00"
ITERATION_START_SHAPE = "octagon"
ITERATION_START_STYLE = "solid"
ITERATION_END_KEY = "iteration_graph_end"
ITERATION_END_LABEL = "Iteration\nEnd"
ITERATION_END_COLOR = "#cc0000"
ITERATION_END_SHAPE = "octagon"
ITERATION_END_STYLE = "solid"

HIDDEN_COLOR = "#ffffff"
HIDDEN_STYLE = GV_STYLE_INVIS
HIDDEN_SHAPE = "point"
HIDDEN_LABEL = ""
HIDDEN_LABEL_START = "s"
HIDDEN_LABEL_END = "e"
HIDDEN_COLOR_START = HIDDEN_COLOR
HIDDEN_COLOR_END = HIDDEN_COLOR

XAGENT_OUTPUT_STYLE="dashed"
XAGENT_OUTPUT_COLOR="#ff00ff"
XAGENT_OUTPUT_FONTCOLOR="#ff00ff"
XAGENT_OUTPUT_LABEL="xagentOutput: {:}::{:}"


HOSTLAYERFUNCTION_LINK_COLOR = GV_STYLE_INVIS
HOSTLAYERFUNCTION_STATE_STYLE = GV_STYLE_INVIS
HOSTLAYERFUNCTION_STATE_COLOR = GV_STYLE_INVIS
HOSTLAYERFUNCTION_STATE_SHAPE = "circle"

CLUSTER_INIT_FUNCTIONS_KEY = "cluster_initFunctions"
CLUSTER_INIT_FUNCTIONS_LABEL = "FLAME GPU Init Functions"
CLUSTER_INIT_FUNCTIONS_COLOR = "#ff0000"
CLUSTER_INIT_FUNCTIONS_STYLE = GV_STYLE_SOLID 

CLUSTER_STEP_FUNCTIONS_KEY = "cluster_stepFunctions"
CLUSTER_STEP_FUNCTIONS_LABEL = "FLAME GPU STEP Functions"
CLUSTER_STEP_FUNCTIONS_COLOR = "#00ff00"
CLUSTER_STEP_FUNCTIONS_STYLE = GV_STYLE_SOLID 

CLUSTER_EXIT_FUNCTIONS_KEY = "cluster_exitFunctions"
CLUSTER_EXIT_FUNCTIONS_LABEL = "FLAME GPU Exit Functions"
CLUSTER_EXIT_FUNCTIONS_COLOR = "#0000ff"
CLUSTER_EXIT_FUNCTIONS_STYLE = GV_STYLE_SOLID 

CLUSTER_HOST_LAYER_FUNCTIONS_KEY = "cluster_hostLayerFunctions"
CLUSTER_HOST_LAYER_FUNCTIONS_LABEL = "hostLayerFunctions"
CLUSTER_HOST_LAYER_FUNCTIONS_COLOR = "#ffff00"
CLUSTER_HOST_LAYER_FUNCTIONS_STYLE = GV_STYLE_SOLID 


CLUSTER_ITERATION_GRAPH_KEY = "cluster_iteration_graph"
CLUSTER_ITERATION_GRAPH_LABEL = "Iteration"
CLUSTER_ITERATION_GRAPH_COLOR = "#ff00ff"
CLUSTER_ITERATION_GRAPH_STYLE = GV_STYLE_SOLID 

CLUSTER_FUNCTION_LAYERS_GRAPH_KEY = "cluster_layers_graph"
CLUSTER_FUNCTION_LAYERS_GRAPH_LABEL = "Function Layers"
CLUSTER_FUNCTION_LAYERS_GRAPH_COLOR = "#00ffff"
CLUSTER_FUNCTION_LAYERS_GRAPH_STYLE = GV_STYLE_SOLID 

CLUSTER_AGENT_FUNCTIONS_KEY_PATTERN = "cluster_agent_{:}"
CLUSTER_AGENT_FUNCTIONS_LABEL_PATTERN = "{:}"
CLUSTER_AGENT_FUNCTIONS_COLORS = ["#eeeeee", "#dddddd", "#cccccc", "#bbbbbb", "#aaaaaa"]
CLUSTER_AGENT_FUNCTIONS_STYLES = [GV_STYLE_SOLID]

CLUSTER_PENWIDTH="3"

# Debug mode constant changes.
if DEBUG_COLORS:
  HIDDEN_COLOR = "#dddddd"
  HIDDEN_STYLE = GV_STYLE_SOLID
  HIDDEN_SHAPE = "star"
  HIDDEN_COLOR_START = "#00ff00"
  HIDDEN_COLOR_END = "#0000ff"

  FUNCTION_LINK_COLOR = "#ff0000"
  STATE_STATE_LINK_COLOR = "#00ff00"


  HOSTLAYERFUNCTION_LINK_COLOR = "#ff00ff"
  HOSTLAYERFUNCTION_STATE_STYLE = GV_STYLE_SOLID
  HOSTLAYERFUNCTION_STATE_COLOR = "#ff00ff"
  HOSTLAYERFUNCTION_STATE_SHAPE = "circle"

# Ortho + ports do not get along. https://gitlab.com/graphviz/graphviz/issues/1415
if not USE_PORTS or USE_ORTHO_SPLINES:
  GV_PORT_N = None
  GV_PORT_E = None
  GV_PORT_S = None
  GV_PORT_W = None




# Dictionary of the expected XML namespaces, making search easier
NAMESPACES = {
  "xmml": "http://www.dcs.shef.ac.uk/~paul/XMML",
  "gpu":  "http://www.dcs.shef.ac.uk/~paul/XMMLGPU"
}

def ith(l, i):
  return l[i % len(l)]

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
    "--format",
    choices=AVAILABLE_OUTPUT_FORMATS,
    default=DEFAULT_OUTPUT_FORMAT,
    help="Output format for rendered file (default: %(default)s)",
    )
  parser.add_argument(
    "--show",
    action="store_true",
    help="Render the graphviz dotfile and open the output file. Implies --render."
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

def nonrecursive_condition_half_to_string(xml):
  if xml is not None:
    value = xml.find("xmml:value", NAMESPACES)
    agent_variable = xml.find("xmml:agentVariable", NAMESPACES)

    if value is not None:
      return "{:}".format(value.text)
    elif agent_variable is not None:
      return "agent->{:}".format(agent_variable.text)
    else:
      return ""
  else:
    return ""

# @todo - this could be much, much better somehow
def recurse_condition_lhs_op_rhs(xml):
  if xml is not None:
    # Find the lhs, op and rhs elements
    lhs = xml.find("xmml:lhs", NAMESPACES)
    op = xml.find("xmml:operator", NAMESPACES)
    rhs = xml.find("xmml:rhs", NAMESPACES)


    lhs_string = ""
    rhs_string = ""

    # If all 3 are present
    if lhs is not None and op is not None and rhs is not None:
      # If the lhs includes a condition, recurse. 
      # Encapsulate this bit as a whole?
      lhs_condition = lhs.find("xmml:condition", NAMESPACES)
      if lhs_condition is not None:
        # Recurse to check the lhs. 
        inner_lhs = recurse_condition_lhs_op_rhs(lhs_condition)
        lhs_string = "({:})".format(inner_lhs)
      else:
        # Not recursive
        lhs_string = nonrecursive_condition_half_to_string(lhs)

      rhs_condition = rhs.find("xmml:condition", NAMESPACES)
      if rhs_condition is not None:
        # Recurse to check the rhs
        inner_rhs = recurse_condition_lhs_op_rhs(rhs_condition)
        rhs_string = "({:})".format(inner_rhs)
      else:
        # Not recursive rhs
        rhs_string = nonrecursive_condition_half_to_string(rhs)

      condition = "{:} {:} {:}".format(lhs_string, op.text, rhs_string)
      return condition
  # If the xml is invalid, return none.
  else:
    return None

def parse_function_condition(xml):
  if xml is not None:
    expression = recurse_condition_lhs_op_rhs(xml)
    string = "{:}".format(expression)
    return string
  else:
    return None

def parse_function_global_condition(xml):
  if xml is not None:

    expression = recurse_condition_lhs_op_rhs(xml)
    maxIters = xml.find("gpu:maxItterations", NAMESPACES)
    mustEvaluateTo = xml.find("gpu:mustEvaluateTo", NAMESPACES)

    string = "({:}) == {:} (upto {:} times)".format(expression,  mustEvaluateTo.text, maxIters.text)
    return string
  else:
    return None


def get_rng_flag(xml):
  flag = False
  if xml is not None:
    element = xml.find("gpu:RNG", NAMESPACES)
    flag = element is not None and element.text == "true"
  return flag

def get_reallocate_flag(xml):
  flag = False
  if xml is not None:
    element = xml.find("gpu:reallocate", NAMESPACES)
    flag = element is not None and element.text == "true"
  return flag

def get_xagent_outputs(xml):
  data = []
  if xml is not None:
    elements = xml.findall("xmml:xagentOutputs/gpu:xagentOutput", NAMESPACES)
    for element in elements:
      agent_name = element.find("xmml:xagentName", NAMESPACES)
      agent_state = element.find("xmml:state", NAMESPACES)
      if agent_name is not None and agent_state is not None:
        # This might not be valid, check at usage.
        data.append({
          "agent_name": agent_name.text, 
          "state": agent_state.text,
        })
  return data



def get_agent_functions(args, xmlroot):
  data = {}
  for xagent in xmlroot.findall('xmml:xagents/gpu:xagent', NAMESPACES):
    xagent_name = xagent.find('xmml:name', NAMESPACES)
    for function in xagent.findall('xmml:functions/gpu:function', NAMESPACES):
      function_name = function.find('xmml:name', NAMESPACES)
      if function_name is not None and xagent_name is not None:
        currentState = function.find('xmml:currentState', NAMESPACES)
        nextState = function.find('xmml:nextState', NAMESPACES)
        condition = parse_function_condition(function.find('xmml:condition', NAMESPACES))
        globalCondition = parse_function_global_condition(function.find('gpu:globalCondition', NAMESPACES))


        rng = get_rng_flag(function)
        xagentOutputs = get_xagent_outputs(function)
        reallocate = get_reallocate_flag(function)

        data[function_name.text] = {
          "agent": xagent_name.text,
          "currentState": currentState.text,
          "nextState": nextState.text,
          "inputs": [],
          "outputs": [],
          "condition": condition,
          "globalCondition": globalCondition,
          "rng": rng,
          "xagentOutputs": xagentOutputs,
          "reallocate": reallocate,
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
  
  # Create the digraph
  dot = graphviz.Digraph(
    name=model_name, 
    comment="FLAME GPU State Diagram for model_name",
    engine=GRAPHVIZ_ENGINE,
    format=args.format,
  )


  # Add some global settings.
  dot.graph_attr["newrank"]="true"
  dot.graph_attr["compound"]="true"
  if USE_ORTHO_SPLINES:
    dot.graph_attr["splines"]="ortho"
    # dot.graph_attr["rankdir"]="ortho"
  else:
    # dot.graph_attr["splines"]="none"
    # dot.graph_attr["splines"]="line"
    # dot.graph_attr["splines"]="polyline"
    # dot.graph_attr["splines"]="curved"
    dot.graph_attr["splines"]="spline"


  dot.graph_attr["ordering"]="out"

  # Populate the digraph.

  # Add a start node and an end node.

  dot.node(
    START_KEY, 
    label=START_LABEL,
    shape=START_SHAPE,
    color=START_COLOR,
  )
  dot.node(
    STOP_KEY, 
    label=STOP_LABEL,
    shape=STOP_SHAPE,
    color=STOP_COLOR,
  )

  # If there are any init functions, add a subgraph.
  # if init_functions and len(init_functions) > 0:
  if render_init_functions(args):
    if len(init_functions) > 0:
      ifg = graphviz.Digraph(
        name=CLUSTER_INIT_FUNCTIONS_KEY,
      )
      ifg.attr(label=CLUSTER_INIT_FUNCTIONS_LABEL)
      ifg.attr(color=CLUSTER_INIT_FUNCTIONS_COLOR)
      ifg.attr(style=CLUSTER_INIT_FUNCTIONS_STYLE)
      ifg.attr(penwidth=CLUSTER_PENWIDTH)

      # Add node per func
      for func in init_functions:
        ifg.node(func, shape=FUNCTION_SHAPE)
        # @todo edges.

      # Add edge between subsequent nodes, if needed.
      for a, b in zip(init_functions, init_functions[1:]):
        ifg.edge(
          a, 
          b,
          tailport=GV_PORT_S,
          headport=GV_PORT_N,
        )

      # Add an invisible nodes
      ifg.node(
        "invisible_initFunctions_start",
        label=HIDDEN_LABEL_START,
        color=HIDDEN_COLOR_START,
        shape=HIDDEN_SHAPE,
        style=HIDDEN_STYLE
      )
      ifg.node(
        "invisible_initFunctions_end",
        label=HIDDEN_LABEL_END,
        color=HIDDEN_COLOR_END,
        shape=HIDDEN_SHAPE,
        style=HIDDEN_STYLE
      )
      ifg.edge(
        "invisible_initFunctions_start",
        init_functions[0],
        color=HIDDEN_COLOR_START,
        style=HIDDEN_STYLE
      )
      ifg.edge(
        init_functions[-1],
        "invisible_initFunctions_end",
        color=HIDDEN_COLOR_END,
        style=HIDDEN_STYLE
      )

      # Add to the main graph
      dot.subgraph(ifg)


  # If there are any exit functions, add a subgraph.
  # if exit_functions and len(exit_functions) > 0:
  if render_exit_functions(args):
    if len(exit_functions) > 0:
      efg = graphviz.Digraph(
        name=CLUSTER_EXIT_FUNCTIONS_KEY, 
      )
      efg.attr(label=CLUSTER_EXIT_FUNCTIONS_LABEL)
      efg.attr(color=CLUSTER_EXIT_FUNCTIONS_COLOR)
      efg.attr(style=CLUSTER_EXIT_FUNCTIONS_STYLE)
      efg.attr(penwidth=CLUSTER_PENWIDTH)

      for func in exit_functions:
        efg.node(func, label=func, shape=FUNCTION_SHAPE)
      
      # Add edge between subsequent nodes, if needed.
      for a, b in zip(exit_functions, exit_functions[1:]):
        efg.edge(
          a,
          b,
          tailport=GV_PORT_S,
          headport=GV_PORT_N,
        )

      efg.node(
        "invisible_exitFunctions_start",
        label=HIDDEN_LABEL_START,
        color=HIDDEN_COLOR_START,
        shape=HIDDEN_SHAPE,
        style=HIDDEN_STYLE
      )
      efg.node(
        "invisible_exitFunctions_end",
        label=HIDDEN_LABEL_END,
        color=HIDDEN_COLOR_END,
        shape=HIDDEN_SHAPE,
        style=HIDDEN_STYLE
      )
      efg.edge(
        "invisible_exitFunctions_start",
        exit_functions[0],
        color=HIDDEN_COLOR_START,
        style=HIDDEN_STYLE
      )
      efg.edge(
        exit_functions[-1],
        "invisible_exitFunctions_end",
        color=HIDDEN_COLOR_END,
        style=HIDDEN_STYLE
      )

      # Add to the main graph
      dot.subgraph(efg)



  # Create a subgraph for the agent functions.
  iteration_graph = graphviz.Digraph(
    name=CLUSTER_ITERATION_GRAPH_KEY
  )
  iteration_graph.attr(label=CLUSTER_ITERATION_GRAPH_LABEL)
  iteration_graph.attr(color=CLUSTER_ITERATION_GRAPH_COLOR)
  iteration_graph.attr(style=CLUSTER_ITERATION_GRAPH_STYLE)
  iteration_graph.attr(penwidth=CLUSTER_PENWIDTH)
  # iteration_graph.attr(rankdir="RL")

  iteration_graph.node(
    ITERATION_START_KEY,
    label=ITERATION_START_LABEL,
    color=ITERATION_START_COLOR,
    shape=ITERATION_START_SHAPE,
    style=ITERATION_START_STYLE
  )
  iteration_graph.node(
    ITERATION_END_KEY,
    label=ITERATION_END_LABEL,
    color=ITERATION_END_COLOR,
    shape=ITERATION_END_SHAPE,
    style=ITERATION_END_STYLE
  )



  layers_graph = graphviz.Digraph(
    name=CLUSTER_FUNCTION_LAYERS_GRAPH_KEY
  )
  layers_graph.attr(label=CLUSTER_FUNCTION_LAYERS_GRAPH_LABEL)
  layers_graph.attr(color=CLUSTER_FUNCTION_LAYERS_GRAPH_COLOR)
  layers_graph.attr(style=CLUSTER_FUNCTION_LAYERS_GRAPH_STYLE)
  layers_graph.attr(penwidth=CLUSTER_PENWIDTH)
  # layers_graph.attr(rankdir="RL")

  layers_graph.node(
    "invisible_layers_graph_start",
    label=HIDDEN_LABEL_START,
    color=HIDDEN_COLOR_START,
    shape=HIDDEN_SHAPE,
    style=HIDDEN_STYLE
  )
  layers_graph.node(
    "invisible_layers_graph_end",
    label=HIDDEN_LABEL_END,
    color=HIDDEN_COLOR_END,
    shape=HIDDEN_SHAPE,
    style=HIDDEN_STYLE
  )


  # If there are any step functions, add a subgraph.
  # if step_functions and len(step_functions) > 0:
  if render_step_functions(args):
    if len(step_functions) > 0:
      sfg = graphviz.Digraph(
        name=CLUSTER_STEP_FUNCTIONS_KEY, 
      )
      sfg.attr(label=CLUSTER_STEP_FUNCTIONS_LABEL)
      sfg.attr(color=CLUSTER_STEP_FUNCTIONS_COLOR)
      sfg.attr(style=CLUSTER_STEP_FUNCTIONS_STYLE)
      sfg.attr(penwidth=CLUSTER_PENWIDTH)

      for func in step_functions:
        sfg.node(func, shape=FUNCTION_SHAPE)
      
      # Add edge between subsequent nodes, if needed.
      for a, b in zip(step_functions, step_functions[1:]):
        sfg.edge(
          a,
          b,
          tailport=GV_PORT_S,
          headport=GV_PORT_N,
        )

      sfg.node(
        "invisible_stepFunctions_start",
        label=HIDDEN_LABEL_START,
        color=HIDDEN_COLOR_START,
        shape=HIDDEN_SHAPE,
        style=HIDDEN_STYLE
      )
      sfg.node(
        "invisible_stepFunctions_end",
        label=HIDDEN_LABEL_END,
        color=HIDDEN_COLOR_END,
        shape=HIDDEN_SHAPE,
        style=HIDDEN_STYLE
      )
      sfg.edge(
        "invisible_stepFunctions_start",
        step_functions[0],
        color=HIDDEN_COLOR_START,
        style=HIDDEN_STYLE
      )
      sfg.edge(
        step_functions[-1],
        "invisible_stepFunctions_end",
        color=HIDDEN_COLOR_END,
        style=HIDDEN_STYLE
      )

      # Also add an edge from the start of the iteration to the start of the step.
      iteration_graph.edge(
        ITERATION_START_KEY,
        "invisible_stepFunctions_start",
        # color=ITERATION_START_COLOR,
        # style=ITERATION_START_STYLE,
        lhead="cluster_stepFunctions",
      )
      # and end of the step to the start of the layers.
      iteration_graph.edge(
        "invisible_stepFunctions_end",
        "invisible_layers_graph_start",
        # ITERATION_END_KEY,
        # color=HIDDEN_COLOR_END,
        # style=HIDDEN_STYLE,
        ltail="cluster_stepFunctions",
        lhead="cluster_layers_graph",
      )

      # Add to the main graph
      iteration_graph.subgraph(sfg)

  # Add edges linking the iteration graph and layer graph.
  iteration_graph.edge(
    ITERATION_START_KEY,
    "invisible_layers_graph_start",
    color=HIDDEN_COLOR_START,
    style=HIDDEN_STYLE,
  )
  iteration_graph.edge(
    "invisible_layers_graph_end",
    ITERATION_END_KEY,
    # color=HIDDEN_COLOR_END,
    # style=HIDDEN_STYLE,
    ltail="cluster_layers_graph"
  )

  # Add iteration loop.
  if ITERATION_LOOP_LINK:
    iteration_graph.edge(
      ITERATION_START_KEY,
      ITERATION_END_KEY,
      # color=HIDDEN_COLOR_END,
      # style=HIDDEN_STYLE,
      lhead="cluster_iteration_graph",
      dir="back"
    )



  layer_count = len(function_layers)

  # Prep a dict to group node by rank.
  rank_count = (layer_count * 2) + 2
  rank_list = OrderedDict() 
  for i in range(layer_count):
    rank_list["s_{:}".format(i)] = []
    rank_list["f_{:}".format(i)] = []
  rank_list["s_{:}".format(i+1)] = []


  # Create a dict of one subgraph per agent
  # For each agent
  agent_subgraphs = {}
  agent_state_connections = {}
  agent_state_nodes = {}

  message_info = {}

  # @todo - support multiple connection states per layer.
  hostLayerFunction_connections = [False]* layer_count
  hostLayerFunction_subgraph = graphviz.Digraph(
    name=CLUSTER_HOST_LAYER_FUNCTIONS_KEY  
  )
  hostLayerFunction_subgraph.attr(label=CLUSTER_HOST_LAYER_FUNCTIONS_LABEL)
  hostLayerFunction_subgraph.attr(color=CLUSTER_HOST_LAYER_FUNCTIONS_COLOR)
  hostLayerFunction_subgraph.attr(style=CLUSTER_HOST_LAYER_FUNCTIONS_STYLE)
  hostLayerFunction_subgraph.attr(penwidth=CLUSTER_PENWIDTH)

  if len(host_layer_functions) > 0:
    label=""
    group="host_layer_group"
    for i in range(layer_count + 1):
      # Add a (hidden) node per state.
      key="{:}_{:}".format("host", i)
      hostLayerFunction_subgraph.node(
        key, 
        label=label, 
        group=group,
        color=HOSTLAYERFUNCTION_STATE_COLOR, 
        fontcolor=HOSTLAYERFUNCTION_STATE_COLOR,
        shape=HOSTLAYERFUNCTION_STATE_SHAPE,
        style=HOSTLAYERFUNCTION_STATE_STYLE,
      )
      rank_list["s_{:}".format(i)].append(key)

    # Add an invisible link between the start of the layers graph and the first host layer function state node + last to end.
    layers_graph.edge(
      "invisible_layers_graph_start",
      "{:}_{:}".format("host", 0),
      color=HIDDEN_COLOR_START,
      style=HIDDEN_STYLE
    )
    layers_graph.edge(
      "{:}_{:}".format("host", layer_count),
      "invisible_layers_graph_end",
      color=HIDDEN_COLOR_END,
      style=HIDDEN_STYLE
    )



  for agent_idx, agent_name in enumerate(agent_names):
    agent_subgraphs[agent_name] = graphviz.Digraph(
      name=CLUSTER_AGENT_FUNCTIONS_KEY_PATTERN.format(agent_name)
    )
    agent_subgraphs[agent_name].attr(label=CLUSTER_AGENT_FUNCTIONS_LABEL_PATTERN.format(agent_name))
    agent_subgraphs[agent_name].attr(color=ith(CLUSTER_AGENT_FUNCTIONS_COLORS, agent_idx))
    agent_subgraphs[agent_name].attr(style=ith(CLUSTER_AGENT_FUNCTIONS_STYLES, agent_idx))
    agent_subgraphs[agent_name].attr(penwidth=CLUSTER_PENWIDTH)
    # Also create a data structure to show if a state/function is used or not. 
    agent_state_connections[agent_name] = {}
    # list of nodes to create per agent state
    agent_state_nodes[agent_name] = {}


    # Add each state once per layer + 1
    for state in agent_states[agent_name]:
      agent_state_connections[agent_name][state] = [{"direct": False, "out": False, "in": False} for i in range((layer_count + 1))]
      agent_state_nodes[agent_name][state] = {x:{"invisible": False} for x in range(layer_count + 1)}
      

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

        function_label_lines = [function_name, ""]
        function_color = FUNCTION_COLOR
        function_shape = FUNCTION_SHAPE
        if function_obj["globalCondition"] is not None:
          function_label_lines.append("globalCondtion: {:}".format(function_obj["globalCondition"]))
          function_color = GLOBAL_CONDITION_COLOR
          function_shape = GLOBAL_CONDITION_SHAPE

        if function_obj["rng"] and RNG_LABEL:
          function_label_lines.append(RNG_LABEL)

        if function_obj["reallocate"] and REALLOCATE_LABEL:
          function_label_lines.append(REALLOCATE_LABEL)

        if function_obj["xagentOutputs"] is not None and len(function_obj["xagentOutputs"]) > 0:
          for xagentOutput in function_obj["xagentOutputs"]:
            xagentOutput_name = xagentOutput["agent_name"]
            xagentOutput_state = xagentOutput["state"]
            xagentOutput_state_after = "{:}_{:}".format(xagentOutput_state, layerIndex + 1)
            # If the agent name is valid and the state is valid.
            if xagentOutput_name in agent_names and xagentOutput_state in agent_states[xagentOutput_name]:
              xagentOutput_label = XAGENT_OUTPUT_LABEL.format(xagentOutput_name, xagentOutput_state)
              # Add a line to the function.
              function_label_lines.append(xagentOutput_label)

              # Add a line, needs to be to the parent graph, so it can cross subgraph.
              # Also need to prevent folding in that case - Need to do the collapsed edges after all agent functions have been parsed. 

              if xagentOutput_name != agent_name:
                layers_graph.edge(
                  function_name,
                  xagentOutput_state_after,
                  xlabel = xagentOutput_label,
                  color = XAGENT_OUTPUT_COLOR,
                  fontcolor = XAGENT_OUTPUT_FONTCOLOR,
                  style = XAGENT_OUTPUT_STYLE,
                )
              else:
                agent_subgraphs[agent_name].edge(
                  function_name,
                  xagentOutput_state_after,
                  xlabel = xagentOutput_label,
                  color = XAGENT_OUTPUT_COLOR,
                  fontcolor = XAGENT_OUTPUT_FONTCOLOR,
                  style = XAGENT_OUTPUT_STYLE,
                )
              # Prevent folding.
              agent_state_connections[xagentOutput_name][xagentOutput_state][layerIndex+1]["in"] = True

        # Add a node for the function.
        function_label = "\n".join(function_label_lines)
        agent_subgraphs[agent_name].node(
          function_name,
          label=function_label,
          color = function_color,
          shape = function_shape,
        )

        # Add it to the relevant rank layer.
        rank_list["f_{:}".format(layerIndex)].append(function_name)

        # print(layerIndex, function_name)
        # print("  ", state_before, state_after)
        # Indicate that the state has an incoming / outgoing edge.
        edge_group = None
        edge_weight = None
        # If the function is direct, and there is no conditions on the function mark this case.
        if function_obj["currentState"] == function_obj["nextState"]:# and function_obj["condition"] is None:
          agent_state_connections[agent_name][function_obj["currentState"]][layerIndex]["direct"] = True
          edge_group = function_obj["currentState"]
          edge_weight = VERT_EDGE_WEIGHT
        agent_state_connections[agent_name][function_obj["currentState"]][layerIndex]["out"] = True
        agent_state_connections[agent_name][function_obj["nextState"]][layerIndex]["in"] = True

        # Add a link between the state_before and the function node, 
        # Optionally label with a function condition. @future - add a decision node?
        edge_label = None
        edge_fontcolor = EDGE_FONTCOLOR
        edge_color = EDGE_COLOR
        if function_obj["condition"] is not None:
          edge_label = function_obj["condition"]
          edge_fontcolor = CONDITION_FONTCOLOR
          edge_color = CONDITION_COLOR


        agent_subgraphs[agent_name].edge(
          state_before, 
          function_name,
          group=edge_group,
          xlabel=edge_label,
          fontcolor=edge_fontcolor,
          color=edge_color,
          tailport=GV_PORT_S,
          headport=GV_PORT_N,
          weight=edge_weight
        )
        # And a link from the function node to teh after state.
        agent_subgraphs[agent_name].edge(
          function_name, 
          state_after,
          group=edge_group,
          tailport=GV_PORT_S,
          headport=GV_PORT_N,
          weight=edge_weight
        )


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
        # Add a node for the host layer function.
        group="host_layer_group"
        hostLayerFunction_subgraph.node(
          function_name,
          shape=FUNCTION_SHAPE,
          group=group,
        )
        state_before = "{:}_{:}".format("host", layerIndex)
        state_after = "{:}_{:}".format("host", layerIndex + 1)

        # Add a link between the state_before and the function node, 
        hostLayerFunction_subgraph.edge(
          state_before, 
          function_name, 
          weight="10", 
          color=HOSTLAYERFUNCTION_LINK_COLOR, 
          group="host_layer_group",
          tailport=GV_PORT_S,
          headport=GV_PORT_N,
        )
        # And a link from the function node to the after state.
        hostLayerFunction_subgraph.edge(
          function_name, 
          state_after, 
          weight="10", 
          color=HOSTLAYERFUNCTION_LINK_COLOR, 
          group="host_layer_group",
          tailport=GV_PORT_S,
          headport=GV_PORT_N,
        )

        # Mark off the connection if staying in the same state
        hostLayerFunction_connections[layerIndex] = True

        # Add a rank same index.
        rank_list["f_{:}".format(layerIndex)].append(function_name)

  # Add missing links and find foldable LINKS
  foldable_state_state_links = OrderedDict()
  state_state_links_to_add = OrderedDict()
  # Should refactor this so that a dag is produced between states, that can be used for auto collapse, rather than this not-ideal datastructure.
  for agent_name in agent_names:
    foldable_state_state_links[agent_name] = {}
    state_state_links_to_add[agent_name] = {}
    for state in agent_states[agent_name]:
      foldable_state_state_links[agent_name][state] = []
      state_state_links_to_add[agent_name][state] = []
      prev_invisible = False
      for startLayer, connection_info in enumerate(agent_state_connections[agent_name][state]):
        if startLayer != len(agent_state_connections[agent_name][state]) - 1:

          # If nothing goes in or out of the state pair, mark as foldable (or render)
          if not any(connection_info.values()):
            # If in an invalid chain, make it invisible and remove the node?
            if prev_invisible:
                state_state_links_to_add[agent_name][state].append(((startLayer, startLayer+1, True)))
                prev_invisible = True
                # Flag the node as invisible, to maintain structure but not render the impossible state.
                agent_state_nodes[agent_name][state][startLayer]["invisible"] = True
            else:
              if STATE_FOLDING:
                foldable_state_state_links[agent_name][state].append((startLayer, startLayer+1))
              else:
                state_state_links_to_add[agent_name][state].append(((startLayer, startLayer+1, False)))
                prev_invisible = False
          else:
            # If there is a non-conditional linear s0->f->s1 relationship, we do not need a link..
            if connection_info["direct"]:
              prev_invisible = False
            # If there is a non-conditional, non-direct output we do not need a visible edge. BUT having an invisible edge helps with alignment it seems.
            elif connection_info["out"]:
              state_state_links_to_add[agent_name][state].append(((startLayer, startLayer+1, True)))
              prev_invisible = True
            else:
              state_state_links_to_add[agent_name][state].append(((startLayer, startLayer+1, False)))
              prev_invisible = False




  # The list of foldable links is in order per agent/state pair.
  # Iterate the lists pairwise, comparing the relevant components, building the list of newlinks.
  # There must be a beter way of doing this.
  for agent_name, states in foldable_state_state_links.items():
    for state, ijlist in states.items():
      new_links = []
      # Build a list of state nodes which are not needed
      folded_states = []
      if len(ijlist):
        src = ijlist[0][0]
        dst = ijlist[0][1]
        # Do not repeat the first element.
        for idx, ij in enumerate(ijlist[1:]):
          if ij[0] == dst:
            dst = ij[1]
            folded_states.append(ij[0])
          else:
            new_links.append((src, dst))
            src = ij[0]
            dst = ij[1]
        # Add the final pair
        new_links.append((src, dst))

      # For each new link, add it.
      for link in new_links:
        state_state_links_to_add[agent_name][state].append((link[0], link[1], False)) 

      # Remove each folded state from the global list of state nodes to create.
      for i in folded_states:
        agent_state_nodes[agent_name][state].pop(i)


  # Add state links.
  for agent_name, states in state_state_links_to_add.items():
    edge_weight = VERT_EDGE_WEIGHT
    for state, edges in states.items():
      for edge in sorted(edges): # sort for better debugging.
        state_before = "{:}_{:}".format(state, edge[0])
        state_after = "{:}_{:}".format(state, edge[1])
        # If the edge is invisible, make it so, but with a strong weight still;
        edge_style = GV_STYLE_SOLID
        if edge[2]:
          edge_style = GV_STYLE_INVIS
        agent_subgraphs[agent_name].edge(
            state_before, 
            state_after, 
            color=STATE_STATE_LINK_COLOR,
            style=edge_style,
            tailport=GV_PORT_S,
            headport=GV_PORT_N,
          )

  # Add state nodes.
  for agent_name, states in agent_state_nodes.items():
    for state, nodes in states.items():
      # Add the state nodes if not folded.
      for i in nodes:
        state_id = "{:}_{:}".format(state, i)
        state_label = state
        if DEBUG_LABLES:
          state_label = state_id
        node_style = GV_STYLE_SOLID
        if "invisible" in nodes[i] and nodes[i]["invisible"]:
          node_style = GV_STYLE_INVIS

        agent_subgraphs[agent_name].node(
          state_id, 
          label=state_label, 
          color=STATE_COLOR, 
          fontcolor=STATE_COLOR,
          group=state,
          style=node_style
        )
        # Add it to the relevant rank layer.
        rank_list["s_{:}".format(i)].append(state_id)     

      # Add an invisible link between the start of the layers graph and the first host layer function state node + last to end.
      layers_graph.edge(
        "invisible_layers_graph_start",
        "{:}_{:}".format(state, 0),
        color=HIDDEN_COLOR_START,
        style=HIDDEN_STYLE
      )
      layers_graph.edge(
        "{:}_{:}".format(state, layer_count),
        "invisible_layers_graph_end",
        color=HIDDEN_COLOR_END,
        style=HIDDEN_STYLE
      )   


  # Add missing links for host layer functions if required.
  if len(host_layer_functions) > 0:
    for startLayer, connected in enumerate(hostLayerFunction_connections):
      if not connected:
        state_before = "{:}_{:}".format("host", startLayer)
        state_after = "{:}_{:}".format("host", startLayer+1)
        # Add the edge
        hostLayerFunction_subgraph.edge(
          state_before, 
          state_after, 
          color=HOSTLAYERFUNCTION_LINK_COLOR,
          tailport=GV_PORT_S,
          headport=GV_PORT_N,
        )

        # Mark as done
        hostLayerFunction_connections[startLayer] = True



  # Add each agent_subgraph to the agent functions subgraph.
  for agent_subgraph in agent_subgraphs.values():
    layers_graph.subgraph(agent_subgraph)

  # If there are any host layer functions, add the host layer subgraph
  if len(host_layer_functions) > 0:
    layers_graph.subgraph(hostLayerFunction_subgraph)


  # Add messages nodes
  for message_name in message_info:
    message_node_key = "message_{:}".format(message_name)
    layers_graph.node(
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

      layers_graph.edge(
        function_key,
        message_node_key,
        color=MESSAGE_COLOR,
        penwidth="3",
        tailport=GV_PORT_S,
        # headport=GV_PORT_N, # ports on paral;lelograms are bad
      )

    # in
    for input_by in message_info[message_name]["input_by"]:
      function_key = "{:}".format(input_by["function"])

      layers_graph.edge(
        message_node_key,
        function_key,
        color=MESSAGE_COLOR,
        penwidth="3",
        # tailport=GV_PORT_S, # ports on paral;lelograms are bad
        headport=GV_PORT_N, 
      )

  # For each rank list element, specify the nodes as having the same rank.
    
  for k, nodes in rank_list.items():
    if len(nodes) > 0:
      ranksame_string = "\t{ rank=same; " + "; ".join(nodes) + "; }"
      layers_graph.body.append(ranksame_string)
    

  iteration_graph.subgraph(layers_graph)

  dot.subgraph(iteration_graph)


  # Make a subgraph for cluster ordering
  # Each cluster needs a start and an end node for this to work. 
  # With invisible links between the inviisble start and the first actual element (s0) etc. 
  # And invisible link between last real and invisible end.
  # Then each cluster needs to be linked together in order with invisible links / real links with head/tail specifications.

  # Start -> init / iterations
  # init -> iteration

  # iteration_start -> Step_start
  # Step_end -> layers_start
  # layers_start -> hostlayers_start
  # layers_start -> each agent state 0.
  # each agebt state N -> layers_end
  # hostlayers_end -> layers_end
  # layers_end -> exit_start
  # exit_end -> stop/end.


  # Connect the clusters.
  if render_init_functions(args):
    dot.edge(
      "invisible_initFunctions_end",
      ITERATION_START_KEY, 
      ltail=CLUSTER_INIT_FUNCTIONS_KEY, 
      lhead=CLUSTER_ITERATION_GRAPH_KEY,
    )
  if render_exit_functions(args):
    dot.edge(
      ITERATION_END_KEY,
      "invisible_exitFunctions_start", 
      ltail=CLUSTER_ITERATION_GRAPH_KEY, 
      lhead= CLUSTER_EXIT_FUNCTIONS_KEY,
    )


  # Calculate where start and end should connect to. Default to the iteration subgraph cluster
  start_dest_node = ITERATION_START_KEY
  start_dest_cluster = CLUSTER_ITERATION_GRAPH_KEY
  end_source_node = ITERATION_END_KEY
  end_source_cluster = CLUSTER_ITERATION_GRAPH_KEY

  # If we have an init, connect the start to init.
  if render_init_functions(args):
    start_dest_node = "invisible_initFunctions_start"
    start_dest_cluster = CLUSTER_INIT_FUNCTIONS_KEY

  # if render_step_functions(args):

  # If we have exit functions, the end connects to the exit.
  if render_exit_functions(args):
    end_source_node = "invisible_exitFunctions_end"
    end_source_cluster = CLUSTER_EXIT_FUNCTIONS_KEY

  # Actually add the start / end links.
  dot.edge(
    START_KEY,
    start_dest_node, 
    lhead=start_dest_cluster,
    group="outer",
    tailport=GV_PORT_S,
    headport=GV_PORT_N,
  )
  dot.edge(
    end_source_node, 
    STOP_KEY,
    ltail=end_source_cluster,
    group="outer",
    tailport=GV_PORT_S,
    headport=GV_PORT_N,
  )

  # dot.attr(rank="same")

  return dot


def output_graphviz(args, graphviz_data):
  verbose_log(args, "Outputting Graphviz Data")

  if graphviz_data is None:
    raise ValueError("Invalid Graphviz Data")

  if args.output:
    # If rendering or showing with an output file, do so
    if args.render or args.show:
      graphviz_data.render(args.output, view=args.show)
    # if only saving the gv file, do so.
    else: 
      graphviz_data.save(args.output)
  else:
    # If showing but no output gv file, render to a temp. 
    if args.show:
      graphviz_data.view(tempfile.mktemp('.gv'))
    # Otherwise, just print the output to console.
    else:
      print(graphviz_data)

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

#! /usr/bin/env python

"""
Python script to generate a 2D grid of vertices, connected vertically and horizontally to the immediate neighbours.
@author Peter Heywood <p.heywood@sheffield.ac.uk>
"""

import os
import sys
import json
import argparse
from distutils.util import strtobool
from collections import OrderedDict
import xml.etree.ElementTree as ET
from xml.dom import expatbuilder


JSON_INDENT = 4

class GridGenerator:

    KEY_VERTICES = "vertices"
    KEY_EDGES = "edges"

    KEY_VERTEX_ID = "id"
    KEY_VERTEX_X = "x"
    KEY_VERTEX_Y = "y"
    KEY_VERTEX_Z = "z"

    KEY_EDGE_ID = "id"
    KEY_EDGE_SOURCE = "source"
    KEY_EDGE_DEST = "destination"
    KEY_EDGE_LENGTH = "length"
    KEY_EDGE_CAPACITY = "capacity"

    DEFAULT_GRID_SIZE = 4
    DEFAULT_EDGE_LENGTH = 90.0
    DEFAULT_EDGE_CAPACITY = 10
    DEFAULT_VERTEX_Z = 0.0

    def __init__(
        self,
        grid_size=None,
        edge_length=None,
        edge_capacity=None,
        spatial=False,
        force=False,
        verbose=False,
        pretty=True
    ):

        self.grid_size = grid_size if grid_size is not None else self.DEFAULT_GRID_SIZE
        self.edge_length = edge_length if edge_length is not None else self.DEFAULT_EDGE_LENGTH
        self.edge_capacity = edge_capacity if edge_capacity is not None else self.DEFAULT_EDGE_CAPACITY

        self.spatial = spatial
        self.force = force
        self.verbose = verbose
        self.pretty = pretty

        self.network_data, self.network_csr  = self.generate_network_json()

        self.output_spatial_configuration()
        self.output_information()

    def get_vertex_id(self, row, col):
        return (row * self.grid_size) + col

    def number_of_edges(self):
        return len(self.network_data[self.KEY_EDGES]) if self.KEY_EDGES in self.network_data else 0

    def number_of_vertices(self):
        return len(self.network_data[self.KEY_VERTICES]) if self.KEY_VERTICES in self.network_data else 0

    def generate_network_json(self):
        network_obj = OrderedDict()
        network_csr = []
        # Add a 0 initial value to network csr
        network_csr.append(0)

        # Calculate the number of vertices in the grid.
        num_vertices = self.grid_size * self.grid_size
        # Calculate the number of edges in the grid.
        # 2 edges between each vertex pair in the square based grid. 4(N(N-1))
        num_edges = 4 * (self.grid_size * (self.grid_size - 1));

        # Generate the required number of vertices in the appropriate locations for CSR to make some sense.
        # Also generate the edges which leave each vertex at the same time.
        vertices = []
        edges = []

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Construct the vertex
                vertex = OrderedDict()
                # Generate id.
                index = self.get_vertex_id(row, col)
                # If the index is >= 0 we continue.
                if index >= 0:
                    vertex[self.KEY_VERTEX_ID] = index
                    # Generate the x y z values based on col/row
                    vertex[self.KEY_VERTEX_X] = self.edge_length * col
                    vertex[self.KEY_VERTEX_Y] = self.edge_length * row
                    vertex[self.KEY_VERTEX_Z] = self.DEFAULT_VERTEX_Z
                    # Append the vertex to the list.
                    vertices.append(vertex)                

                    # Construct each edge. There should be up to 4 edges which leave each vertex.
                    # Ordered N W E S to ensure Bnode is increasing.
                    # Nodes in the outer row/col at each end are entrance / exit only so only have a single edge leaving them.


                    # NORTH
                    if row > 0 :
                        destination = self.get_vertex_id(row - 1, col)
                        if destination >= 0:
                            edge = OrderedDict()
                            edge[self.KEY_EDGE_ID] = len(edges)
                            edge[self.KEY_EDGE_SOURCE] = index
                            edge[self.KEY_EDGE_DEST] = destination
                            edge[self.KEY_EDGE_LENGTH] = self.edge_length
                            edge[self.KEY_EDGE_CAPACITY] = self.edge_capacity
                            edges.append(edge)
                    # WEST
                    if col > 0:
                        destination = self.get_vertex_id(row, col - 1)
                        if destination >= 0:
                            edge = OrderedDict()
                            edge[self.KEY_EDGE_ID] = len(edges)
                            edge[self.KEY_EDGE_SOURCE] = index
                            edge[self.KEY_EDGE_DEST] = destination
                            edge[self.KEY_EDGE_LENGTH] = self.edge_length
                            edge[self.KEY_EDGE_CAPACITY] = self.edge_capacity
                            edges.append(edge)
                    # EAST
                    if col < self.grid_size - 1:
                        destination = self.get_vertex_id(row, col + 1)
                        if destination >= 0:
                            edge = OrderedDict()
                            edge[self.KEY_EDGE_ID] = len(edges)
                            edge[self.KEY_EDGE_SOURCE] = index
                            edge[self.KEY_EDGE_DEST] = destination
                            edge[self.KEY_EDGE_LENGTH] = self.edge_length
                            edge[self.KEY_EDGE_CAPACITY] = self.edge_capacity
                            edges.append(edge)
                    # SOUTH
                    if row < self.grid_size - 1:
                        destination = self.get_vertex_id(row + 1, col)
                        if destination >= 0:
                            edge = OrderedDict()
                            edge[self.KEY_EDGE_ID] = len(edges)
                            edge[self.KEY_EDGE_SOURCE] = index
                            edge[self.KEY_EDGE_DEST] = destination
                            edge[self.KEY_EDGE_LENGTH] = self.edge_length
                            edge[self.KEY_EDGE_CAPACITY] = self.edge_capacity
                            edges.append(edge)

                    # Set the start index of the edges for the vertex in the csr.
                    network_csr.append(len(edges))

        network_obj[self.KEY_VERTICES] = vertices
        network_obj[self.KEY_EDGES] = edges

        # print(len(vertices), len(edges))
        return network_obj, network_csr

    def get_road_network_edge_count(self):
        return len(self.network_data[self.KEY_EDGES])

    def get_road_network_vertex_count(self):
        return len(self.network_data[self.KEY_VERTICES])

    def output_network_to_disk(self, output_file):
        self.output_json_to_disk(output_file, self.network_data)

    def output_json_to_disk(self, output_file, data):

        if self.output_file_overwrite_check(output_file):
            print("Output JSON to Disk {:}".format(output_file))
            with open(output_file, "w") as f:
                if self.pretty:
                    json.dump(data, f, sort_keys=False, indent=4)
                else:
                    json.dump(data, f, sort_keys=False)
            return True
        else:
            printf("JSON not written.")
            return False

    def output_file_exists(self, output_file):
        return os.path.isfile(output_file)

    def output_file_overwrite_check(self, output_file):
        if self.output_file_exists(output_file):
            overwrite = self.force or self.user_yes_no_query("The output file exists, do you wish to overwrite?")
            return overwrite
        else:
            return True


    def user_yes_no_query(self, question):
        # http://stackoverflow.com/questions/3041986/python-command-line-yes-no-input
        sys.stdout.write('%s [y/n]\n' % question)
        while True:
            try:
                return strtobool(input().lower())
            except ValueError:
                sys.stdout.write('Please respond with \'y\' or \'n\'.\n')


    def output_spatial_configuration(self):
        if self.spatial:
            # Find the min and max X/y/z.
            xvals = [v["x"] for v in self.network_data[self.KEY_VERTICES]]
            yvals = [v["y"] for v in self.network_data[self.KEY_VERTICES]]
            zvals = [v["z"] for v in self.network_data[self.KEY_VERTICES]]

            min_x = min(xvals)
            max_x = max(xvals)
            min_y = min(yvals)
            max_y = max(yvals)
            min_z = min(zvals)
            max_z = max(zvals)

            # Calc the range
            range_x = max_x - min_x
            range_y = max_y - min_y
            range_z = max_z - min_z

            # If any range is less than the edge length (i.e. z) bump the max val.
            if(range_x < self.edge_length):
                max_x = self.edge_length
            if(range_y < self.edge_length):
                max_y = self.edge_length
            if(range_z < self.edge_length):
                max_z = self.edge_length


            # Calc the comm radius
            range_min = min([max_x, max_y, max_z])
            comm_radius = min(range_min, self.edge_length)
            # If it does not divide perfectly into the min range, round up?
            # @future - this should not be an issue for this generator. 

            # Generate XML
            xml_root = ET.Element("gpu:partitioningSpatial")
            xml_radius = ET.SubElement(xml_root, "gpu:radius")
            xml_radius.text = str(comm_radius)
            xml_xmin = ET.SubElement(xml_root, "gpu:xmin")
            xml_xmin.text = str(min_x)
            xml_xmax = ET.SubElement(xml_root, "gpu:xmax")
            xml_xmax.text = str(max_x)
            xml_ymin = ET.SubElement(xml_root, "gpu:ymin")
            xml_ymin.text = str(min_y)
            xml_ymax = ET.SubElement(xml_root, "gpu:ymax")
            xml_ymax.text = str(max_y)
            xml_zmin = ET.SubElement(xml_root, "gpu:zmin")
            xml_zmin.text = str(min_z)
            xml_zmax = ET.SubElement(xml_root, "gpu:zmax")
            xml_zmax.text = str(max_z)

            # Print to stdout
            ugly_xml = ET.tostring(xml_root, 'utf-8')
            reparsed = expatbuilder.parseString(ugly_xml, False)
            pretty_xml = reparsed.toprettyxml(indent="\t")
            if len(pretty_xml) > 0:
                start_char = pretty_xml.find('\n') + 1
                print(pretty_xml[start_char:])


    def output_information(self):

        edge_count = len(self.network_data[self.KEY_EDGES])
        vertex_count = len(self.network_data[self.KEY_VERTICES])
        capacity = edge_count * self.edge_capacity

        print("edge count   {:}".format(edge_count))
        print("vertex count {:}".format(vertex_count))
        print("capacity     {:}".format(capacity))


        print("info!")

def main():
    parser = argparse.ArgumentParser(description="Generate a grid network stored as JSON")
    parser.add_argument(
        "network_file",
        type=str,
        help="file for network output"
    )
   
    parser.add_argument(
        "-g",
        "--grid-size",
        type=int,
        help="The grid size for the artificial network"
    )
    parser.add_argument(
        "-l",
        "--edge-length",
        type=float,
        help="The length for each edge in the grid."
    )
    parser.add_argument(
        "-c",
        "--edge-capacity",
        type=int,
        help="The capacity for each edge in the grid."
    )
    parser.add_argument(
        "--spatial",
        action="store_true",
        help="output the spatial partitioning configuration required."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="increase verbosity of output"
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="force overwriting of files"
    )
    parser.add_argument(
        "-p",
        "--pretty",
        action="store_true",
        help="Produce pretty-printed JSON output (newlines)"
    )

    args = parser.parse_args()

    generator = GridGenerator(
        args.grid_size,
        args.edge_length,
        args.edge_capacity,
        args.spatial,
        args.force,
        args.verbose,
        args.pretty
    )
    generator.output_network_to_disk(args.network_file)

if __name__ == "__main__":
    main();

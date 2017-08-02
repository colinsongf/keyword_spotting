# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Transforms a float-trained graph into an equivalent quantized version.

An example of command-line usage is:
bazel build tensorflow/contrib/quantization/tools:quantize_graph \
&& bazel-bin/tensorflow/contrib/quantization/tools/quantize_graph \
--input=tensorflow_inception_graph.pb
--output_node_names="softmax2" --print_nodes --output=/tmp/quantized_graph.pb \
--mode=eightbit --logtostderr

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import collections
import tensorflow as tf

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import app
from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import gfile
from .octbit_ops import octbit_mat_mul

flags = flags_lib
FLAGS = flags.FLAGS


flags.DEFINE_boolean("octbit_print_nodes", False, """Lists all nodes in the model.""")
flags.DEFINE_string("octbit_input", "", """TensorFlow 'GraphDef' file to load.""")
flags.DEFINE_string("octbit_output_node_names", "",
                    """Output node names, comma separated.""")
flags.DEFINE_string("octbit_output", "", """File to save the output graph to.""")
flags.DEFINE_integer("octbit_bitdepth", 8,
                     """How many bits to quantize the graph to.""")
flags.DEFINE_string("octbit_mode", "round",
                    """What transformation to apply (round, quantize,"""
                    """ eightbit, weights, or weights_rounded).""")
flags.DEFINE_string("octbit_test_input_dims", "1,224,224,3",
                    """The size of the input tensor to use when testing a"""
                    """ graph loaded from a file.""")
flags.DEFINE_boolean("octbit_strip_redundant_quantization", True,
                     """Removes redundant dequantize/quantize pairs.""")
flags.DEFINE_boolean("octbit_load_quantization_so", True,
                     """Explicitly load the quantization ops library""")


def print_input_nodes(current_node, nodes_map, indent, already_visited):
    print(" " * indent + current_node.op + ":" + current_node.name)
    already_visited[current_node.name] = True
    for input_node_name in current_node.input:
        if input_node_name in already_visited:
            continue
        input_node = nodes_map[input_node_name]
        print_input_nodes(input_node, nodes_map, indent + 1, already_visited)


def create_node(op, name, inputs):
    new_node = node_def_pb2.NodeDef()
    new_node.op = op
    new_node.name = name
    for input_name in inputs:
        new_node.input.extend([input_name])
    return new_node


def create_constant_node(name, value, dtype, shape=None):
    node = create_node("Const", name, [])
    set_attr_dtype(node, "dtype", dtype)
    set_attr_tensor(node, "value", value, dtype, shape)
    return node


def copy_attr(node, key, attr_value):
    try:
        node.attr[key].CopyFrom(attr_value)
    except KeyError:
        pass


def set_attr_dtype(node, key, value):
    try:
        node.attr[key].CopyFrom(
                attr_value_pb2.AttrValue(type=value.as_datatype_enum))
    except KeyError:
        pass


def set_attr_shape(node, key, value):
    try:
        node.attr[key].CopyFrom(
                attr_value_pb2.AttrValue(shape=tensor_shape.as_shape(value).as_proto()))
    except KeyError:
        pass


def set_attr_tensor(node, key, value, dtype, shape=None):
    try:
        node.attr[key].CopyFrom(
                attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                        value, dtype=dtype, shape=shape)))
    except KeyError:
        pass


def set_attr_string(node, key, value):
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(s=value))
    except KeyError:
        pass


def set_attr_int_list(node, key, value):
    list_value = attr_value_pb2.AttrValue.ListValue(i=value)
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(list=list_value))
    except KeyError:
        pass


def set_attr_bool(node, key, value):
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(b=value))
    except KeyError:
        pass


def set_attr_int(node, key, value):
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(i=value))
    except KeyError:
        pass


def set_attr_float(node, key, value):
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(f=value))
    except KeyError:
        pass


def node_name_from_input(node_name):
    """Strips off ports and other decorations to get the underlying node name."""
    if node_name.startswith("^"):
        node_name = node_name[1:]
    m = re.search(r"(.*):\d+$", node_name)
    if m:
        node_name = m.group(1)
    return node_name


def ensure_tensor_name_has_port(node_name):
    """Makes sure that a tensor name has :0 if no explicit port exists."""
    m = re.search(r"(.*):\d+$", node_name)
    if m:
        name_with_port = node_name
    else:
        name_with_port = node_name + ":0"
    return name_with_port


def unique_node_name_from_input(node_name):
    """Replaces invalid characters in input names to get a unique node name."""
    return node_name.replace(":", "__port__").replace("^", "__hat__")


def octize_weight_int8_signed(input_node):
    input_tensor = input_node.attr["value"].tensor
    tensor_value = tensor_util.MakeNdarray(input_tensor)
    tensor_shape = input_tensor.tensor_shape
    tensor_shape_list = tensor_util.TensorShapeProtoToList(tensor_shape)
    nmax = max(abs(tensor_value.max()), abs(tensor_value.min()))
    scale = nmax / 127.
    b = np.zeros(shape=(tensor_shape_list[1]), dtype=float)
    transpose_tensor_value = np.zeros(shape=(tensor_shape_list[1],
                                      tensor_shape_list[0]), dtype=float)
    tensor_value = np.round(tensor_value / scale)
    for i in range(tensor_shape_list[0]):
        for j in range(tensor_shape_list[1]):
            b[j] += tensor_value[i, j] * 127
    for i in range(tensor_shape_list[1]):
        for j in range(tensor_shape_list[0]):
            transpose_tensor_value[i][j] = tensor_value[j][i]

    new_node = create_constant_node(input_node.name,
                                    transpose_tensor_value,
                                    dtypes.qint8,
                                    shape=[
                                        tensor_shape_list[1],
                                        tensor_shape_list[0]])
    return new_node, scale, b


def default_octbit_matmul_name_check(name):
    '''update date: 20170308
        related structure: GRURNCell'''
    # not softmax matrix and rnn layers' matrix
    if name != "model/linear/linear/MatMul" and \
            "MatMul" in name and not ("cell_0" in name):
        return True
    return False


def attention_model_matmul_name_check(name):
    '''update date: 20170704
       related structure: attention_ctc/model/linear_transform
    '''
    # weight matrix except softmax and input trans
    if name != "model/input_linear_trans_1/MatMul" and \
            name != "model/output_linear_trans_1/MatMul" and \
            name != "model/input_feed_forward/conv1_1/MatMul" and \
            name != "model/input_feed_forward/conv2_1/MatMul" and \
            name.endswith("/MatMul"):
        return True
    return False
    
def simplify_frozen_graph(frozen_graph_def, session):
    '''remove redundant nodes in given frozen graph.
        related structure: GRURNCell's l2_normalize'''
    def _get_value(input_node):
        input_tensor = input_node.attr["value"].tensor
        tensor_value = tensor_util.MakeNdarray(input_tensor)
        return tensor_value

    nd_a = dict([(i.name, i) for i in frozen_graph_def.node])

    norm_input = False
    norm_state = False
    for k, v in nd_a.items():
        if k.endswith("input_trans_matrix/linear_weights/scale"):
            norm_input = True
        if k.endswith("state_trans_matrix/linear_weights/scale"):
            norm_state = True

    # reorder the node's connection
    for nd in frozen_graph_def.node:
        if norm_input:
            if "MatMul/Enter" in nd.name:
                nd.input[0] = '/'.join(
                    nd.input[0].split('/')[:-1]) + "/linear_weights/read"
        if norm_state:
            if "MatMul_1/Enter" in nd.name:
                 nd.input[0] = '/'.join(
                    nd.input[0].split('/')[:-1]) + "/linear_weights/read"

    # modify value
    new_frozen_graph_def = graph_pb2.GraphDef()
    for nd in frozen_graph_def.node:
        if (norm_input and \
                nd.name.endswith("/input_trans_matrix/linear_weights")) or \
            (norm_state and \
                nd.name.endswith("/state_trans_matrix/linear_weights")):
            print("cal l2_norm for node", nd.name)
            var = _get_value(nd)
            scale = _get_value(nd_a[nd.name + "/scale"])
            dim = _get_value(nd_a['/'.join(nd.name.split('/')[:-1]) + \
                "/l2_normalize/Sum/reduction_indices"])
            norm_var = session.run(
                tf.multiply(scale, tf.nn.l2_normalize(var, dim=dim)))
            replace_node = create_constant_node(
                nd.name, norm_var, tf.float32, list(np.shape(norm_var)))
            new_frozen_graph_def.node.extend([replace_node])
        else:
            new_frozen_graph_def.node.extend([nd])
    return new_frozen_graph_def


def simplify_frozen_graph_for_attention(frozen_graph_def, session):
    '''remove redundant nodes in given frozen graph.
        related structure: attention model's l2_normalize'''
    def _get_value(input_node):
        input_tensor = input_node.attr["value"].tensor
        tensor_value = tensor_util.MakeNdarray(input_tensor)
        return tensor_value

    nd_a = dict([(i.name, i) for i in frozen_graph_def.node])

    norm_input = False
    for k, v in nd_a.items():
        if k.endswith("scale"):
            norm_input = True
            break

    # reorder the node's connection
    delete_nodes_name = []
    for nd in frozen_graph_def.node:
        if norm_input:
            if nd.name.endswith("/MatMul") and \
                    not nd.name.endswith("self_attention/MatMul"):
                assert(nd.input[1].endswith("/Squeeze"))
                delete_nodes_name.append(nd.input[1])
                tmp = '_'.join(nd.input[1].split('/')[-2].split("_")[:-1])
                nd.input[1] = '/'.join(
                    nd.input[1].split('/')[:-2]) + "/%s/weights/read" % tmp

    # modify value
    new_frozen_graph_def = graph_pb2.GraphDef()
    for nd in frozen_graph_def.node:
        if (norm_input and \
                nd.name.endswith("/weights")):
            print("cal l2_norm for node", nd.name)
            var = _get_value(nd)
            scale = _get_value(nd_a[nd.name[:-8] + "/scale"])
            tmp = nd.name.split('/')[-2] + "_1"
            dim = _get_value(nd_a['/'.join(nd.name.split('/')[:-2]) + \
                "/%s/l2_normalize/Sum/reduction_indices" % tmp])
            norm_var = session.run(
                tf.squeeze(
                    tf.multiply(scale, tf.nn.l2_normalize(var, dim=dim)),
                    [0, 1]
                )
            )
            replace_node = create_constant_node(
                nd.name, norm_var, tf.float32, list(np.shape(norm_var)))
            new_frozen_graph_def.node.extend([replace_node])
        elif nd.name not in delete_nodes_name:
            new_frozen_graph_def.node.extend([nd])

    return new_frozen_graph_def


def simplify_frozen_graph_for_ptc(frozen_graph_def, session):
    '''remove redundant nodes in given frozen graph.
        related structure: RankNormalizedClippedBasicLSTMCell's
        l2_normalize'''
    def _get_value(input_node):
        input_tensor = input_node.attr["value"].tensor
        tensor_value = tensor_util.MakeNdarray(input_tensor)
        return tensor_value

    nd_a = dict([(i.name, i) for i in frozen_graph_def.node])

    # TODO: no state!
    norm_input = False
    for k, v in nd_a.items():
        if k.endswith(
                "/RankNormalizedClippedBasicLSTMCell/input_weights_scale"):
            norm_input = True

    # reorder the node's connection
    new_nodes = []
    for nd in frozen_graph_def.node:
        # MatMul node
        if norm_input and \
                "RankNormalizedClippedBasicLSTMCell/MatMul" in nd.name:
            # TODO: different frames problem? need a special node
            # whose child node is input_weights and its frame is
            # in 'while'
            scope = '/'.join(nd.name.split('/')[:-1])
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(
                nd_a[scope + "/l2_normalize/Square/Enter"])
            new_node.name = scope + "/new_node"
            new_nodes.append(new_node)
            nd.input[1] = new_node.name

    # modify value
    new_frozen_graph_def = graph_pb2.GraphDef()
    for nd in frozen_graph_def.node:
        if (norm_input and \
                nd.name.endswith(
                    "/RankNormalizedClippedBasicLSTMCell/input_weights")):
            print("cal l2_norm for node", nd.name)
            var = _get_value(nd)
            scale = _get_value(nd_a[nd.name + "_scale"])
            # TODO: only support -1
            dim = -1
            norm_var = session.run(
                tf.multiply(scale, tf.nn.l2_normalize(var, dim=dim)))
            replace_node = create_constant_node(
                nd.name, norm_var, tf.float32, list(np.shape(norm_var)))
            new_frozen_graph_def.node.extend([replace_node])
        else:
            new_frozen_graph_def.node.extend([nd])
    # add new nodes into graph
    new_frozen_graph_def.node.extend(new_nodes)
    return new_frozen_graph_def


class GraphRewriter(object):
    """Takes a float graph, and rewrites it in quantized form."""

    def __init__(self, input_graph, mode="octbit",
                 transfer_model = "rnn"):
        """Sets up the class to rewrite a float graph.

        Args:
            input_graph: A float graph to transform.
            mode: A string controlling how quantization is performed -
                round, quantize, eightbit, or weights.

        Raises:
            ValueError: Two nodes with the same name were found in the graph.
        """
        self.input_graph = input_graph
        self.nodes_map = self.create_nodes_map(input_graph)
        self.output_graph = None
        self.mode = mode
        self.dict_input = dict()
        if transfer_model == "rnn":
            self.matmul_name_check = default_octbit_matmul_name_check
        elif transfer_model == "attention":
            self.matmul_name_check = attention_model_matmul_name_check
        self.debug_print = False

    def create_nodes_map(self, graph):
        """Builds a mapping of node names to their defs from the graph."""
        nodes_map = {}
        for node in graph.node:
            if node.name not in nodes_map.keys():
                nodes_map[node.name] = node
            else:
                raise ValueError("Duplicate node names detected.")
        return nodes_map

    def rewrite(self, output_node_names):
        """Triggers rewriting of the float graph.

        Args:
            output_node_names: A list of names of the nodes that produce the final
                results.

        Returns:
            A quantized version of the float graph.
        """
        self.output_graph = graph_pb2.GraphDef()
        output_nodes = [self.nodes_map[output_node_name]
                        for output_node_name in output_node_names]
        if self.mode == "octbit":
            self.already_visited = {}
            for output_node in output_nodes:
                self.octize_nodes_recursively(output_node)
        else:
            print("Bad mode - " + self.mode + ".")
        return self.output_graph

    def octize_nodes_recursively(self, current_node):
        """The entry point for quantizing nodes to eight bit and back."""
        if current_node.name in self.already_visited:
            return
        self.already_visited[current_node.name] = True

        input_node_name_matmul = None
        if self.matmul_name_check(current_node.name) and \
                current_node.op == "MatMul":
            input_node_name_matmul = self.octbit_matmul_prologue_node(
                    self.nodes_map[node_name_from_input(current_node.input[1])])

        for input_node_name in current_node.input:
            input_node_name = node_name_from_input(input_node_name)
            input_node = self.nodes_map[input_node_name]
            self.octize_nodes_recursively(input_node)

        if self.matmul_name_check(current_node.name) and \
                current_node.op == "MatMul":
            print("octbit convert op:", current_node.name)
            self.octbit_matmul_node(current_node, input_node_name_matmul)
        else:
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(current_node)
            self.add_output_graph_node(new_node)


    def octbit_matmul_prologue_node(self, current_node):
        if self.debug_print:
            print("now", current_node.name)
        if current_node.name in self.already_visited:
            if self.debug_print:
                print("enter already visit!")
            if current_node.op == "Const":
                return current_node.name
        else:
            self.already_visited[current_node.name] = True
            if current_node.op == "Enter" or \
                    current_node.op == "Identity":
                if self.debug_print:
                    print(current_node.name,
                          "->child->",
                          current_node.input[0])
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(current_node)
                new_node.name = current_node.name
                set_attr_dtype(new_node, "T", dtypes.qint8)
                self.add_output_graph_node(new_node)
            elif current_node.op == "Const":
                if self.debug_print:
                    print(current_node.name, "is octbit const")
                new_node, scale, bias = octize_weight_int8_signed(current_node)
                set_attr_dtype(new_node, "dtype", dtypes.qint8)
                self.dict_input[current_node.name] = (scale, bias)
                self.add_output_graph_node(new_node)
                return current_node.name
            else:
                raise ValueError("unexpected op on %s, op = %s" %
                    (current_node.name, current_node.op))
        if len(current_node.input) != 1:
            raise ValueError("unexpected input number on %s" %
                (current_node.name, current_node.op))
        return self.octbit_matmul_prologue_node(
            self.nodes_map[node_name_from_input(current_node.input[0])])


    def octbit_matmul_node(self, current_node, input_node_name):
        scale, bias = self.dict_input[input_node_name]
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(current_node)
        new_node.op = "OctbitMatMul"
        set_attr_dtype(new_node, "T", dtypes.float32)
        set_attr_bool(new_node, "transpose_b", True)
        set_attr_float(new_node, "scale", scale)
        set_attr_tensor(new_node, "bias", bias, dtypes.float32)
        self.add_output_graph_node(new_node)

    def add_output_graph_node(self, output_node):
        """Inserts one node into the new graph."""
        self.output_graph.node.extend([output_node])

    def remove_dead_nodes(self, output_names):
        """Removes nodes that are no longer needed for inference from the graph."""
        old_output_graph = self.output_graph
        self.output_graph = graph_util.extract_sub_graph(old_output_graph,
                                                         output_names)

    def set_input_graph(self, new_input_graph):
        self.input_graph = new_input_graph
        self.nodes_map = self.create_nodes_map(self.input_graph)


def main(unused_args):
    if not gfile.Exists(FLAGS.input):
        print("Input graph file '" + FLAGS.input + "' does not exist!")
        return -1

    known_modes = ["octbit"]
    if not any(FLAGS.mode in s for s in known_modes):
        print("mode is '" + FLAGS.mode + "', not in " + ", ".join(known_modes) + ".")
        return -1

    tf_graph = graph_pb2.GraphDef()
    with gfile.Open(FLAGS.input, "rb") as f:
        data = f.read()
        tf_graph.ParseFromString(data)

    graph = ops.Graph()
    with graph.as_default():
        importer.import_graph_def(tf_graph, input_map={}, name="")

    rewriter = GraphRewriter(tf_graph, FLAGS.mode)

    output_graph = rewriter.rewrite(FLAGS.output_node_names.split(","))

    f = gfile.FastGFile(FLAGS.output, "wb")
    f.write(output_graph.SerializeToString())

    return 0


if __name__ == "__main__":
    app.run()


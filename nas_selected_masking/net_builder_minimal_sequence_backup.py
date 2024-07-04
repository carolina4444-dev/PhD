import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal

LEAF_ENCODING = 1

class TreeNode:
    def __init__(self, value, input_shape=None, conv1d_params=None):
        self.value = value
        self.input_shape = input_shape
        self.conv1d_params = conv1d_params
        self.conv1d_layer = None
        self.children = []
        self.parent = None  # Add a parent attribute


def calculate_conv1d_output_shape(input_shape, kernel_size, strides, filters):
    input_length, input_channels = input_shape
    output_length = (input_length - kernel_size) // strides + 1
    return (output_length, filters)


def dfs_traverse(sequence, conv1d_params=None, parent=None, index=0, initial_shape=None):
    if index >= len(sequence) or sequence[index] == LEAF_ENCODING:
        return None, index + 1

    node_value = sequence[index]
    node = TreeNode(value=node_value)
    # Set the parent for the current node
    node.parent = parent

    if parent is not None:
        # Calculate Conv1D parameters based on the parent node's output shape
        parent_output_shape = calculate_conv1d_output_shape(parent.conv1d_params['input_shape'],
                                                             parent.conv1d_params['kernel_size'],
                                                             parent.conv1d_params['strides'],
                                                             parent.conv1d_params['filters'])

        # Set the input shape for the current node
        node.input_shape = parent_output_shape

        # Calculate Conv1D parameters for the current node
        node.conv1d_params = {
            'input_shape': parent_output_shape,
            'kernel_size': min(parent.conv1d_params['kernel_size'], parent_output_shape[1]),
            'strides': min(parent.conv1d_params['strides'], parent_output_shape[1] - parent.conv1d_params['kernel_size'] + 1),
            'filters': parent.conv1d_params['filters']
        }

        
    else:  # Root node
        # Set the input shape for the current node
        node.input_shape = conv1d_params['input_shape']

        # Use the provided conv1d_params for the root node
        node.conv1d_params = {
            'input_shape': conv1d_params['input_shape'],
            'kernel_size': conv1d_params['kernel_size'],
            'strides': conv1d_params['strides'],
            'filters': conv1d_params['filters']
        }

    index += 1

    while index < len(sequence) and sequence[index] != LEAF_ENCODING:
        child, index = dfs_traverse(sequence, conv1d_params=node.conv1d_params, parent=node, index=index,
                                    initial_shape=initial_shape)
        node.children.append(child)

    return node, index + 1



def infer_output_shape(node):
    if node.parent is not None and node.parent.conv1d_layer is not None:
        # Flow down: The output shape of the parent's Conv1D layer affects the child's output shape
        parent_output_shape = calculate_conv1d_output_shape(node.parent.conv1d_params['input_shape'],
                                                             node.parent.conv1d_params['kernel_size'],
                                                             node.parent.conv1d_params['strides'],
                                                             node.parent.conv1d_params['filters'])
    else:
        # If no parent or parent has no Conv1D layer, use the initial input shape
        parent_output_shape = node.input_shape

    # Infer output shape based on Conv1D layer parameters and input shape
    filters = node.conv1d_params['filters']
    kernel_size = min(node.conv1d_params['kernel_size'], parent_output_shape[1])
    strides = min(node.conv1d_params['strides'], parent_output_shape[1] - kernel_size + 1)

    # Calculate the output shape
    output_length = (parent_output_shape[1] - kernel_size) // strides + 1
    output_shape = (None, output_length, filters)

    return output_shape


def is_leaf(node):
    return len(node.children) == 0


def get_parent_output(node):
    while node.parent is not None and node.parent.conv1d_layer is None:
        node = node.parent

    return node.parent.conv1d_layer.output if node.parent else None


def build_model(node, input_layer):
    if is_leaf(node):
        # Leaf node, create a Dense layer using the parent node's output shape
        parent_output_shape = calculate_conv1d_output_shape(node.parent.conv1d_params['input_shape'],
                                                             node.parent.conv1d_params['kernel_size'],
                                                             node.parent.conv1d_params['strides'],
                                                             node.parent.conv1d_params['filters'])[1:]
        units = parent_output_shape[-1]  # Use the last dimension of the parent's output shape as the number of units
        leaf_layer = layers.Dense(units, activation='relu')(input_layer[-1])
        return [leaf_layer]

    #output_shape = infer_output_shape(node)
    output_shape = calculate_conv1d_output_shape(node.conv1d_params['input_shape'],
                                                             node.conv1d_params['kernel_size'],
                                                             node.conv1d_params['strides'],
                                                             node.conv1d_params['filters'])

    # Automatically determine the kernel size based on the output shape
    kernel_size = min(node.conv1d_params['kernel_size'], output_shape[-1])

    # Use "same" padding to automatically adjust for input size changes    
    node.conv1d_layer = layers.Convolution1D(filters=output_shape[-1], kernel_size=kernel_size, activation='relu',
                                      padding='same', strides=output_shape[1], input_shape=output_shape[1:], 
                                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05))

    child_outputs = []
    child_inputs = [input_layer] if not isinstance(input_layer, list) else input_layer
    outs = []
    for child in node.children:
        
        for child_in in child_inputs:
            
            if child_in is not None:
                conv_output = node.conv1d_layer(child_in)
                child_outputs.append(conv_output)

        child_input = build_model(child, child_outputs)
        outs.append(child_input)

    # Concatenate only the outputs of leaf nodes
    # Apply the current node's Conv1D layer to the concatenated child inputs
            
    # if len(child_inputs) > 1:
    #     for child_input in child_inputs:
    #         conv_output = node.conv1d_layer(child_input)
    #     #concatenated_inputs = layers.concatenate(child_inputs, axis=-2)
    # elif len(child_inputs) == 1:
    #     concatenated_inputs = child_inputs[0]
    flattened_list = [item for sublist in outs for item in sublist]


    return flattened_list



# root_conv1d_params = {
#             'input_shape': (250, 128),
#             'kernel_size': 7,
#             'strides': 1,
#             'filters': 128
#         }

# root_node, index = dfs_traverse([0, 0, 1, 1, 1], root_conv1d_params)

# print(root_node.children)
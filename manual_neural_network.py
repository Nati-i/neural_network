"""
python manual neural network
"""

"""
Notes
    The super keyword will run the __init__ function in the super class
    Graph is a global variable
"""

import numpy as np

# Operation

class Operation():

    def __init__(self,input_nodes=[]):

        self.input_nodes = input_nodes
        self.output_nodes = []

        # For every node the operation is done
        # Append the answer that the operation results
        # into the output_nodes
        for node in input_nodes:
            node.output_nodes.append(self)

        _default_graph.operations.append(self)

    def compute(self):
        pass


class add(Operation):

    def __init__(self,x,y):
        # super keyword will run the __init__ of Operation
        super().__init__([x,y])
    
    def compute(self,x_var,y_var):
        self.inputs = [x_var,y_var]
        return x_var + y_var


class multiply(Operation):

    def __init__(self,x,y):
        # super keyword will run the __init__ of Operation
        super().__init__([x,y])
    
    def compute(self,x_var,y_var):
        self.inputs = [x_var,y_var]
        return x_var * y_var


class matmul(Operation):

    def __init__(self,x,y):
        # super keyword will run the __init__ of Operation
        super().__init__([x,y])
    
    def compute(self,x_var,y_var):
        self.inputs = [x_var,y_var]
        return x_var.dot(y_var) # Assuming it is a numpy array


# Variables, Placeholders and Graph Classes

'''
Placeholder: 
    An empty node that needs a value to be provided
Variables: 
    Changable parameter of a graph
Graph: 
    Global variable connecting variables and placeholders to operations

'''

class Placeholder():

    def __init__(self):

        self.output_nodes = []

        _default_graph.placeholders.append(self)

class Variable():

    def __init__(self,initial_value=None):

        self.value = initial_value
        self.output_nodes = []

        _default_graph.variables.append(self)

class Graph():

    def __init__(self):

        self.operations = []
        self.placeholders = []
        self.variables = []
    
    def set_as_default(self):

        global _default_graph
        _default_graph = self

# Creating a Session to execute node operations
# Using PostOrder Tree Traversal

def traverse_postorder(operation):
    """ 
    PostOrder Traversal of Nodes. Basically makes sure computations are done in 
    the correct order (Ax first , then Ax + b). Feel free to copy and paste this code.
    It is not super important for understanding the basic fundamentals of deep learning.
    """
    
    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder
## Makes sure we execute the computations in the proper order

class Session():

    def run(self,operation,feed_dict={}):
        # feed_dict maps placeholders to 

        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:

            if type(node) == Placeholder:
                node.output = feed_dict[node]
            
            elif type(node) == Variable:
                # If variable give the value to the variable
                node.output = node.value

            else:
                # Then it is an Operation
                node.inputs = [input_node.output for input_node in node.input_nodes]

                node.output = node.compute(*node.inputs)
            
            # Convert lists to np arrays
            if type(node.output) == list:
                node.output = np.array(node.output)
            
        # Return the requested node value
        return operation.output


# Classification


class Sigmoid(Operation):
    # Activation Function

    def __init__(self,z):

        super().__init__([z])

    def compute(self,z_val):
        return 1 / (1 + np.exp(-z_val))


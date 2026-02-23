import math
import numpy as np
from graphviz import Digraph

class Node:
  def __init__(self, data, _children=(), _op='', label=' '):
    self.data = data
    self._prev = set(_children)
    self._op = _op
    self.grad = 0.0
    self.label = label
    self._backward = lambda: None

  # NODE REPRESENTATION
  def __repr__(self):
    return f"Node(data={self.data}, grad={self.grad}, label={self.label})"

  # NODE OPERATIONS & LOCAL GRADIENTS
  # Node(a).__add__(Node(b)) || Node(a).__add__(int) || Node(a).__add__(float)
  def __add__(self, other):
    if isinstance(other, Node):
      out=Node(self.data + other.data, (self, other), '+')
    else:
      out=Node(self.data + other, (self, other), '+')

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward

    return out

  # Node(a).__mul__(Node(b)) || Node(a).__mul__(int) || Node(a).__mul__(float)
  def __mul__(self, other):
    if isinstance(other, Node):
      out=Node(self.data * other.data, (self, other), '*')
    else:
      out=Node(self.data * other, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward

    return out

  def __neg__(self):
    out = self * -1

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
      return other + (-self)


  # To handle cases where: int.__op__Node(s) || float.__op__Node(s)
  def __radd__(self, other):
    return self.data + other

  def __rmul__(self, other):
    return self.data * other

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Node(t, (self, ), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out

  def exp(self):
    x = self.data
    out = Node(math.exp(x), (self, ), 'exp')

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward

    return out

  def __truediv__(self, other):
    return self * other**-1

  def __pow__(self, other):
    if isinstance(other, (int, float)):
      out = Node(self.data**other, (self,), f'**{other}')
    else:
      out = Node(self.data**other.data, (self, other), '**')

    def _backward():
      if isinstance(other, (int, float)):
        self.grad += other * self.data**(other-1) * out.grad
      else:
        self.grad += other.data * self.data**(other-1) * out.grad
        other.grad += out.data * math.log(self.data) * out.grad
    out._backward = _backward

    return out

  # BACKWARD PROPAGATION
  def backward(self):
    topological = []
    visited = set()
    def build_topological(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topological(child)
        topological.append(v)
    build_topological(self)

    self.grad = 1.0
    for node in reversed(topological):
      node._backward()

  # STRUCTURE
  def draw_dot(self):

    # builds a set of all nodes and edges in a graph
    def trace(self):
      nodes, edges = set(), set()
      def build(v):
        if v not in nodes:
          nodes.add(v)
          for child in v._prev:
            edges.add((child, v))
            build(child)
      build(self)
      return nodes, edges

    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right

    nodes, edges = trace(self)
    for n in nodes:
      uid = str(id(n))
      # for any value in the graph, create a rectangular ('record') node for it
      dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
      if n._op:
        # if this value is a result of some operation, create an op node for it
        dot.node(name = uid + n._op, label = n._op)
        # and connect this node to it
        dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
      # connect n1 to the op node of n2
      dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

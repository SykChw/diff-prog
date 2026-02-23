class Neuron:
  def __init__(self, ninputs):
    self.w = [Node(random.uniform(-1,1)) for _ in range(ninputs)]
    self.b = Node(random.uniform(-1,1))

  def __call__(self, x):
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = act.tanh()
    return out

  def parameters(self):
    return self.w + [self.b]

def Layer:
  def __init__(self, ninputs, noutputs):
    self.neurons = [Neuron(ninputs) for _ in range(noutputs)]

  def __call__(self, inputs):
    outs = [n(inputs) for n in self.neurons]
    return outs

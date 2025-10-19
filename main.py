from simpleautodiff import *

Node.verbose = True

# create root nodes
x1 = Node(2)
x2 = Node(5)

# create computational graph and evaluate function value
v1 = log(x1)
v2 = mul(x1, x2)
v3 = sin(x2)
v4 = add(v1, v2)
y = sub(v4, v3)

# perform forward-mode autodiff
forward(x1)

# perform reverse-mode autodiff
grads = reverse(y, trace=True)
print("=== reverse gradients ===")
for node, grad in grads.items():
    print(f"d{y.name}/d{node.name} = {grad:.6f}")

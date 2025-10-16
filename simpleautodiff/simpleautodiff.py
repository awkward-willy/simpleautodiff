from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from math import log as math_log
from math import sin as math_sin
from math import cos as math_cos


class Node:
    verbose = False
    input_count = 0
    intermediate_count = 0

    def __init__(
        self,
        value,
        parent_nodes: Optional[Sequence["Node"]] = None,
        operator="input",
    ):
        if parent_nodes is None:
            parent_nodes = []

        self.value = value
        self.parent_nodes: List[Node] = list(parent_nodes)
        self.child_nodes = []
        self.operator = operator
        self.grad_wrt_parents = []
        self.partial_derivative = 0.0
        self.adjoint = 0.0

        if self.operator == "input":
            Node.input_count += 1
            self.name = "x%d" % (Node.input_count)
        else:
            Node.intermediate_count += 1
            self.name = "v%d" % (Node.intermediate_count)

        if Node.verbose:
            print(
                "{:<2} = {:<18} = {:<8}".format(
                    self.name,
                    self.operator + str([p.name for p in self.parent_nodes]),
                    self.value.__round__(3),
                )
            )

    def reset(self) -> None:
        """Clear cached derivatives prior to a new autodiff pass."""
        self.partial_derivative = 0.0
        self.adjoint = 0.0

    @property
    def is_input(self) -> bool:
        return self.operator == "input"

    def __repr__(self) -> str:
        return f"Node(name={self.name!r}, value={self.value!r}, operator={self.operator!r})"


def add(node1: Node, node2: Node) -> Node:
    value = node1.value + node2.value
    parent_nodes = [node1, node2]
    newNode = Node(value, parent_nodes, "add")
    newNode.grad_wrt_parents = [1, 1]
    node1.child_nodes.append(newNode)
    node2.child_nodes.append(newNode)
    return newNode


def sub(node1, node2):
    value = node1.value - node2.value
    parent_nodes = [node1, node2]
    newNode = Node(value, parent_nodes, "sub")
    newNode.grad_wrt_parents = [1, -1]
    node1.child_nodes.append(newNode)
    node2.child_nodes.append(newNode)
    return newNode


def mul(node1, node2):
    value = node1.value * node2.value
    parent_nodes = [node1, node2]
    newNode = Node(value, parent_nodes, "mul")
    newNode.grad_wrt_parents = [node2.value, node1.value]
    node1.child_nodes.append(newNode)
    node2.child_nodes.append(newNode)
    return newNode


def log(node):
    value = math_log(node.value)
    parent_nodes = [node]
    newNode = Node(value, parent_nodes, "log")
    newNode.grad_wrt_parents = [1 / (node.value)]
    node.child_nodes.append(newNode)
    return newNode


def sin(node):
    value = math_sin(node.value)
    parent_nodes = [node]
    newNode = Node(value, parent_nodes, "sin")
    newNode.grad_wrt_parents = [math_cos(node.value)]
    node.child_nodes.append(newNode)
    return newNode


def topological_order(rootNode):
    """Return nodes reachable from ``rootNode`` ordered for forward accumulation."""

    def add_children(node):
        if node not in visited:
            visited.add(node)
            for child in node.child_nodes:
                add_children(child)
            ordering.append(node)

    ordering, visited = [], set()
    add_children(rootNode)
    return list(reversed(ordering))


def _reverse_topological_order(output_node: Node) -> List[Node]:
    ordering: List[Node] = []
    visited = set()

    def visit(node: Node) -> None:
        if node in visited:
            return
        visited.add(node)
        for parent in node.parent_nodes:
            visit(parent)
        ordering.append(node)

    visit(output_node)
    return list(reversed(ordering))


def _reset_nodes(nodes: Iterable[Node]) -> None:
    for node in nodes:
        node.reset()


def forward(rootNode: Node, *, trace: bool = True):
    """Perform forward-mode autodiff starting from ``rootNode``.

    Parameters
    ----------
    rootNode:
        The input node with respect to which partial derivatives are computed.
    trace:
        When ``True`` (default) prints the intermediate steps as in the
        original educational example.

    Returns
    -------
    dict
        Mapping of each visited node to its partial derivative with respect to
        ``rootNode``.
    """

    ordering = topological_order(rootNode)
    _reset_nodes(ordering)
    rootNode.partial_derivative = 1.0

    if trace:
        print("=== ordering ===")
        print([obj.name for obj in ordering])

    for node in ordering[1:]:
        if len(node.grad_wrt_parents) != len(node.parent_nodes):
            raise ValueError(
                f"Node {node.name} expects {len(node.parent_nodes)} local gradients "
                f"but received {len(node.grad_wrt_parents)}"
            )

        partial_derivative = 0.0
        if trace:
            print("=== node: {} ===".format(node.name))

        for parent, local_grad in zip(node.parent_nodes, node.grad_wrt_parents):
            dparent_droot = parent.partial_derivative
            contribution = local_grad * dparent_droot
            partial_derivative += contribution

            if trace:
                print(
                    "  d{} / d{} = {:<8} * {:<8} = {:<8}".format(
                        node.name,
                        parent.name,
                        local_grad.__round__(3),
                        dparent_droot.__round__(3),
                        contribution.__round__(3),
                    )
                )

        node.partial_derivative = partial_derivative

        if Node.verbose and trace:
            symbol_process = ""
            value_process = ""
            for parent, local_grad in zip(node.parent_nodes, node.grad_wrt_parents):
                symbol_process += (
                    "(d"
                    + node.name
                    + "/d"
                    + parent.name
                    + ")"
                    + "(d"
                    + parent.name
                    + "/d"
                    + rootNode.name
                    + ") + "
                )
                value_process += (
                    "("
                    + str(local_grad.__round__(3))
                    + ")("
                    + str(parent.partial_derivative.__round__(3))
                    + ") + "
                )
            print(
                "d{:<2}/d{:<2} = {:<45} \n\t= {:<30} = {:<5}".format(
                    node.name,
                    rootNode.name,
                    symbol_process.strip(" + "),
                    value_process.strip(" + "),
                    str(node.partial_derivative.__round__(3)),
                )
            )

    return {node: node.partial_derivative for node in ordering}


def reverse(
    output_node: Node,
    *,
    trace: bool = False,
    target_nodes: Optional[Sequence[Node]] = None,
):
    """Perform reverse-mode autodiff starting from a scalar ``output_node``.

    Parameters
    ----------
    output_node:
        The terminal node representing the scalar function value whose
        gradient is desired.
    trace:
        When ``True`` prints intermediate accumulation steps for educational
        purposes.
    target_nodes:
        Optional iterable of nodes for which gradients should be returned. If
        omitted the gradients for all input nodes are provided.

    Returns
    -------
    dict
        Mapping of nodes to the gradient of ``output_node`` with respect to
        each node.
    """

    ordering = _reverse_topological_order(output_node)
    _reset_nodes(ordering)
    output_node.adjoint = 1.0

    if trace:
        print("=== reverse ordering ===")
        print([node.name for node in ordering])

    for node in ordering:
        if trace:
            print(f"=== node: {node.name} ===")

        for parent, local_grad in zip(node.parent_nodes, node.grad_wrt_parents):
            contribution = node.adjoint * local_grad
            parent.adjoint += contribution

            if trace:
                print(
                    "  d{} / d{} += {:<8} * {:<8} = {:<8}".format(
                        output_node.name,
                        parent.name,
                        node.adjoint.__round__(3),
                        local_grad.__round__(3),
                        parent.adjoint.__round__(3),
                    )
                )

    if target_nodes is None:
        target_nodes = [node for node in ordering if node.is_input]

    gradients = {}
    for node in target_nodes:
        if node not in ordering:
            raise ValueError(f"Node {node} is not connected to output {output_node}")
        gradients[node] = node.adjoint

    return gradients

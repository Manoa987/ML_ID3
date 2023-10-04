from __future__ import annotations
from typing import Dict
from attribute import Attribute


class Node:

    def __init__(self,attribute: Attribute,value: int) -> None:
        self.children : Dict[str, Node] = {}
        self.value = value
        self.attribute = attribute

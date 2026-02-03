"""
b_tree.py

B-tree with:
- create (constructor)
- insert
- update
- delete
- search / traverse
- NEW: dump_structure() to visualize the tree shape (levels, nodes, keys) after each operation

`max_keys` must be an odd integer >= 3. Internally we use minimum degree t = (max_keys + 1)//2,
so max keys per node is (2*t - 1) == max_keys.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Iterable


class BTreeNode:
    """
    A node in a B-tree.

    keys[i] has associated values[i]
    children count is:
        - 0 if leaf
        - len(keys)+1 if internal
    """

    def __init__(self, t: int, leaf: bool):
        self.t = t
        self.leaf = leaf
        self.keys: List[Any] = []
        self.values: List[Any] = []
        self.children: List["BTreeNode"] = []

    # -----------------------------
    # Core search / traverse
    # -----------------------------

    def search(self, key) -> Optional[Tuple["BTreeNode", int]]:
        i = 0
        while i < len(self.keys) and key > self.keys[i]:
            i += 1

        if i < len(self.keys) and self.keys[i] == key:
            return (self, i)

        if self.leaf:
            return None

        return self.children[i].search(key)

    def traverse(self) -> List[Tuple[Any, Any]]:
        result: List[Tuple[Any, Any]] = []
        for i in range(len(self.keys)):
            if not self.leaf:
                result.extend(self.children[i].traverse())
            result.append((self.keys[i], self.values[i]))
        if not self.leaf:
            result.extend(self.children[len(self.keys)].traverse())
        return result

    # -----------------------------
    # Insert
    # -----------------------------

    def insert_non_full(self, key, value) -> None:
        i = len(self.keys) - 1

        if self.leaf:
            while i >= 0 and self.keys[i] > key:
                i -= 1

            # If key already exists in this leaf, update value
            if i >= 0 and self.keys[i] == key:
                self.values[i] = value
                return

            self.keys.insert(i + 1, key)
            self.values.insert(i + 1, value)
            return

        # Internal node: descend into the right child
        while i >= 0 and self.keys[i] > key:
            i -= 1
        i += 1

        # If that child is full, split it
        if len(self.children[i].keys) == 2 * self.t - 1:
            self.split_child(i)

            # After split, decide which of the two children to descend into
            if self.keys[i] < key:
                i += 1

        self.children[i].insert_non_full(key, value)

    def split_child(self, i: int) -> None:
        """
        Split full child children[i] into two nodes and push median key up into this node.
        """
        t = self.t
        y = self.children[i]
        z = BTreeNode(t, y.leaf)

        median_key = y.keys[t - 1]
        median_val = y.values[t - 1]

        # z gets y's keys after the median
        z.keys = y.keys[t:]
        z.values = y.values[t:]

        # If internal, also split children
        if not y.leaf:
            z.children = y.children[t:]

        # y keeps keys before the median
        y.keys = y.keys[: t - 1]
        y.values = y.values[: t - 1]
        if not y.leaf:
            y.children = y.children[:t]

        # Insert new child z after y
        self.children.insert(i + 1, z)
        # Insert median into this node
        self.keys.insert(i, median_key)
        self.values.insert(i, median_val)

    # -----------------------------
    # Delete
    # -----------------------------

    def remove(self, key) -> None:
        idx = self.find_key(key)

        # Case 1: key is present in this node
        if idx < len(self.keys) and self.keys[idx] == key:
            if self.leaf:
                self.remove_from_leaf(idx)
            else:
                self.remove_from_non_leaf(idx)
            return

        # Case 2: key is not present in this node
        if self.leaf:
            raise KeyError(f"Key {key} not found in the tree.")

        # Determine whether key is expected in the last child
        flag = (idx == len(self.keys))

        # Ensure child[idx] has at least t keys before descending
        if len(self.children[idx].keys) < self.t:
            self.fill(idx)

        # If we merged the last child with previous, idx can now be out of range
        if flag and idx > len(self.keys):
            self.children[idx - 1].remove(key)
        else:
            self.children[idx].remove(key)

    def find_key(self, key) -> int:
        idx = 0
        while idx < len(self.keys) and self.keys[idx] < key:
            idx += 1
        return idx

    def remove_from_leaf(self, idx: int) -> None:
        self.keys.pop(idx)
        self.values.pop(idx)

    def remove_from_non_leaf(self, idx: int) -> None:
        """
        Remove self.keys[idx] from an internal node.

        Strategy:
        - If left child has >= t keys: swap with predecessor and delete predecessor in left child.
        - Else if right child has >= t keys: swap with successor and delete successor in right child.
        - Else: merge left+key+right, then delete key from merged node.
        """
        key = self.keys[idx]

        if len(self.children[idx].keys) >= self.t:
            pred_key, pred_val = self.get_predecessor(idx)
            self.keys[idx] = pred_key
            self.values[idx] = pred_val
            self.children[idx].remove(pred_key)
        elif len(self.children[idx + 1].keys) >= self.t:
            succ_key, succ_val = self.get_successor(idx)
            self.keys[idx] = succ_key
            self.values[idx] = succ_val
            self.children[idx + 1].remove(succ_key)
        else:
            self.merge(idx)
            self.children[idx].remove(key)

    def get_predecessor(self, idx: int) -> Tuple[Any, Any]:
        cur = self.children[idx]
        while not cur.leaf:
            cur = cur.children[-1]
        return (cur.keys[-1], cur.values[-1])

    def get_successor(self, idx: int) -> Tuple[Any, Any]:
        cur = self.children[idx + 1]
        while not cur.leaf:
            cur = cur.children[0]
        return (cur.keys[0], cur.values[0])

    def fill(self, idx: int) -> None:
        """
        Ensure that children[idx] has at least t keys by:
        - borrowing from left sibling, or
        - borrowing from right sibling, or
        - merging with a sibling.
        """
        if idx != 0 and len(self.children[idx - 1].keys) >= self.t:
            self.borrow_from_prev(idx)
        elif idx != len(self.keys) and len(self.children[idx + 1].keys) >= self.t:
            self.borrow_from_next(idx)
        else:
            if idx != len(self.keys):
                self.merge(idx)
            else:
                self.merge(idx - 1)

    def borrow_from_prev(self, idx: int) -> None:
        child = self.children[idx]
        sibling = self.children[idx - 1]

        # Bring separator key down into child (front)
        child.keys.insert(0, self.keys[idx - 1])
        child.values.insert(0, self.values[idx - 1])

        # If internal, move sibling's last child pointer
        if not child.leaf:
            child.children.insert(0, sibling.children.pop())

        # Move sibling's last key up to separator position
        self.keys[idx - 1] = sibling.keys.pop()
        self.values[idx - 1] = sibling.values.pop()

    def borrow_from_next(self, idx: int) -> None:
        child = self.children[idx]
        sibling = self.children[idx + 1]

        # Bring separator key down into child (end)
        child.keys.append(self.keys[idx])
        child.values.append(self.values[idx])

        # If internal, move sibling's first child pointer
        if not child.leaf:
            child.children.append(sibling.children.pop(0))

        # Move sibling's first key up to separator position
        self.keys[idx] = sibling.keys.pop(0)
        self.values[idx] = sibling.values.pop(0)

    def merge(self, idx: int) -> None:
        """
        Merge children[idx] + separator key[idx] + children[idx+1] into children[idx].
        """
        child = self.children[idx]
        sibling = self.children[idx + 1]

        # Pull separator key down
        child.keys.append(self.keys[idx])
        child.values.append(self.values[idx])

        # Append sibling contents
        child.keys.extend(sibling.keys)
        child.values.extend(sibling.values)

        if not child.leaf:
            child.children.extend(sibling.children)

        # Remove separator key and sibling pointer from this node
        self.keys.pop(idx)
        self.values.pop(idx)
        self.children.pop(idx + 1)

    def __str__(self) -> str:
        return f"Keys={self.keys} Leaf={self.leaf}"


class BTree:
    """
    B-tree with odd max_keys >= 3.

    Public operations:
      - insert(key, value)
      - update(key, value)
      - delete(key)
      - search(key)
      - traverse()
      - dump_structure()  <-- NEW
    """

    def __init__(self, max_keys: int):
        if max_keys < 3 or max_keys % 2 == 0:
            raise ValueError("max_keys must be an odd number >= 3")

        self.max_keys = max_keys
        self.t = (max_keys + 1) // 2  # minimum degree
        self.root = BTreeNode(self.t, leaf=True)

    def search(self, key) -> Optional[Tuple[BTreeNode, int]]:
        return self.root.search(key)

    def traverse(self) -> List[Tuple[Any, Any]]:
        return self.root.traverse()

    def insert(self, key, value) -> None:
        r = self.root
        if len(r.keys) == 2 * self.t - 1:
            s = BTreeNode(self.t, leaf=False)
            s.children.append(r)
            s.split_child(0)
            self.root = s
            s.insert_non_full(key, value)
        else:
            r.insert_non_full(key, value)

    def delete(self, key) -> None:
        self.root.remove(key)

        # If root becomes empty, shrink height
        if len(self.root.keys) == 0:
            if not self.root.leaf:
                self.root = self.root.children[0]
            else:
                self.root = BTreeNode(self.t, leaf=True)

    def update(self, key, value) -> None:
        found = self.search(key)
        if not found:
            raise KeyError(f"Key {key} not found; cannot update.")
        node, idx = found
        node.values[idx] = value

    # ------------------------------------------------------------------
    # NEW: Structure dump
    # ------------------------------------------------------------------

    def dump_structure(self, show_values: bool = False) -> str:
        """
        Return a multi-line string showing the B-tree structure level-by-level.

        Example output:

            BTree(max_keys=3, t=2)
            Level 0: [10 | 20]
            Level 1: [5 | 6 | 7]   [12 | 17]   [30]

        Notes:
        - This is a "shape visualization", not a full graph dump.
        - `show_values=True` will display key:value pairs (useful when values differ).
        """
        lines: List[str] = []
        lines.append(f"BTree(max_keys={self.max_keys}, t={self.t})")

        # Breadth-first traversal
        queue: List[Tuple[BTreeNode, int]] = [(self.root, 0)]
        current_level = 0
        level_nodes: List[BTreeNode] = []

        def fmt_node(n: BTreeNode) -> str:
            if show_values:
                items = [f"{k}:{v}" for k, v in zip(n.keys, n.values)]
            else:
                items = [str(k) for k in n.keys]
            inside = " | ".join(items)
            return f"[{inside}]"

        while queue:
            node, lvl = queue.pop(0)
            if lvl != current_level:
                # Flush previous level
                lines.append(
                    f"Level {current_level}: " + "   ".join(fmt_node(n) for n in level_nodes)
                )
                level_nodes = []
                current_level = lvl

            level_nodes.append(node)

            if not node.leaf:
                for child in node.children:
                    queue.append((child, lvl + 1))

        # Flush last level
        lines.append(
            f"Level {current_level}: " + "   ".join(fmt_node(n) for n in level_nodes)
        )

        return "\n".join(lines)

    def __str__(self) -> str:
        return str(self.traverse())


# ------------------------------------------------------------
# Demo / Test harness
# ------------------------------------------------------------
if __name__ == "__main__":
    # Choose odd max_keys: 3 gives classic minimum degree t=2 (2-3-4 tree)
    btree = BTree(max_keys=3)

    def do(op_name: str, fn):
        print("\n" + "=" * 70)
        print(op_name)
        fn()
        print(btree.dump_structure())
        print("In-order:", btree.traverse())

    # Insert sequence designed to trigger splits
    inserts = [(10, 'a'), (20, 'b'), (5, 'c'), (6, 'd'),
               (12, 'e'), (30, 'f'), (7, 'g'), (17, 'h')]

    for k, v in inserts:
        do(f"INSERT {k}={v}", lambda k=k, v=v: btree.insert(k, v))

    # Update (should not change shape)
    do("UPDATE 7='G'", lambda: btree.update(7, 'G'))

    # Deletes designed to trigger merges/borrows
    for k in [6, 7, 5, 10, 12]:
        do(f"DELETE {k}", lambda k=k: btree.delete(k))

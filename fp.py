import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations

import math

# ===================== FP-Growth Logic with Tree Output =====================
class FPNode:
    def __init__(self, item_name, count, parent):
        self.item_name = item_name
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None

    def increment(self, count):
        self.count += count

class FPTree:
    def __init__(self, transactions, min_support):
        self.root = FPNode(None, 1, None)
        self.header_table = {}
        self.min_support = min_support
        self.build_header_table(transactions)
        self.build_fp_tree(transactions)

    def build_header_table(self, transactions):
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        self.header_table = {item: [count, None] for item, count in item_counts.items() if count >= self.min_support}

    def update_header(self, node):
        item = node.item_name
        if self.header_table[item][1] is None:
            self.header_table[item][1] = node
        else:
            current = self.header_table[item][1]
            while current.link is not None:
                current = current.link
            current.link = node

    def insert_transaction(self, transaction):
        transaction = [item for item in transaction if item in self.header_table]
        transaction.sort(key=lambda item: (-self.header_table[item][0], item))
        current_node = self.root
        for item in transaction:
            if item in current_node.children:
                current_node.children[item].increment(1)
            else:
                new_node = FPNode(item, 1, current_node)
                current_node.children[item] = new_node
                self.update_header(new_node)
            current_node = current_node.children[item]

    def build_fp_tree(self, transactions):
        for transaction in transactions:
            self.insert_transaction(transaction)

    def tree_to_string(self, node):
        result = []
        for child in node.children.values():
            subtree = self.tree_to_string(child)
            result.append(f"<{child.item_name}:{child.count}" + (", " + subtree if subtree else "") + ">")
        return ", ".join(result)

    def build_conditional_tree(self, conditional_patterns):
        transactions = []
        for path, count in conditional_patterns:
            transactions.extend([path] * count)
        if not transactions:
            return None
        return FPTree(transactions, self.min_support)

    def mine_patterns(self):
        patterns = {}
        conditional_pattern_bases = {}
        conditional_fp_trees = {}
        frequent_patterns = {}

        items = sorted(self.header_table.items(), key=lambda x: x[1][0])
        for item, (support, node) in items:
            conditional_patterns = []
            while node is not None:
                path = []
                parent = node.parent
                while parent and parent.item_name is not None:
                    path.append(parent.item_name)
                    parent = parent.parent
                if path:
                    conditional_patterns.append((path[::-1], node.count))
                node = node.link

            conditional_pattern_bases[item] = conditional_patterns
            conditional_tree = self.build_conditional_tree(conditional_patterns)
            conditional_fp_trees[item] = self.tree_to_string(conditional_tree.root) if conditional_tree else ""

            if conditional_tree:
                _, _, subtree_patterns = conditional_tree.mine_patterns()
                for pattern, count in subtree_patterns.items():
                    full_pattern = tuple(list(pattern) + [item])
                    patterns[full_pattern] = count
                    frequent_patterns[full_pattern] = count

            patterns[(item,)] = support
            frequent_patterns[(item,)] = support

        return conditional_pattern_bases, conditional_fp_trees, frequent_patterns

# ===================== Streamlit App =====================
st.set_page_config(page_title="FP-Growth Frequent Itemsets", layout="wide")
st.title("FP-Growth Frequent Itemsets Mining")

st.markdown("""
Upload a *Transaction CSV* and discover *Frequent Itemsets* and *Conditional FP-Trees* using the **FP-Growth** algorithm.
""")

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(df.head())

    column_options = df.columns.tolist()
    item_col = st.selectbox("Select the column containing items per transaction:", column_options)

    transactions = df[item_col].dropna().astype(str).apply(lambda x: x.split(',')).tolist()

    min_support_percent = st.number_input("Enter Minimum Support (%)", min_value=1, max_value=100, value=5)

    total_transactions = len(transactions)
    min_support_count = math.ceil((min_support_percent / 100) * total_transactions)

    if st.button("Run FP-Growth"):
        tree = FPTree(transactions, min_support_count)
        conditional_pattern_bases, conditional_fp_trees, frequent_patterns = tree.mine_patterns()

        table = []
        for item in sorted(conditional_pattern_bases.keys()):
            cpb = ', '.join([f"{p}:{c}" for p, c in conditional_pattern_bases[item]])
            tree_str = conditional_fp_trees[item]
            freq_patterns = [str(p) + f":{c}" for p, c in frequent_patterns.items() if item in p and p[-1] == item]
            freq_str = ', '.join(freq_patterns)
            table.append([item, cpb, tree_str, freq_str])

        headers = ["Item", "Conditional Pattern Base", "Conditional FP-Tree", "Frequent Pattern Generated"]
        result_df = pd.DataFrame(table, columns=headers)

        st.success(f"Mined Frequent Itemsets with FP-Growth!")
        st.write("### FP-Growth Conditional Pattern Table")
        st.dataframe(result_df)

        st.download_button("Download Frequent Itemsets Table", result_df.to_csv(index=False), file_name="fp_growth_table.csv")

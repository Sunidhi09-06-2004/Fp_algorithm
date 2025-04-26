import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
import math

# ===================== FP-Growth Code =====================
class FPNode:
    id_counter = 0

    def __init__(self, item_name, count, parent):
        self.item_name = item_name
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None
        self.uid = FPNode.id_counter
        FPNode.id_counter += 1

    def increment(self, count):
        self.count += count

def build_header_table(transactions, min_support_count):
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1
    return {item: [cnt, None] for item, cnt in item_counts.items() if cnt >= min_support_count}

def sort_items(transaction, header_table):
    return sorted([item for item in transaction if item in header_table],
                  key=lambda item: header_table[item][0], reverse=True)

def insert_tree(transaction, root, header_table, count):
    current_node = root
    for item in transaction:
        if item not in current_node.children:
            new_node = FPNode(item, count, current_node)
            current_node.children[item] = new_node
            if header_table[item][1] is None:
                header_table[item][1] = new_node
            else:
                node = header_table[item][1]
                while node.link:
                    node = node.link
                node.link = new_node
        else:
            current_node.children[item].increment(count)
        current_node = current_node.children[item]

def construct_fp_tree(transactions, min_support_count):
    header_table = build_header_table(transactions, min_support_count)
    if len(header_table) == 0:
        return None, None
    root = FPNode(None, 1, None)
    for transaction in transactions:
        sorted_items = sort_items(transaction, header_table)
        if sorted_items:
            insert_tree(sorted_items, root, header_table, 1)  # Each transaction counted once
    return root, header_table

def ascend_fp_tree(node):
    path = []
    while node.parent and node.parent.item_name is not None:
        node = node.parent
        path.append(node.item_name)
    return path[::-1]

def find_prefix_paths(base_pattern, node):
    conditional_patterns = []
    while node:
        path = ascend_fp_tree(node)
        if path:
            conditional_patterns.append((path, node.count))
        node = node.link
    return conditional_patterns

def build_conditional_pattern_base(header_table, item):
    node = header_table[item][1]
    patterns = []
    while node:
        path = ascend_fp_tree(node)
        if path:
            patterns.append((path, node.count))
        node = node.link
    return patterns

def construct_conditional_fp_tree(conditional_patterns, min_support_count):
    transactions = []
    for path, count in conditional_patterns:
        for _ in range(count):
            transactions.append(path)
    return construct_fp_tree(transactions, min_support_count)

def mine_fp_tree(header_table, prefix, frequent_itemsets, min_support_count):
    sorted_items = sorted(header_table.items(), key=lambda x: x[1][0])
    for base_item, (count, node) in sorted_items:
        new_freq_set = prefix.copy()
        new_freq_set.add(base_item)
        frequent_itemsets.append((new_freq_set, count))
        conditional_patterns = find_prefix_paths(base_item, node)
        conditional_tree, new_header_table = construct_conditional_fp_tree(conditional_patterns, min_support_count)
        if new_header_table:
            mine_fp_tree(new_header_table, new_freq_set, frequent_itemsets, min_support_count)

# ===================== Association Rules =====================
def generate_association_rules(frequent_itemsets, min_confidence_percent):
    min_confidence = min_confidence_percent / 100.0
    rules = []
    itemset_support = {frozenset(itemset): support for itemset, support in frequent_itemsets}
    for itemset, support in frequent_itemsets:
        if len(itemset) > 1:
            items = list(itemset)
            for i in range(1, len(items)):
                for antecedent in combinations(items, i):
                    antecedent = frozenset(antecedent)
                    consequent = frozenset(itemset) - antecedent
                    if consequent and antecedent in itemset_support:
                        confidence = itemset_support[frozenset(itemset)] / itemset_support[antecedent]
                        if confidence >= min_confidence:
                            rules.append((set(antecedent), set(consequent), round(confidence*100, 2), support))
    return rules

# ===================== Streamlit App =====================
st.set_page_config(page_title="FP-Growth Frequent Itemsets", layout="wide")
st.title("FP-Growth Frequent Itemsets Mining")

st.markdown("""
Upload a *Transaction CSV* and discover *Frequent Itemsets* and *Association Rules*  
using the **FP-Growth** algorithm.
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
    min_confidence_percent = st.number_input("Enter Minimum Confidence (%)", min_value=1, max_value=100, value=60)

    total_transactions = len(transactions)
    min_support_count = math.ceil((min_support_percent / 100) * total_transactions)

    if st.button("Run FP-Growth"):
        root, header_table = construct_fp_tree(transactions, min_support_count)

        if root is None:
            st.warning("No frequent items found with the given support.")
        else:
            frequent_itemsets = []
            mine_fp_tree(header_table, set(), frequent_itemsets, min_support_count)

            result_df = pd.DataFrame([(list(items), support) for items, support in frequent_itemsets],
                                     columns=["Itemset", "Support"])

            if result_df.empty:
                st.warning("No frequent itemsets found.")
            else:
                st.success(f"Found {len(result_df)} frequent itemsets!")
                st.write("### Frequent Itemsets")
                st.dataframe(result_df)

                item_count = defaultdict(int)
                for items, support in frequent_itemsets:
                    for item in items:
                        item_count[item] += support

                top_items = sorted(item_count.items(), key=lambda x: x[1], reverse=True)[:10]
                if top_items:
                    item_names = [x[0] for x in top_items]
                    item_support = [x[1] for x in top_items]

                    fig, ax = plt.subplots()
                    ax.bar(item_names, item_support, color='skyblue')
                    plt.xticks(rotation=45)
                    plt.title("Top Frequent Items")
                    st.pyplot(fig)

            rules = generate_association_rules(frequent_itemsets, min_confidence_percent)

            if rules:
                rules_df = pd.DataFrame([(
                    list(antecedent), list(consequent), confidence, support
                ) for antecedent, consequent, confidence, support in rules],
                columns=["Antecedent", "Consequent", "Confidence (%)", "Support"])

                st.write("### Association Rules")
                st.dataframe(rules_df)

                st.download_button("Download Frequent Itemsets", result_df.to_csv(index=False), file_name="frequent_itemsets.csv")
                st.download_button("Download Association Rules", rules_df.to_csv(index=False), file_name="association_rules.csv")
            else:
                st.info("No association rules found with the given confidence.")

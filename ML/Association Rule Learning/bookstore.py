"""
ASSOCIATION RULE LEARNING - APRIORI ALGORITHM - BOOKSTORE SCENARIO
===================================================================
Discovering which book genres are frequently purchased together

Perfect Scenario for Apriori:
- A small bookstore wants to understand customer buying patterns
- Customers often buy multiple books in one visit
- We want to find hidden associations between genres/titles
- Use results to create combo deals, improve shelf placement, recommendations

Dataset: Bookstore Daily Transactions (Generated)
Items (Book Genres/Titles):
- Fiction
- Self-Help
- Science
- History
- Children
- Cooking
- Biography
- Travel

Why Apriori:
- Finds frequent itemsets from transaction data
- Generates Association Rules: IF {Fiction} THEN {Self-Help}
- Supports business decisions: cross-sells, bundle offers, shelf layout

Apriori Concepts:
- Support    = How often itemset appears in all transactions
- Confidence = How often rule is correct
- Lift       = How much better than random chance

Approach:
1. Generate realistic bookstore transactions
2. Calculate item frequencies
3. Find frequent 1-itemsets (C1 → L1)
4. Find frequent 2-itemsets (C2 → L2)
5. Find frequent 3-itemsets (C3 → L3)
6. Generate Association Rules
7. Filter by Confidence and Lift
8. Visualize and Interpret
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("ASSOCIATION RULE LEARNING - APRIORI ALGORITHM")
print("SCENARIO: BOOKSTORE PURCHASE PATTERN DISCOVERY")
print("=" * 80)


# ============================================================================
# STEP 1: GENERATE REALISTIC BOOKSTORE TRANSACTION DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: GENERATE REALISTIC BOOKSTORE TRANSACTION DATA")
print("=" * 80)

np.random.seed(42)

ITEMS = ['Fiction', 'Self-Help', 'Science', 'History',
         'Children', 'Cooking', 'Biography', 'Travel']

n_transactions = 200

print(f"\nGenerating {n_transactions} bookstore transactions...")
print(f"Available book genres: {ITEMS}")

# Define realistic buying probabilities and co-occurrence tendencies
# These simulate real-world patterns:
#   - Fiction buyers often also buy Self-Help or Biography
#   - Science + History often bought together (academic readers)
#   - Cooking + Travel (lifestyle buyers)
#   - Children books often bought alone or with Fiction

transaction_templates = [
    # (template, weight) - weight = how often this pattern occurs
    (['Fiction', 'Self-Help'],          10),
    (['Fiction', 'Biography'],           8),
    (['Fiction', 'Self-Help', 'Biography'], 7),
    (['Science', 'History'],            9),
    (['Science', 'History', 'Biography'], 6),
    (['Cooking', 'Travel'],             9),
    (['Cooking', 'Self-Help'],          6),
    (['Cooking', 'Travel', 'Self-Help'], 5),
    (['Children', 'Fiction'],           7),
    (['Children'],                      5),
    (['Self-Help', 'Biography'],        7),
    (['History', 'Biography'],          6),
    (['Fiction'],                       5),
    (['Science'],                       4),
    (['Travel'],                        4),
    (['Fiction', 'Science'],            4),
    (['Self-Help'],                     4),
    (['Cooking'],                       3),
    (['History', 'Travel'],             4),
    (['Fiction', 'Self-Help', 'Cooking'], 3),
]

templates, weights = zip(*transaction_templates)
weights = np.array(weights, dtype=float)
weights /= weights.sum()

transactions = []
for _ in range(n_transactions):
    idx = np.random.choice(len(templates), p=weights)
    base = list(templates[idx])
    # Occasionally add a random extra item (10% chance)
    if np.random.rand() < 0.10:
        extras = [i for i in ITEMS if i not in base]
        if extras:
            base.append(np.random.choice(extras))
    transactions.append(sorted(set(base)))

print(f"\n✓ {n_transactions} transactions generated successfully!")

# Display first 15 transactions
print("\n--- Sample Transactions (First 15) ---")
print(f"{'TID':<6} {'Items Purchased'}")
print("-" * 50)
for i, t in enumerate(transactions[:15], 1):
    print(f"T{i:<5} {', '.join(t)}")

# Transaction length distribution
lengths = [len(t) for t in transactions]
print(f"\n--- Transaction Size Distribution ---")
from collections import Counter
size_counts = Counter(lengths)
for size in sorted(size_counts):
    bar = "█" * size_counts[size]
    print(f"  Size {size}: {size_counts[size]:>3} transactions  {bar}")

print(f"\n  Average items per transaction: {np.mean(lengths):.2f}")
print(f"  Max items in one transaction:  {max(lengths)}")
print(f"  Min items in one transaction:  {min(lengths)}")


# ============================================================================
# STEP 2: ITEM FREQUENCY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: ITEM FREQUENCY ANALYSIS")
print("=" * 80)

print("\nExplanation:")
print("  Before running Apriori, we analyze how often each item appears.")
print("  Support = (Transactions containing item) / (Total transactions)")

item_counts = defaultdict(int)
for t in transactions:
    for item in t:
        item_counts[item] += 1

item_freq_df = pd.DataFrame({
    'Item': list(item_counts.keys()),
    'Count': list(item_counts.values())
})
item_freq_df['Support'] = item_freq_df['Count'] / n_transactions
item_freq_df = item_freq_df.sort_values('Support', ascending=False).reset_index(drop=True)

print("\n--- Individual Item Frequencies ---")
print(f"{'Rank':<6} {'Item':<12} {'Count':<8} {'Support':<10} {'Bar'}")
print("-" * 60)
for _, row in item_freq_df.iterrows():
    bar = "█" * int(row['Support'] * 40)
    print(f"  {_+1:<4} {row['Item']:<12} {row['Count']:<8} {row['Support']:.4f}    {bar}")

print(f"\nMost popular genre: {item_freq_df.iloc[0]['Item']} "
      f"(appears in {item_freq_df.iloc[0]['Count']} transactions, "
      f"support = {item_freq_df.iloc[0]['Support']:.2%})")


# ============================================================================
# STEP 3: APRIORI ALGORITHM - CORE IMPLEMENTATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: APRIORI ALGORITHM - CORE IMPLEMENTATION")
print("=" * 80)

MIN_SUPPORT = 0.10   # 10% minimum support
MIN_CONFIDENCE = 0.40  # 40% minimum confidence
MIN_LIFT = 1.0       # Lift > 1 means positive association

print(f"\nApriori Parameters:")
print(f"  Minimum Support    : {MIN_SUPPORT:.0%}  (itemset must appear in ≥{int(MIN_SUPPORT*n_transactions)} transactions)")
print(f"  Minimum Confidence : {MIN_CONFIDENCE:.0%}  (rule must be correct ≥{MIN_CONFIDENCE:.0%} of the time)")
print(f"  Minimum Lift       : {MIN_LIFT:.1f}   (rule must be better than random chance)")

print(f"""
How Apriori Works:
  Step A: Find all frequent 1-itemsets (L1) with support ≥ {MIN_SUPPORT:.0%}
  Step B: Generate candidate 2-itemsets (C2) from L1 pairs
  Step C: Find frequent 2-itemsets (L2) from C2
  Step D: Generate candidate 3-itemsets (C3) from L2 triples
  Step E: Find frequent 3-itemsets (L3) from C3
  Step F: Generate Association Rules from all frequent itemsets
  Step G: Filter rules by Confidence ≥ {MIN_CONFIDENCE:.0%} and Lift ≥ {MIN_LIFT:.1f}

Apriori Property (Anti-Monotone):
  "If an itemset is infrequent, ALL its supersets are also infrequent"
  This PRUNES the search space dramatically!
""")

# Helper functions
def get_support(itemset, transactions):
    count = sum(1 for t in transactions if set(itemset).issubset(set(t)))
    return count / len(transactions)

def get_frequent_itemsets(candidates, transactions, min_support):
    frequent = {}
    for candidate in candidates:
        sup = get_support(candidate, transactions)
        if sup >= min_support:
            frequent[tuple(sorted(candidate))] = sup
    return frequent


# ============================================================================
# STEP 4: GENERATE FREQUENT 1-ITEMSETS (L1)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: GENERATE FREQUENT 1-ITEMSETS (L1)")
print("=" * 80)

print(f"\nGenerating all 1-itemset candidates (C1): {len(ITEMS)} items")
print("Pruning items below minimum support threshold...")

C1 = [[item] for item in ITEMS]
L1 = get_frequent_itemsets(C1, transactions, MIN_SUPPORT)

print(f"\n--- Candidate 1-Itemsets (C1) → Frequent 1-Itemsets (L1) ---")
print(f"{'Item':<15} {'Support':<12} {'Count':<8} {'Status'}")
print("-" * 55)
for item in ITEMS:
    sup = get_support([item], transactions)
    count = int(sup * n_transactions)
    status = "✓ FREQUENT" if sup >= MIN_SUPPORT else "✗ Pruned"
    mark = "✓" if sup >= MIN_SUPPORT else "✗"
    print(f"  {item:<13} {sup:<12.4f} {count:<8} {status}")

print(f"\n  C1 size: {len(C1)} items")
print(f"  L1 size: {len(L1)} items (after pruning {len(C1)-len(L1)} below {MIN_SUPPORT:.0%} support)")

print(f"\n--- L1: All Frequent 1-Itemsets ---")
for itemset, sup in sorted(L1.items(), key=lambda x: -x[1]):
    bar = "█" * int(sup * 40)
    label = '{' + itemset[0] + '}'; print(f"  {label:<14}  support = {sup:.4f}  {bar}")


# ============================================================================
# STEP 5: GENERATE FREQUENT 2-ITEMSETS (L2)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: GENERATE FREQUENT 2-ITEMSETS (L2)")
print("=" * 80)

l1_items = [list(k) for k in L1.keys()]
C2_raw = list(combinations([item[0] for item in l1_items], 2))

print(f"\nApriori Join Step: Combine L1 items to generate C2 candidates")
print(f"  L1 has {len(l1_items)} frequent items")
print(f"  C2 = C({len(l1_items)}, 2) = {len(C2_raw)} candidate pairs")

L2 = get_frequent_itemsets(C2_raw, transactions, MIN_SUPPORT)

print(f"\n--- Candidate 2-Itemsets (C2) → Frequent 2-Itemsets (L2) ---")
print(f"{'Itemset':<30} {'Support':<12} {'Count':<8} {'Status'}")
print("-" * 65)
for pair in C2_raw:
    sup = get_support(list(pair), transactions)
    count = int(sup * n_transactions)
    status = "✓ FREQUENT" if sup >= MIN_SUPPORT else "✗ Pruned"
    itemset_label = f"{{{pair[0]}, {pair[1]}}}"
    print(f"  {itemset_label:<28} {sup:<12.4f} {count:<8} {status}")

print(f"\n  C2 size: {len(C2_raw)} pairs")
print(f"  L2 size: {len(L2)} pairs (pruned {len(C2_raw)-len(L2)} below {MIN_SUPPORT:.0%} support)")

print(f"\n--- L2: All Frequent 2-Itemsets ---")
for itemset, sup in sorted(L2.items(), key=lambda x: -x[1]):
    bar = "█" * int(sup * 40)
    lbl = '{' + ', '.join(itemset) + '}'; print(f"  {lbl:<32}  support = {sup:.4f}  {bar}")


# ============================================================================
# STEP 6: GENERATE FREQUENT 3-ITEMSETS (L3)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: GENERATE FREQUENT 3-ITEMSETS (L3)")
print("=" * 80)

l2_items = [list(k) for k in L2.keys()]
all_l2_items = sorted(set(item for pair in l2_items for item in pair))
C3_raw = list(combinations(all_l2_items, 3))

# Apriori Pruning: candidate 3-itemset is valid only if ALL its 2-subsets are in L2
def apriori_prune_c3(candidates, L2_keys):
    pruned = []
    for candidate in candidates:
        subsets = list(combinations(candidate, 2))
        if all(tuple(sorted(s)) in L2_keys for s in subsets):
            pruned.append(candidate)
    return pruned

L2_keys = set(L2.keys())
C3_pruned = apriori_prune_c3(C3_raw, L2_keys)

print(f"\nApriori Join Step: Combine L2 frequent pairs to generate C3")
print(f"  All items in L2: {all_l2_items}")
print(f"  Raw C3 combinations: {len(C3_raw)}")
print(f"  After Apriori pruning (all 2-subsets must be in L2): {len(C3_pruned)}")

if C3_raw:
    print(f"\n  Pruning demonstration:")
    for c in C3_raw[:min(6, len(C3_raw))]:
        subs = list(combinations(c, 2))
        all_in_l2 = all(tuple(sorted(s)) in L2_keys for s in subs)
        status = "✓ kept" if all_in_l2 else "✗ pruned"
        clbl = '{' + ', '.join(c) + '}'; print(f"    {clbl} — subsets in L2? {all_in_l2} → {status}")

L3 = get_frequent_itemsets(C3_pruned, transactions, MIN_SUPPORT)

print(f"\n  C3 (pruned) size: {len(C3_pruned)}")
print(f"  L3 size: {len(L3)}")

if L3:
    print(f"\n--- L3: All Frequent 3-Itemsets ---")
    for itemset, sup in sorted(L3.items(), key=lambda x: -x[1]):
        bar = "█" * int(sup * 40)
        lbl3 = '{' + ', '.join(itemset) + '}'; print(f"  {lbl3:<42}  support = {sup:.4f}  {bar}")
else:
    print("\n  No frequent 3-itemsets found above minimum support threshold.")

# Combine all frequent itemsets
all_frequent = {}
all_frequent.update(L1)
all_frequent.update(L2)
all_frequent.update(L3)

print(f"\n--- Summary of All Frequent Itemsets ---")
print(f"  L1 (1-item):  {len(L1)} itemsets")
print(f"  L2 (2-items): {len(L2)} itemsets")
print(f"  L3 (3-items): {len(L3)} itemsets")
print(f"  Total:        {len(all_frequent)} frequent itemsets")


# ============================================================================
# STEP 7: GENERATE ASSOCIATION RULES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: GENERATE ASSOCIATION RULES")
print("=" * 80)

print("""
For each frequent itemset with ≥ 2 items, we generate all possible rules:
  Example: {Fiction, Self-Help, Biography}
    → Rule 1: {Fiction, Self-Help} → {Biography}
    → Rule 2: {Fiction, Biography} → {Self-Help}
    → Rule 3: {Self-Help, Biography} → {Fiction}
    → Rule 4: {Fiction} → {Self-Help, Biography}
    → Rule 5: {Self-Help} → {Fiction, Biography}
    → Rule 6: {Biography} → {Fiction, Self-Help}

For each rule: Antecedent → Consequent
  Support    = support({Antecedent ∪ Consequent})
  Confidence = support({A ∪ C}) / support({A})
  Lift       = Confidence / support({C})
""")

rules = []

for itemset, sup_itemset in all_frequent.items():
    if len(itemset) < 2:
        continue
    # Generate all non-empty proper subsets as antecedent
    for size in range(1, len(itemset)):
        for antecedent in combinations(itemset, size):
            antecedent = tuple(sorted(antecedent))
            consequent = tuple(sorted(set(itemset) - set(antecedent)))

            sup_ant = all_frequent.get(antecedent, get_support(list(antecedent), transactions))
            sup_cons = all_frequent.get(consequent, get_support(list(consequent), transactions))

            if sup_ant == 0:
                continue

            confidence = sup_itemset / sup_ant
            lift = confidence / sup_cons if sup_cons > 0 else 0
            leverage = sup_itemset - (sup_ant * sup_cons)
            conviction = (1 - sup_cons) / (1 - confidence) if confidence < 1 else float('inf')

            rules.append({
                'Antecedent': '{' + ', '.join(antecedent) + '}',
                'Consequent': '{' + ', '.join(consequent) + '}',
                'Support': round(sup_itemset, 4),
                'Confidence': round(confidence, 4),
                'Lift': round(lift, 4),
                'Leverage': round(leverage, 4),
                'Conviction': round(conviction, 4) if conviction != float('inf') else 999.0
            })

rules_df = pd.DataFrame(rules)
print(f"  Total rules generated (before filtering): {len(rules_df)}")

# Filter by minimum confidence and lift
strong_rules = rules_df[
    (rules_df['Confidence'] >= MIN_CONFIDENCE) &
    (rules_df['Lift'] >= MIN_LIFT)
].sort_values(['Lift', 'Confidence'], ascending=False).reset_index(drop=True)

print(f"  Rules after filtering (Confidence ≥ {MIN_CONFIDENCE:.0%}, Lift ≥ {MIN_LIFT:.1f}): {len(strong_rules)}")


# ============================================================================
# STEP 8: DISPLAY STRONG ASSOCIATION RULES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: STRONG ASSOCIATION RULES DISCOVERED")
print("=" * 80)

print(f"\n  Filter: Confidence ≥ {MIN_CONFIDENCE:.0%}  |  Lift ≥ {MIN_LIFT:.1f}")
print(f"  Sorted by: Lift (descending), then Confidence")

print(f"\n{'#':<4} {'Antecedent':<22} {'→'} {'Consequent':<22} {'Sup':>6} {'Conf':>7} {'Lift':>7}")
print("-" * 80)
for i, row in strong_rules.iterrows():
    print(f"  {i+1:<3} {row['Antecedent']:<22} {'→':^3} {row['Consequent']:<22} "
          f"{row['Support']:>6.4f} {row['Confidence']:>6.4f} {row['Lift']:>7.4f}")

print(f"\n  Total strong rules: {len(strong_rules)}")


# ============================================================================
# STEP 9: RULE INTERPRETATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: RULE INTERPRETATION")
print("=" * 80)

print("\n--- Top 5 Rules Explained ---")
for i, row in strong_rules.head(5).iterrows():
    print(f"\n  Rule #{i+1}: {row['Antecedent']} → {row['Consequent']}")
    print(f"    Support    = {row['Support']:.4f}  → This combo appears in {row['Support']*100:.1f}% of all transactions")
    print(f"    Confidence = {row['Confidence']:.4f}  → {row['Confidence']*100:.1f}% of customers buying {row['Antecedent']}")
    print(f"                           also buy {row['Consequent']}")
    print(f"    Lift       = {row['Lift']:.4f}  → Buying {row['Antecedent']} makes buying {row['Consequent']}")
    if row['Lift'] > 1:
        print(f"                           {row['Lift']:.2f}x MORE likely than random chance ✓ Positive association")
    elif row['Lift'] < 1:
        print(f"                           {row['Lift']:.2f}x LESS likely than random chance ✗ Negative association")
    else:
        print(f"                           Independent — no association")


# ============================================================================
# STEP 10: BUSINESS RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: BUSINESS RECOMMENDATIONS")
print("=" * 80)

print("""
Based on the Association Rules discovered, here are actionable insights for
the bookstore owner:

1. SHELF PLACEMENT
   Place frequently co-purchased genres near each other.
   Customers who find one easily should be able to spot the other.

2. BUNDLE DEALS / COMBO OFFERS
   Create discounted bundles for top associated genre pairs.
   E.g., "Buy Fiction + Self-Help together and save 15%"

3. RECOMMENDATION SYSTEM
   At checkout: "Customers who bought [X] also bought [Y]"
   Use confidence scores to rank recommendations.

4. PROMOTIONAL EMAILS
   Segment customers by past purchases and send targeted offers
   based on the highest-lift associations.

5. SEASONAL PROMOTIONS
   Use high-support rules for wide campaigns (many customers).
   Use high-lift rules for targeted, niche promotions.
""")

print("--- Specific Rule-Based Recommendations ---")
top_rules = strong_rules.head(6)
for i, row in top_rules.iterrows():
    ant = row['Antecedent'].strip('{}')
    con = row['Consequent'].strip('{}')
    print(f"\n  Rule: {row['Antecedent']} → {row['Consequent']}")
    print(f"  Lift = {row['Lift']:.2f} | Confidence = {row['Confidence']:.2%}")
    if row['Lift'] >= 1.5:
        print(f"  💡 Strong bundle opportunity: Promote {ant} with {con} together.")
    elif row['Lift'] >= 1.2:
        print(f"  📚 Place {con} display near {ant} shelf section.")
    else:
        print(f"  📧 Send {con} recommendation to customers who purchased {ant}.")


# ============================================================================
# STEP 11: COMPREHENSIVE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: CREATE COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

# ── VIZ 1: Item Frequency Bar Chart ─────────────────────────────────────────
print("\n📊 Creating item frequency chart...")
fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.YlOrBr(np.linspace(0.4, 0.9, len(item_freq_df)))
bars = ax.bar(item_freq_df['Item'], item_freq_df['Support'],
              color=colors, edgecolor='black', linewidth=0.8)
ax.axhline(y=MIN_SUPPORT, color='red', linestyle='--', linewidth=2,
           label=f'Min Support = {MIN_SUPPORT:.0%}')
ax.set_xlabel('Book Genre', fontsize=12, fontweight='bold')
ax.set_ylabel('Support (Frequency)', fontsize=12, fontweight='bold')
ax.set_title('Individual Book Genre Support in Transactions', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
for bar, val in zip(bars, item_freq_df['Support']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.2%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('apriori_viz_1_item_frequency.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: apriori_viz_1_item_frequency.png")

# ── VIZ 2: Apriori Step-by-Step Pruning ─────────────────────────────────────
print("\n📊 Creating Apriori pruning steps chart...")
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

def plot_itemset_step(ax, candidates, frequent_dict, title, max_show=12):
    all_c = list(candidates)[:max_show]
    labels = ['{' + ', '.join(c) + '}' for c in all_c]
    supports = [get_support(list(c), transactions) for c in all_c]
    colors_bar = ['#5cb85c' if s >= MIN_SUPPORT else '#d9534f' for s in supports]
    y_pos = range(len(labels))
    ax.barh(y_pos, supports, color=colors_bar, edgecolor='black', linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(x=MIN_SUPPORT, color='black', linestyle='--', linewidth=2,
               label=f'Min Support={MIN_SUPPORT:.0%}')
    ax.set_xlabel('Support', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    green_patch = mpatches.Patch(color='#5cb85c', label='Frequent ✓')
    red_patch = mpatches.Patch(color='#d9534f', label='Pruned ✗')
    ax.legend(handles=[green_patch, red_patch], fontsize=8, loc='lower right')

plot_itemset_step(axes[0], C1, L1, f'C1 → L1\n({len(L1)}/{len(C1)} frequent)')
plot_itemset_step(axes[1], C2_raw, L2, f'C2 → L2\n({len(L2)}/{len(C2_raw)} frequent)')
if C3_pruned:
    plot_itemset_step(axes[2], C3_pruned, L3, f'C3 → L3\n({len(L3)}/{len(C3_pruned)} frequent)')
else:
    axes[2].text(0.5, 0.5, 'No C3 Candidates\n(Apriori Pruning)', ha='center',
                 va='center', transform=axes[2].transAxes, fontsize=13, fontweight='bold')
    axes[2].set_title('C3 → L3\n(Pruned by Apriori)', fontsize=11, fontweight='bold')

plt.suptitle('Apriori Algorithm: Candidate Generation & Pruning Steps', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('apriori_viz_2_pruning_steps.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: apriori_viz_2_pruning_steps.png")

# ── VIZ 3: Support-Confidence Scatter for All Rules ─────────────────────────
print("\n📊 Creating Support-Confidence-Lift scatter plot...")
fig, ax = plt.subplots(figsize=(12, 7))
sc = ax.scatter(rules_df['Support'], rules_df['Confidence'],
                c=rules_df['Lift'], cmap='RdYlGn', s=80,
                edgecolors='black', linewidths=0.5, alpha=0.85)
plt.colorbar(sc, ax=ax, label='Lift')
ax.axhline(y=MIN_CONFIDENCE, color='red', linestyle='--', linewidth=2, label=f'Min Conf = {MIN_CONFIDENCE:.0%}')
ax.axvline(x=MIN_SUPPORT, color='blue', linestyle='--', linewidth=2, label=f'Min Sup = {MIN_SUPPORT:.0%}')
# Annotate top rules
for _, row in strong_rules.head(5).iterrows():
    ax.annotate(f"{row['Antecedent']}→{row['Consequent']}",
                (row['Support'], row['Confidence']),
                textcoords='offset points', xytext=(5, 5),
                fontsize=7, color='black', alpha=0.8)
ax.set_xlabel('Support', fontsize=12, fontweight='bold')
ax.set_ylabel('Confidence', fontsize=12, fontweight='bold')
ax.set_title('All Association Rules — Support vs Confidence (color = Lift)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.fill_betweenx([MIN_CONFIDENCE, 1], MIN_SUPPORT, ax.get_xlim()[1] if ax.get_xlim()[1] > MIN_SUPPORT else 0.5,
                  alpha=0.07, color='green', label='Valid Region')
plt.tight_layout()
plt.savefig('apriori_viz_3_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: apriori_viz_3_scatter.png")

# ── VIZ 4: Top Rules Heatmap (Confidence) ───────────────────────────────────
print("\n📊 Creating confidence heatmap...")
top_r = strong_rules.head(12)
pivot_data = pd.DataFrame(0.0, index=ITEMS, columns=ITEMS)
for _, row in top_r.iterrows():
    ants = row['Antecedent'].strip('{}').split(', ')
    cons = row['Consequent'].strip('{}').split(', ')
    for a in ants:
        for c in cons:
            a, c = a.strip(), c.strip()
            if a in ITEMS and c in ITEMS:
                pivot_data.loc[a, c] = max(pivot_data.loc[a, c], row['Confidence'])

fig, ax = plt.subplots(figsize=(10, 8))
mask = pivot_data == 0
sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd',
            mask=mask, ax=ax, linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Confidence'}, vmin=0.3, vmax=1.0)
ax.set_title('Association Rule Confidence Heatmap\n(Row → Column)', fontsize=13, fontweight='bold')
ax.set_xlabel('Consequent (Y)', fontsize=11, fontweight='bold')
ax.set_ylabel('Antecedent (X)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('apriori_viz_4_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: apriori_viz_4_heatmap.png")

# ── VIZ 5: Top 10 Rules by Lift ─────────────────────────────────────────────
print("\n📊 Creating top rules by lift chart...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

top10 = strong_rules.head(10).copy()
top10['Rule'] = top10['Antecedent'] + ' → ' + top10['Consequent']

# By Lift
colors_lift = plt.cm.YlGn(np.linspace(0.4, 0.9, len(top10)))
axes[0].barh(range(len(top10)), top10['Lift'].values[::-1],
             color=colors_lift[::-1], edgecolor='black')
axes[0].set_yticks(range(len(top10)))
axes[0].set_yticklabels(top10['Rule'].values[::-1], fontsize=8)
axes[0].axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Lift=1 (Random)')
axes[0].set_xlabel('Lift', fontsize=11, fontweight='bold')
axes[0].set_title('Top 10 Rules by Lift', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(axis='x', alpha=0.3)

# By Confidence
colors_conf = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(top10)))
axes[1].barh(range(len(top10)), top10['Confidence'].values[::-1],
             color=colors_conf[::-1], edgecolor='black')
axes[1].set_yticks(range(len(top10)))
axes[1].set_yticklabels(top10['Rule'].values[::-1], fontsize=8)
axes[1].axvline(x=MIN_CONFIDENCE, color='red', linestyle='--', linewidth=2,
                label=f'Min Conf={MIN_CONFIDENCE:.0%}')
axes[1].set_xlabel('Confidence', fontsize=11, fontweight='bold')
axes[1].set_title('Top 10 Rules by Confidence', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(axis='x', alpha=0.3)

plt.suptitle('Strongest Association Rules Discovered', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('apriori_viz_5_top_rules.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: apriori_viz_5_top_rules.png")

# ── VIZ 6: Transaction Size & Co-occurrence Matrix ───────────────────────────
print("\n📊 Creating co-occurrence matrix...")
cooccur = pd.DataFrame(0, index=ITEMS, columns=ITEMS)
for t in transactions:
    for a, b in combinations(t, 2):
        cooccur.loc[a, b] += 1
        cooccur.loc[b, a] += 1

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cooccur, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            linewidths=0.5, linecolor='white')
axes[0].set_title('Item Co-occurrence Matrix\n(Times bought together)', fontsize=12, fontweight='bold')

# Transaction size distribution
size_series = pd.Series(lengths)
size_series.value_counts().sort_index().plot(kind='bar', ax=axes[1],
    color=plt.cm.YlOrBr(np.linspace(0.4, 0.85, size_series.nunique())),
    edgecolor='black')
axes[1].set_xlabel('Number of Items in Transaction', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Number of Transactions', fontsize=11, fontweight='bold')
axes[1].set_title('Transaction Size Distribution', fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x', rotation=0)
axes[1].grid(axis='y', alpha=0.3)
for p in axes[1].patches:
    axes[1].annotate(str(int(p.get_height())),
                     (p.get_x() + p.get_width()/2., p.get_height()),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('Bookstore Transaction Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('apriori_viz_6_cooccurrence.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: apriori_viz_6_cooccurrence.png")


# ============================================================================
# STEP 12: GENERATE COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 12: GENERATE COMPREHENSIVE REPORT")
print("=" * 80)

report = f"""
{'='*80}
ASSOCIATION RULE LEARNING - APRIORI ALGORITHM
SCENARIO: BOOKSTORE PURCHASE PATTERN DISCOVERY
{'='*80}

BUSINESS OBJECTIVE
{'='*80}
A small bookstore wants to discover hidden purchasing patterns:
  • Which book genres are frequently bought together?
  • What recommendations should be made at checkout?
  • How should shelves be arranged to boost cross-sales?
  • Which genres to bundle in promotional combo offers?

Why Apriori for Bookstore Data?
  • Transactions are naturally basket-style (multiple items per visit)
  • No labels or target variable — purely unsupervised pattern mining
  • Actionable rules: IF customer buys X THEN recommend Y
  • Scalable: Apriori pruning handles large item catalogs efficiently

DATASET SUMMARY
{'='*80}
Total transactions: {n_transactions}
Unique book genres: {len(ITEMS)}
Genres: {', '.join(ITEMS)}

Transaction Statistics:
  Average items per transaction: {np.mean(lengths):.2f}
  Most items in one transaction: {max(lengths)}
  Least items in one transaction: {min(lengths)}

Most Popular Genres:
{chr(10).join([f'  {i+1}. {row["Item"]:<15} support = {row["Support"]:.4f} ({row["Count"]} transactions)'
               for i, (_, row) in enumerate(item_freq_df.iterrows())])}

APRIORI PARAMETERS
{'='*80}
  Minimum Support    : {MIN_SUPPORT:.0%}  (appears in ≥ {int(MIN_SUPPORT*n_transactions)} transactions)
  Minimum Confidence : {MIN_CONFIDENCE:.0%}  (rule is correct ≥ {MIN_CONFIDENCE:.0%} of antecedent cases)
  Minimum Lift       : {MIN_LIFT:.1f}   (positive association only)

APRIORI EXECUTION SUMMARY
{'='*80}
Step 1 - Frequent 1-Itemsets (L1):
  Candidates (C1): {len(C1)}
  Frequent (L1):   {len(L1)}
  Pruned:          {len(C1) - len(L1)}

Step 2 - Frequent 2-Itemsets (L2):
  Candidates (C2): {len(C2_raw)}
  Frequent (L2):   {len(L2)}
  Pruned:          {len(C2_raw) - len(L2)}

Step 3 - Frequent 3-Itemsets (L3):
  Candidates (C3): {len(C3_pruned)} (after Apriori pruning from raw {len(C3_raw)})
  Frequent (L3):   {len(L3)}

Total Frequent Itemsets: {len(all_frequent)}

ASSOCIATION RULES
{'='*80}
Total rules generated: {len(rules_df)}
Strong rules (Conf ≥ {MIN_CONFIDENCE:.0%}, Lift ≥ {MIN_LIFT:.1f}): {len(strong_rules)}

--- All Strong Rules (sorted by Lift) ---
{'#':<4} {'Antecedent':<25} → {'Consequent':<25} {'Sup':>8} {'Conf':>8} {'Lift':>8}
{'-'*85}
{chr(10).join([f"  {i+1:<3} {row['Antecedent']:<25} → {row['Consequent']:<25} {row['Support']:>8.4f} {row['Confidence']:>8.4f} {row['Lift']:>8.4f}"
               for i, row in strong_rules.iterrows()])}

TOP RULES EXPLAINED
{'='*80}
{chr(10).join([
    f"Rule #{i+1}: {row['Antecedent']} → {row['Consequent']}"
    f"\n  Support    = {row['Support']:.4f}  ({row['Support']*100:.1f}% of all transactions)"
    f"\n  Confidence = {row['Confidence']:.4f}  ({row['Confidence']*100:.1f}% of people buying {row['Antecedent']} also buy {row['Consequent']})"
    f"\n  Lift       = {row['Lift']:.4f}  ({row['Lift']:.2f}x more likely than random)"
    for i, row in strong_rules.head(5).iterrows()
])}

BUSINESS RECOMMENDATIONS
{'='*80}

1. SHELF PLACEMENT
   • Place Self-Help near Fiction section (strong co-purchase signal)
   • Place Biography near Fiction & History sections
   • Create a "Lifestyle Corner" with Cooking + Travel together
   • Keep Science and History shelves adjacent

2. BUNDLE PROMOTIONS
   • "Reader's Combo": Fiction + Self-Help → save 15%
   • "Explorer's Pack": Cooking + Travel → save 12%
   • "Knowledge Bundle": Science + History + Biography → save 20%

3. CHECKOUT RECOMMENDATIONS
   At checkout, suggest based on cart:
   • Cart has Fiction? → Recommend Self-Help or Biography
   • Cart has Cooking? → Recommend Travel
   • Cart has Science? → Recommend History or Biography

4. EMAIL CAMPAIGNS
   • Target Fiction buyers: offer discount on Self-Help
   • Target Science buyers: announce new History arrivals
   • Target Cooking buyers: highlight Travel bestsellers

5. LOYALTY PROGRAM
   • Award bonus points for purchasing high-lift pairs
   • Create genre-crossing reading challenges (Fiction + Science, etc.)

APRIORI ALGORITHM EXPLANATION
{'='*80}
Core Concepts:

  SUPPORT = P(A and B) = transactions with ItemsetAB / all transactions
    -> Measures HOW FREQUENTLY the itemset appears
    -> High support = popular combination

  CONFIDENCE = P(B | A) = support(A,B) / support(A)
    -> Measures HOW OFTEN the rule is correct
    -> "If customer buys A, they buy B with """ + f"{MIN_CONFIDENCE:.0%}" + """+ probability"

  LIFT = Confidence / P(B) = P(A and B) / (P(A) x P(B))
    -> Measures HOW MUCH BETTER than random chance
    -> Lift > 1: Positive association (buying A encourages buying B)
    -> Lift = 1: Independent (no association)
    -> Lift < 1: Negative association (buying A discourages buying B)

Anti-Monotone (Apriori) Property:
  "If itemset X is infrequent, all supersets of X are also infrequent"
  This eliminates entire branches from the search tree, making the algorithm
  scalable to thousands of items and millions of transactions.

ADVANTAGES OF APRIORI
{'='*80}
  ✓ Unsupervised: No labels needed — learns from raw transactions
  ✓ Interpretable: Rules are human-readable IF→THEN statements
  ✓ Actionable: Direct business application (shelf placement, bundles, recommendations)
  ✓ Scalable: Anti-monotone pruning reduces search space
  ✓ Flexible: Works on any categorical transaction data

LIMITATIONS
{'='*80}
  ⚠ Computationally expensive with very large item sets (exponential search)
  ⚠ Sensitive to support threshold (too low = too many rules, too high = miss patterns)
  ⚠ Does not capture sequence/order of purchases
  ⚠ Correlation ≠ Causation (lift shows association, not why it happens)

FILES GENERATED
{'='*80}
  • apriori_viz_1_item_frequency.png   — Individual genre support
  • apriori_viz_2_pruning_steps.png    — C1→L1, C2→L2, C3→L3 pruning
  • apriori_viz_3_scatter.png          — Support vs Confidence scatter (color=Lift)
  • apriori_viz_4_heatmap.png          — Confidence heatmap (A → B)
  • apriori_viz_5_top_rules.png        — Top 10 rules by Lift and Confidence
  • apriori_viz_6_cooccurrence.png     — Co-occurrence matrix + transaction sizes

{'='*80}
END OF REPORT
{'='*80}
"""

print(report)

with open('apriori_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("\n✓ Report saved to: apriori_report.txt")


# ============================================================================
# STEP 13: SAVE RESULTS TO CSV
# ============================================================================
strong_rules.to_csv('apriori_strong_rules.csv', index=False)
item_freq_df.to_csv('apriori_item_frequency.csv', index=False)
print("✓ Strong rules saved to: apriori_strong_rules.csv")
print("✓ Item frequencies saved to: apriori_item_frequency.csv")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("APRIORI ALGORITHM ANALYSIS COMPLETE!")
print("=" * 80)

print(f"""
📊 Summary:
  ✓ Generated {n_transactions} bookstore transactions across {len(ITEMS)} genres
  ✓ Found {len(L1)} frequent 1-itemsets, {len(L2)} frequent 2-itemsets, {len(L3)} frequent 3-itemsets
  ✓ Generated {len(rules_df)} candidate rules
  ✓ Discovered {len(strong_rules)} strong rules (Conf ≥ {MIN_CONFIDENCE:.0%}, Lift ≥ {MIN_LIFT:.1f})
  ✓ Created 6 comprehensive visualizations

💡 Key Findings:
  • Strongest association: {strong_rules.iloc[0]['Antecedent']} → {strong_rules.iloc[0]['Consequent']}
    (Lift = {strong_rules.iloc[0]['Lift']:.2f}, Confidence = {strong_rules.iloc[0]['Confidence']:.2%})
  • Most popular genre: {item_freq_df.iloc[0]['Item']} ({item_freq_df.iloc[0]['Support']:.2%} support)
  • Best bundle opportunity: top lift pair for promotions

🛒 Business Impact:
  • Shelf rearrangement based on co-occurrence patterns
  • {len(strong_rules)} actionable recommendation rules ready to deploy
  • Seasonal bundle deals backed by data
""")
print("=" * 80)
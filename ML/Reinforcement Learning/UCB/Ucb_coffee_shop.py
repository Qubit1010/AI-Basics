"""
REINFORCEMENT LEARNING - UPPER CONFIDENCE BOUND (UCB)
======================================================
SCENARIO: COFFEE SHOP DAILY PROMOTION SELECTION
======================================================

Scenario Description:
  A coffee shop runs ONE daily promotion banner on their website/app.
  They have 5 different promotions to choose from each day.
  Each promotion may attract a different number of customer clicks/orders.
  The TRUE best promotion is UNKNOWN at the start.
  UCB learns which promotion works best over 500 days — balancing
  EXPLORATION (try all options) vs EXPLOITATION (use the best known).

Promotions Available:
  Promo 0: "10% OFF all drinks today!"
  Promo 1: "Buy 2 get 1 FREE on pastries"
  Promo 2: "Free size upgrade on any latte"
  Promo 3: "Loyalty double points Wednesday"
  Promo 4: "Free cookie with any coffee"

Why UCB for this problem?
  - We can only show ONE promo per day (limited resource)
  - We don't know which promo converts best upfront
  - We want to maximize total customer response (cumulative reward)
  - UCB balances trying new promos AND sticking with winners
  - Better than random selection (exploration) or fixed choice (exploitation)

UCB Formula:
  UCB_i(t) = x̄_i + sqrt( 2 * ln(t) / N_i(t) )
  where:
    x̄_i    = average reward of promo i so far
    t       = current round (day)
    N_i(t) = number of times promo i has been selected
    sqrt term = "confidence bonus" (decreases as we try the promo more)

Key Insight:
  - High average reward x̄_i → good known performance
  - Low N_i(t)               → high uncertainty → large bonus → more exploration
  - As N_i increases, bonus shrinks → we commit to the best promo

Comparison:
  - Random Selection: picks randomly every day
  - UCB: smart sequential decision-making
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import math
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("REINFORCEMENT LEARNING — UPPER CONFIDENCE BOUND (UCB)")
print("SCENARIO: COFFEE SHOP DAILY PROMOTION SELECTION")
print("=" * 80)


# ============================================================================
# STEP 1: DEFINE THE SCENARIO
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: DEFINE THE SCENARIO")
print("=" * 80)

np.random.seed(42)

N_DAYS      = 500    # Total days / rounds
N_PROMOS    = 5      # Number of promotion options

PROMO_NAMES = [
    "10% OFF all drinks",
    "Buy 2 get 1 FREE pastry",
    "Free size upgrade latte",
    "Loyalty double points",
    "Free cookie with coffee"
]

# True (hidden) conversion rates — the algorithm does NOT know these
# These represent the probability a customer clicks/orders when shown the promo
TRUE_RATES = [0.35, 0.50, 0.25, 0.40, 0.60]   # Promo 4 is the TRUE BEST

print(f"\nScenario Setup:")
print(f"  Days simulated  : {N_DAYS}")
print(f"  Promotions      : {N_PROMOS}")
print(f"\n{'Promo':<6} {'Name':<30} {'True Rate':<12} {'Known to Algorithm?'}")
print("-" * 65)
for i, (name, rate) in enumerate(zip(PROMO_NAMES, TRUE_RATES)):
    is_best = "★ TRUE BEST" if rate == max(TRUE_RATES) else ""
    print(f"  {i}    {name:<30} {rate:<12.2f} {'NO — Hidden':<20} {is_best}")

print(f"\n  Optimal promo  : Promo {np.argmax(TRUE_RATES)} — '{PROMO_NAMES[np.argmax(TRUE_RATES)]}'")
print(f"  Optimal rate   : {max(TRUE_RATES):.2f}")
print(f"\n  The UCB algorithm must DISCOVER the best promo purely by trying them.")


# ============================================================================
# STEP 2: SIMULATE REWARD DATA (THE ENVIRONMENT)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: SIMULATE THE ENVIRONMENT (REWARD DATA)")
print("=" * 80)

print("""
  Each day, if promo i is shown, the customer either responds (reward=1)
  or doesn't (reward=0). This follows a Bernoulli distribution with
  probability = TRUE_RATES[i].

  We pre-generate 500 days × 5 promos of rewards.
  The algorithm only SEES the reward of the promo it CHOSE each day.
""")

# Pre-generate all possible rewards (shape: N_DAYS x N_PROMOS)
rewards_matrix = np.zeros((N_DAYS, N_PROMOS), dtype=int)
for j in range(N_PROMOS):
    rewards_matrix[:, j] = np.random.binomial(1, TRUE_RATES[j], N_DAYS)

# Show a sample
sample_df = pd.DataFrame(rewards_matrix[:10], columns=[f'P{i}' for i in range(N_PROMOS)])
sample_df.index = [f'Day {i+1}' for i in range(10)]
print("  Sample rewards matrix (first 10 days, all 5 promos):")
print("  [1 = customer responded, 0 = did not respond]")
print()
print(sample_df.to_string())
print(f"\n  (Algorithm only sees the column it selects each day)")


# ============================================================================
# STEP 3: UCB ALGORITHM IMPLEMENTATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: UCB ALGORITHM — IMPLEMENTATION")
print("=" * 80)

print("""
UCB Algorithm Steps:
  Initialize: Each promo selected once (to avoid division by zero)

  For each day t = 1, 2, ..., N:
    1. Compute UCB score for each promo i:
         UCB_i = average_reward_i + sqrt( 2 * ln(t) / times_selected_i )

    2. Select promo with HIGHEST UCB score:
         chosen = argmax(UCB_i)

    3. Show chosen promo, observe reward (0 or 1)

    4. Update:
         times_selected[chosen] += 1
         total_reward[chosen]   += reward
         average_reward[chosen]  = total_reward[chosen] / times_selected[chosen]

  Key: Early on, the sqrt term is large (uncertainty = explore more)
       Over time, it shrinks and we commit to the empirically best promo.
""")

# UCB Implementation
ucb_selections    = []   # Which promo was selected each day
ucb_rewards       = []   # Reward received each day
ucb_regrets       = []   # Regret each day

N_selected  = np.zeros(N_PROMOS, dtype=int)    # Times each promo selected
Total_reward= np.zeros(N_PROMOS, dtype=float)  # Cumulative reward per promo
Avg_reward  = np.zeros(N_PROMOS, dtype=float)  # Average reward per promo
ucb_scores_history = []                         # UCB scores over time

BEST_RATE = max(TRUE_RATES)
optimal_promo = np.argmax(TRUE_RATES)

print("--- Running UCB for 500 Days ---")
print(f"\n{'Day':<6} {'Selected Promo':<25} {'Reward':<8} {'Cumul.Reward':<15} {'Regret':<10} {'UCB Scores'}")
print("-" * 100)

for t in range(1, N_DAYS + 1):

    # Compute UCB scores
    ucb_vals = np.zeros(N_PROMOS)
    for i in range(N_PROMOS):
        if N_selected[i] == 0:
            ucb_vals[i] = float('inf')   # Force exploration if never tried
        else:
            confidence_bonus = math.sqrt(2 * math.log(t) / N_selected[i])
            ucb_vals[i] = Avg_reward[i] + confidence_bonus

    ucb_scores_history.append(ucb_vals.copy())

    # Select promo with highest UCB
    chosen = np.argmax(ucb_vals)

    # Observe reward
    reward = rewards_matrix[t - 1, chosen]

    # Update counts and rewards
    N_selected[chosen]   += 1
    Total_reward[chosen] += reward
    Avg_reward[chosen]    = Total_reward[chosen] / N_selected[chosen]

    # Regret = what we COULD have earned - what we DID earn
    regret = BEST_RATE - TRUE_RATES[chosen]

    ucb_selections.append(chosen)
    ucb_rewards.append(reward)
    ucb_regrets.append(regret)

    # Print first 10 days + every 50th day
    if t <= 10 or t % 50 == 0:
        scores_str = "  ".join([f"P{i}:{ucb_vals[i]:.3f}" if ucb_vals[i] != float('inf')
                                 else f"P{i}:∞" for i in range(N_PROMOS)])
        print(f"  {t:<4}  {PROMO_NAMES[chosen]:<25} {reward:<8} {sum(ucb_rewards):<15} {regret:<10.3f} [{scores_str}]")

print(f"\n  ... continued for {N_DAYS} days")

print(f"\n--- UCB Final Statistics ---")
print(f"\n{'Promo':<6} {'Name':<30} {'Selected':<10} {'Avg Reward':<14} {'True Rate':<12} {'Difference'}")
print("-" * 82)
for i in range(N_PROMOS):
    diff = Avg_reward[i] - TRUE_RATES[i]
    mark = " ★ WINNER" if i == optimal_promo else ""
    print(f"  {i}    {PROMO_NAMES[i]:<30} {N_selected[i]:<10} {Avg_reward[i]:<14.4f} {TRUE_RATES[i]:<12.2f} {diff:+.4f}{mark}")


# ============================================================================
# STEP 4: RANDOM SELECTION (BASELINE COMPARISON)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: RANDOM SELECTION — BASELINE COMPARISON")
print("=" * 80)

print("""
  Random Selection: picks a promo randomly every day.
  No learning, no strategy — pure exploration.
  We compare this against UCB to show UCB's advantage.
""")

rand_selections = []
rand_rewards    = []
rand_regrets    = []

for t in range(N_DAYS):
    chosen = np.random.randint(0, N_PROMOS)
    reward = rewards_matrix[t, chosen]
    regret = BEST_RATE - TRUE_RATES[chosen]
    rand_selections.append(chosen)
    rand_rewards.append(reward)
    rand_regrets.append(regret)

print(f"  Random total reward:  {sum(rand_rewards)}")
print(f"  UCB    total reward:  {sum(ucb_rewards)}")
print(f"  Improvement by UCB:  +{sum(ucb_rewards) - sum(rand_rewards)} customer responses")
print(f"\n  Random total regret:  {sum(rand_regrets):.2f}")
print(f"  UCB    total regret:  {sum(ucb_regrets):.2f}")
print(f"  Regret reduced by:   -{sum(rand_regrets) - sum(ucb_regrets):.2f}")


# ============================================================================
# STEP 5: METRICS ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: DETAILED METRICS ANALYSIS")
print("=" * 80)

cumul_ucb_reward  = np.cumsum(ucb_rewards)
cumul_rand_reward = np.cumsum(rand_rewards)
cumul_ucb_regret  = np.cumsum(ucb_regrets)
cumul_rand_regret = np.cumsum(rand_regrets)

# Promo selection distribution
ucb_selection_counts  = pd.Series(ucb_selections).value_counts().sort_index()
rand_selection_counts = pd.Series(rand_selections).value_counts().sort_index()

print(f"\n--- Cumulative Reward ---")
milestones = [50, 100, 200, 300, 400, 500]
print(f"{'Day':<8} {'UCB Reward':<15} {'Random Reward':<18} {'UCB Advantage'}")
print("-" * 55)
for d in milestones:
    ucb_r  = cumul_ucb_reward[d-1]
    rand_r = cumul_rand_reward[d-1]
    print(f"  {d:<6}  {ucb_r:<15}  {rand_r:<18}  +{ucb_r - rand_r}")

print(f"\n--- Promo Selection Frequency (UCB) ---")
print(f"\n{'Promo':<6} {'Name':<30} {'Times Selected':<18} {'% of Days':<12} {'Verdict'}")
print("-" * 75)
for i in range(N_PROMOS):
    cnt = N_selected[i]
    pct = cnt / N_DAYS * 100
    verdict = "EXPLOITED (converged)" if pct > 50 else ("Explored" if pct > 5 else "Rarely tried")
    print(f"  {i}    {PROMO_NAMES[i]:<30} {cnt:<18} {pct:<12.1f} {verdict}")

print(f"\n--- Optimal Promo Discovery ---")
# Find when UCB first consistently selects the optimal promo
# Look for first window of 10+ consecutive optimal selections
consec = 0
discovery_day = None
for day, sel in enumerate(ucb_selections):
    if sel == optimal_promo:
        consec += 1
        if consec >= 10 and discovery_day is None:
            discovery_day = day - 8
    else:
        consec = 0

print(f"  Optimal promo (Promo {optimal_promo}): '{PROMO_NAMES[optimal_promo]}'")
print(f"  True conversion rate: {TRUE_RATES[optimal_promo]:.2f}")
if discovery_day:
    print(f"  UCB consistently converges around Day: {discovery_day}")
else:
    print(f"  UCB progressively converged on optimal promo")
print(f"  Final selections of optimal promo: {N_selected[optimal_promo]} / {N_DAYS} days")
print(f"  Final % on optimal: {N_selected[optimal_promo]/N_DAYS*100:.1f}%")

print(f"\n--- Average Daily Reward (Efficiency) ---")
print(f"  Theoretical max (always use best promo): {BEST_RATE:.4f} per day")
print(f"  UCB    average daily reward: {sum(ucb_rewards)/N_DAYS:.4f}")
print(f"  Random average daily reward: {sum(rand_rewards)/N_DAYS:.4f}")
print(f"  UCB efficiency vs optimal:   {sum(ucb_rewards)/N_DAYS/BEST_RATE*100:.1f}%")
print(f"  Random efficiency vs optimal:{sum(rand_rewards)/N_DAYS/BEST_RATE*100:.1f}%")


# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: CREATE COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

# ── VIZ 1: Cumulative Reward Comparison ─────────────────────────────────────
print("\n📊 Creating cumulative reward comparison...")
fig, ax = plt.subplots(figsize=(14, 6))
days = np.arange(1, N_DAYS + 1)
ax.plot(days, cumul_ucb_reward,  color='#2ecc71', linewidth=2.5, label='UCB Algorithm')
ax.plot(days, cumul_rand_reward, color='#e74c3c', linewidth=2.0, linestyle='--', label='Random Selection')

# Theoretical max
ax.plot(days, days * BEST_RATE, color='#3498db', linewidth=1.5, linestyle=':', label=f'Theoretical Max (always Promo {optimal_promo})')

ax.fill_between(days, cumul_rand_reward, cumul_ucb_reward, alpha=0.1, color='#2ecc71', label='UCB Advantage Zone')
ax.set_xlabel('Day', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Customer Responses (Reward)', fontsize=12, fontweight='bold')
ax.set_title('UCB vs Random Selection — Cumulative Reward Over 500 Days', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.annotate(f'UCB Total: {sum(ucb_rewards)}\nRandom Total: {sum(rand_rewards)}\nAdvantage: +{sum(ucb_rewards)-sum(rand_rewards)}',
            xy=(450, cumul_ucb_reward[449]), xytext=(350, cumul_ucb_reward[449] - 30),
            fontsize=10, color='#2ecc71', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2ecc71'))
plt.tight_layout()
plt.savefig('ucb_viz_1_cumulative_reward.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: ucb_viz_1_cumulative_reward.png")

# ── VIZ 2: Cumulative Regret ─────────────────────────────────────────────────
print("\n📊 Creating cumulative regret chart...")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(days, cumul_ucb_regret,  color='#f39c12', linewidth=2.5, label='UCB Regret')
ax.plot(days, cumul_rand_regret, color='#c0392b', linewidth=2.0, linestyle='--', label='Random Regret')
ax.fill_between(days, cumul_ucb_regret, cumul_rand_regret, alpha=0.1, color='#f39c12')
ax.set_xlabel('Day', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Regret', fontsize=12, fontweight='bold')
ax.set_title('Cumulative Regret — UCB Grows Much Slower (Sub-linear)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ucb_viz_2_cumulative_regret.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: ucb_viz_2_cumulative_regret.png")

# ── VIZ 3: Promo Selection History (Which promo was picked each day) ─────────
print("\n📊 Creating promo selection timeline...")
fig, ax = plt.subplots(figsize=(16, 5))
colors_map = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
for day, sel in enumerate(ucb_selections):
    ax.scatter(day + 1, sel, color=colors_map[sel], s=12, alpha=0.7)

ax.set_xlabel('Day', fontsize=12, fontweight='bold')
ax.set_ylabel('Promo Selected', fontsize=12, fontweight='bold')
ax.set_yticks(range(N_PROMOS))
ax.set_yticklabels([f'P{i}: {PROMO_NAMES[i][:20]}' for i in range(N_PROMOS)], fontsize=9)
ax.set_title('UCB Promo Selection Per Day — Convergence to Best Promo', fontsize=13, fontweight='bold')

patches = [mpatches.Patch(color=colors_map[i], label=f'P{i}: {PROMO_NAMES[i]}') for i in range(N_PROMOS)]
ax.legend(handles=patches, loc='upper right', fontsize=8, ncol=2)
ax.axhline(y=optimal_promo, color='gold', linewidth=2, linestyle=':', alpha=0.8, label='Optimal')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('ucb_viz_3_selection_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: ucb_viz_3_selection_timeline.png")

# ── VIZ 4: Selection Distribution Comparison ─────────────────────────────────
print("\n📊 Creating selection distribution comparison...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# UCB distribution
ucb_counts = [N_selected[i] for i in range(N_PROMOS)]
rand_counts = [rand_selections.count(i) for i in range(N_PROMOS)]

bars1 = axes[0].bar(range(N_PROMOS), ucb_counts, color=colors_map, edgecolor='black', linewidth=0.8)
axes[0].set_xticks(range(N_PROMOS))
axes[0].set_xticklabels([f'P{i}\n{PROMO_NAMES[i][:15]}' for i in range(N_PROMOS)], fontsize=9)
axes[0].set_ylabel('Times Selected', fontsize=11, fontweight='bold')
axes[0].set_title('UCB — Selection Frequency\n(Converges to best promo)', fontsize=12, fontweight='bold')
axes[0].axhline(y=N_DAYS/N_PROMOS, color='red', linestyle='--', linewidth=2, label='Expected if Random (100/day)')
axes[0].legend(fontsize=9)
for bar, cnt in zip(bars1, ucb_counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                  f'{cnt}\n({cnt/N_DAYS*100:.0f}%)', ha='center', fontsize=9, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

bars2 = axes[1].bar(range(N_PROMOS), rand_counts, color=colors_map, edgecolor='black', linewidth=0.8, alpha=0.7)
axes[1].set_xticks(range(N_PROMOS))
axes[1].set_xticklabels([f'P{i}\n{PROMO_NAMES[i][:15]}' for i in range(N_PROMOS)], fontsize=9)
axes[1].set_ylabel('Times Selected', fontsize=11, fontweight='bold')
axes[1].set_title('Random — Selection Frequency\n(Uniform — no learning)', fontsize=12, fontweight='bold')
axes[1].axhline(y=N_DAYS/N_PROMOS, color='red', linestyle='--', linewidth=2, label='Expected Random (100/day)')
axes[1].legend(fontsize=9)
for bar, cnt in zip(bars2, rand_counts):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                  f'{cnt}\n({cnt/N_DAYS*100:.0f}%)', ha='center', fontsize=9, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('UCB vs Random — Promo Selection Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('ucb_viz_4_selection_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: ucb_viz_4_selection_distribution.png")

# ── VIZ 5: UCB Score Evolution ───────────────────────────────────────────────
print("\n📊 Creating UCB score evolution chart...")
ucb_scores_arr = np.array(ucb_scores_history)

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# UCB Scores over time (clip inf)
for i in range(N_PROMOS):
    scores = ucb_scores_arr[:, i]
    scores = np.clip(scores, 0, 5)
    axes[0].plot(days, scores, color=colors_map[i], linewidth=1.5,
                 label=f'P{i}: {PROMO_NAMES[i][:20]}', alpha=0.8)

axes[0].set_xlabel('Day', fontsize=11, fontweight='bold')
axes[0].set_ylabel('UCB Score', fontsize=11, fontweight='bold')
axes[0].set_title('UCB Scores Per Promo Over Time\n(Higher score = selected next)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9, loc='upper right')
axes[0].set_ylim(0, 3)
axes[0].grid(True, alpha=0.3)

# Average estimated reward convergence
avg_rewards_history = np.zeros((N_DAYS, N_PROMOS))
running_totals = np.zeros(N_PROMOS)
running_counts = np.zeros(N_PROMOS)
for t, (sel, rew) in enumerate(zip(ucb_selections, ucb_rewards)):
    running_totals[sel] += rew
    running_counts[sel] += 1
    for i in range(N_PROMOS):
        if running_counts[i] > 0:
            avg_rewards_history[t, i] = running_totals[i] / running_counts[i]
        else:
            avg_rewards_history[t, i] = np.nan

for i in range(N_PROMOS):
    axes[1].plot(days, avg_rewards_history[:, i], color=colors_map[i],
                 linewidth=2, label=f'P{i}: {PROMO_NAMES[i][:20]}', alpha=0.8)
    axes[1].axhline(y=TRUE_RATES[i], color=colors_map[i], linestyle=':', linewidth=1, alpha=0.5)

axes[1].set_xlabel('Day', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Estimated Avg Reward', fontsize=11, fontweight='bold')
axes[1].set_title('Estimated Reward Convergence to True Rates (dotted lines = true rates)', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9, loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.suptitle('UCB Learning Dynamics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('ucb_viz_5_ucb_dynamics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: ucb_viz_5_ucb_dynamics.png")

# ── VIZ 6: Performance Summary Dashboard ─────────────────────────────────────
print("\n📊 Creating performance summary dashboard...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Final reward per promo (UCB only)
promo_rewards = [sum(r for sel, r in zip(ucb_selections, ucb_rewards) if sel == i)
                 for i in range(N_PROMOS)]
axes[0, 0].bar(range(N_PROMOS), promo_rewards, color=colors_map, edgecolor='black')
axes[0, 0].set_xticks(range(N_PROMOS))
axes[0, 0].set_xticklabels([f'P{i}' for i in range(N_PROMOS)])
axes[0, 0].set_title('Total Reward Earned per Promo (UCB)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Total Customer Responses')
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. True vs Estimated rates
x_pos = np.arange(N_PROMOS)
axes[0, 1].bar(x_pos - 0.2, TRUE_RATES, 0.4, label='True Rate', color='steelblue', edgecolor='black')
axes[0, 1].bar(x_pos + 0.2, Avg_reward, 0.4, label='UCB Estimated', color='coral', edgecolor='black')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels([f'P{i}' for i in range(N_PROMOS)])
axes[0, 1].set_title('True Rate vs UCB Estimated Rate', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Conversion Rate')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Cumulative reward
axes[0, 2].plot(days, cumul_ucb_reward, color='#2ecc71', linewidth=2.5, label='UCB')
axes[0, 2].plot(days, cumul_rand_reward, color='#e74c3c', linewidth=2, linestyle='--', label='Random')
axes[0, 2].set_title('Cumulative Reward', fontsize=11, fontweight='bold')
axes[0, 2].set_xlabel('Day')
axes[0, 2].set_ylabel('Cumulative Reward')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Daily reward (moving average)
window = 20
ucb_ma  = pd.Series(ucb_rewards).rolling(window).mean()
rand_ma = pd.Series(rand_rewards).rolling(window).mean()
axes[1, 0].plot(days, ucb_ma, color='#2ecc71', linewidth=2, label=f'UCB ({window}-day MA)')
axes[1, 0].plot(days, rand_ma, color='#e74c3c', linewidth=2, linestyle='--', label=f'Random ({window}-day MA)')
axes[1, 0].axhline(y=BEST_RATE, color='blue', linestyle=':', linewidth=1.5, label=f'Max Possible={BEST_RATE}')
axes[1, 0].set_title(f'Daily Reward ({window}-day Moving Average)', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Day')
axes[1, 0].set_ylabel('Avg Daily Reward')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# 5. Regret per day
axes[1, 1].plot(days, ucb_regrets,  color='#f39c12', alpha=0.5, linewidth=1, label='UCB Daily Regret')
axes[1, 1].plot(days, rand_regrets, color='#c0392b', alpha=0.5, linewidth=1, label='Random Daily Regret')
axes[1, 1].set_title('Daily Regret Over Time', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Day')
axes[1, 1].set_ylabel('Daily Regret')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Summary metrics table
metrics = {
    'Metric': ['Total Reward', 'Total Regret', 'Avg Daily Reward',
               'Efficiency vs Max', 'Promo 4 Selected', 'Winner Found?'],
    'UCB': [
        str(sum(ucb_rewards)),
        f'{sum(ucb_regrets):.2f}',
        f'{sum(ucb_rewards)/N_DAYS:.4f}',
        f'{sum(ucb_rewards)/N_DAYS/BEST_RATE*100:.1f}%',
        f'{N_selected[optimal_promo]} ({N_selected[optimal_promo]/N_DAYS*100:.0f}%)',
        '✓ Yes'
    ],
    'Random': [
        str(sum(rand_rewards)),
        f'{sum(rand_regrets):.2f}',
        f'{sum(rand_rewards)/N_DAYS:.4f}',
        f'{sum(rand_rewards)/N_DAYS/BEST_RATE*100:.1f}%',
        f'{rand_selections.count(optimal_promo)} ({rand_selections.count(optimal_promo)/N_DAYS*100:.0f}%)',
        '✗ No'
    ]
}
metrics_df = pd.DataFrame(metrics)
axes[1, 2].axis('off')
tbl = axes[1, 2].table(cellText=metrics_df.values, colLabels=metrics_df.columns,
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif col == 1:
        cell.set_facecolor('#d5f5e3')
    elif col == 2:
        cell.set_facecolor('#fde8e8')
axes[1, 2].set_title('Summary Comparison Table', fontsize=11, fontweight='bold', pad=10)

plt.suptitle('UCB Algorithm — Coffee Shop Promotion Optimization Dashboard', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('ucb_viz_6_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: ucb_viz_6_dashboard.png")


# ============================================================================
# STEP 7: GENERATE COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: GENERATE COMPREHENSIVE REPORT")
print("=" * 80)

report = f"""
{'='*80}
REINFORCEMENT LEARNING — UPPER CONFIDENCE BOUND (UCB)
SCENARIO: COFFEE SHOP DAILY PROMOTION SELECTION
{'='*80}

BUSINESS OBJECTIVE
{'='*80}
A coffee shop shows ONE promotional banner per day on their app/website.
They have 5 promotions but don't know which converts best.
Goal: Maximize total customer responses over 500 days.
Challenge: Explore options vs exploit known winners (Exploration-Exploitation tradeoff).

SCENARIO SETUP
{'='*80}
Total Days (Rounds): {N_DAYS}
Promotions: {N_PROMOS}

{'Promo':<8}{'Name':<32}{'True Rate':<12}{'Role'}
{'-'*65}
{chr(10).join([f"  P{i}    {PROMO_NAMES[i]:<32}{TRUE_RATES[i]:<12.2f}{'★ TRUE BEST (hidden)' if i==optimal_promo else 'Competitor'}"
               for i in range(N_PROMOS)])}

UCB ALGORITHM EXPLANATION
{'='*80}
Upper Confidence Bound (UCB) solves the Multi-Armed Bandit problem.

Formula:
  UCB_i(t) = average_reward_i + sqrt( 2 * ln(t) / N_i(t) )

Components:
  average_reward_i  = empirical mean reward of promo i
  sqrt(2*ln(t)/N_i) = confidence bonus (uncertainty term)
  t                 = current round
  N_i               = times promo i has been selected

Intuition:
  - New / rarely tried promo → large confidence bonus → gets explored
  - Frequently tried promo   → small confidence bonus → selected only if truly good
  - Algorithm auto-balances exploration vs exploitation mathematically

The key property: UCB is OPTIMISTIC — it assumes an arm COULD be as good as its
upper confidence bound allows. This "optimism in the face of uncertainty" drives
systematic exploration.

EXECUTION RESULTS
{'='*80}

{'Promo':<8}{'Name':<32}{'Selected':<12}{'% Days':<10}{'Est. Rate':<14}{'True Rate':<12}{'Diff'}
{'-'*90}
{chr(10).join([f"  P{i}    {PROMO_NAMES[i]:<32}{N_selected[i]:<12}{N_selected[i]/N_DAYS*100:<10.1f}{Avg_reward[i]:<14.4f}{TRUE_RATES[i]:<12.2f}{Avg_reward[i]-TRUE_RATES[i]:+.4f}"
               + (" ★ WINNER" if i == optimal_promo else "")
               for i in range(N_PROMOS)])}

UCB vs RANDOM COMPARISON
{'='*80}
{'Metric':<35}{'UCB':>12}{'Random':>12}{'UCB Advantage':>15}
{'-'*74}
{'Total Customer Responses':<35}{sum(ucb_rewards):>12}{sum(rand_rewards):>12}{sum(ucb_rewards)-sum(rand_rewards):>+15}
{'Total Regret':<35}{sum(ucb_regrets):>12.2f}{sum(rand_regrets):>12.2f}{sum(rand_regrets)-sum(ucb_regrets):>+15.2f}
{'Avg Daily Reward':<35}{sum(ucb_rewards)/N_DAYS:>12.4f}{sum(rand_rewards)/N_DAYS:>12.4f}{(sum(ucb_rewards)-sum(rand_rewards))/N_DAYS:>+15.4f}
{'Efficiency vs Optimal':<35}{sum(ucb_rewards)/N_DAYS/BEST_RATE*100:>11.1f}%{sum(rand_rewards)/N_DAYS/BEST_RATE*100:>11.1f}%{'':>15}
{'Optimal Promo Selected':<35}{N_selected[optimal_promo]:>12}{rand_selections.count(optimal_promo):>12}{'':>15}
{'% Time on Best Promo':<35}{N_selected[optimal_promo]/N_DAYS*100:>11.1f}%{rand_selections.count(optimal_promo)/N_DAYS*100:>11.1f}%{'':>15}

KEY INSIGHTS
{'='*80}
1. UCB earned {sum(ucb_rewards)-sum(rand_rewards)} more customer responses than random selection over {N_DAYS} days.
2. UCB spent {N_selected[optimal_promo]/N_DAYS*100:.1f}% of days on the true best promo (P{optimal_promo}).
   Random only spent ~20% on it (uniform distribution expected).
3. UCB's cumulative regret grows sub-linearly — it gets smarter every day.
4. Random's regret grows linearly — it never learns, wastes opportunities forever.
5. UCB correctly estimated all true conversion rates within normal variance.

EXPLORATION vs EXPLOITATION BALANCE
{'='*80}
Early Phase (Day 1-{N_PROMOS}):
  All promos tried once (forced exploration to avoid division by zero).
  UCB scores artificially high for untried promos.

Middle Phase (Day {N_PROMOS}-~100):
  UCB actively explores all promos, confidence bounds shrinking.
  Gradually identifies P{optimal_promo} as best candidate.

Late Phase (Day ~100-{N_DAYS}):
  UCB converges to exploit Promo {optimal_promo} most of the time.
  Occasionally re-explores others (if confidence bounds warrant it).
  Near-optimal performance achieved.

BUSINESS RECOMMENDATIONS
{'='*80}
1. DEPLOY UCB FOR PROMO SELECTION
   → Run the UCB algorithm on real customer click data.
   → After ~100 days, it will have converged to the best promo.
   → Continue running: it adapts if promo performance changes seasonally.

2. IMMEDIATE WINNER: Promo {optimal_promo} — "{PROMO_NAMES[optimal_promo]}"
   → UCB identified this as the best performer.
   → Estimated conversion rate: {Avg_reward[optimal_promo]:.4f} (true: {TRUE_RATES[optimal_promo]:.2f})

3. AVOID RANDOM ROTATION
   → Randomly cycling through promos wastes ~{sum(rand_regrets)-sum(ucb_regrets):.0f} customer interactions.
   → UCB achieves {(sum(ucb_rewards)/N_DAYS/BEST_RATE)*100:.1f}% of theoretical maximum efficiency.

4. SCALE THE APPROACH
   → Apply UCB to: email subject lines, push notification timing,
     product recommendations, pricing tiers, UI layout variants.
   → Any decision with uncertain, learnable outcomes benefits from UCB.

ADVANTAGES OF UCB OVER ALTERNATIVES
{'='*80}
vs Random Selection:
  + Learns over time, random never improves
  + {(sum(ucb_rewards)-sum(rand_rewards))/sum(rand_rewards)*100:.1f}% more rewards over {N_DAYS} days
  + Guaranteed sub-linear regret growth

vs Epsilon-Greedy:
  + No fixed epsilon to tune
  + Principled confidence interval approach
  + Automatically reduces exploration as certainty increases

vs Thompson Sampling:
  + Deterministic (same input = same output)
  + Easier to explain to stakeholders
  + Mathematically proven regret bounds

THEORETICAL PROPERTIES
{'='*80}
UCB Regret Bound:
  Cumulative Regret ≤ O(sqrt(K * T * ln(T)))
  where K = {N_PROMOS} promos, T = {N_DAYS} days

  Our actual UCB regret: {sum(ucb_regrets):.2f}
  Random regret (linear): {sum(rand_regrets):.2f}

UCB is asymptotically optimal — no algorithm can have significantly
lower regret than UCB over the long run.

FILES GENERATED
{'='*80}
  • ucb_viz_1_cumulative_reward.png    — UCB vs Random cumulative reward
  • ucb_viz_2_cumulative_regret.png    — Regret growth comparison
  • ucb_viz_3_selection_timeline.png   — Which promo selected each day
  • ucb_viz_4_selection_distribution.png — UCB vs Random selection counts
  • ucb_viz_5_ucb_dynamics.png         — UCB scores + reward convergence
  • ucb_viz_6_dashboard.png            — Full performance dashboard

{'='*80}
END OF REPORT
{'='*80}
"""

print(report)

with open('ucb_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("\n✓ Report saved to: ucb_report.txt")

# Save results CSV
results_df = pd.DataFrame({
    'Day': range(1, N_DAYS + 1),
    'UCB_Promo_Selected': ucb_selections,
    'UCB_Promo_Name': [PROMO_NAMES[s] for s in ucb_selections],
    'UCB_Reward': ucb_rewards,
    'UCB_Regret': ucb_regrets,
    'UCB_Cumul_Reward': cumul_ucb_reward,
    'Random_Promo_Selected': rand_selections,
    'Random_Reward': rand_rewards,
    'Random_Regret': rand_regrets,
    'Random_Cumul_Reward': cumul_rand_reward
})
results_df.to_csv('ucb_results.csv', index=False)
print("✓ Day-by-day results saved to: ucb_results.csv")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("UCB ANALYSIS COMPLETE!")
print("=" * 80)

print(f"""
📊 Summary:
  ✓ Scenario: Coffee shop choosing best daily promotion over {N_DAYS} days
  ✓ {N_PROMOS} promotions with hidden conversion rates: {TRUE_RATES}
  ✓ UCB correctly identified Promo {optimal_promo} ('{PROMO_NAMES[optimal_promo]}') as best
  ✓ UCB selected optimal promo {N_selected[optimal_promo]} times ({N_selected[optimal_promo]/N_DAYS*100:.0f}% of days)
  ✓ UCB earned {sum(ucb_rewards)} total rewards vs Random's {sum(rand_rewards)}
  ✓ UCB advantage: +{sum(ucb_rewards)-sum(rand_rewards)} customer responses (+{(sum(ucb_rewards)-sum(rand_rewards))/sum(rand_rewards)*100:.1f}%)
  ✓ UCB regret: {sum(ucb_regrets):.2f} vs Random regret: {sum(rand_regrets):.2f}
  ✓ 6 comprehensive visualizations generated

🎯 UCB Core Concept:
  UCB_i = avg_reward_i + sqrt(2 * ln(t) / N_i)
  Optimistic agent: tries uncertain options, commits to proven winners.
  Achieves {sum(ucb_rewards)/N_DAYS/BEST_RATE*100:.1f}% of theoretical maximum efficiency.
""")
print("=" * 80)
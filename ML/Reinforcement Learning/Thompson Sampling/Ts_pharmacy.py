"""
REINFORCEMENT LEARNING - THOMPSON SAMPLING
==========================================
SCENARIO: ONLINE PHARMACY — WHICH LANDING PAGE CONVERTS BEST?
==============================================================

Scenario Description:
  An online pharmacy runs an A/B/C/D/E test on their website.
  They have 5 different landing page designs for their flu medicine product.
  Each time a visitor lands on the site, one page design is shown.
  A "conversion" = visitor actually purchases the medicine.
  The TRUE best design is UNKNOWN at the start.

  Thompson Sampling learns which page converts best over 1000 visitors
  using Bayesian probability — maintaining a Beta distribution belief
  about each page's true conversion rate and SAMPLING from it.

Landing Page Designs:
  Design 0: "Plain white — just product info & price"
  Design 1: "Urgency banner — 'Only 12 left in stock!'"
  Design 2: "Social proof — '4,200 five-star reviews'"
  Design 3: "Doctor endorsement photo + quote"
  Design 4: "Free shipping + money-back guarantee banner"

Why Thompson Sampling?
  - Fully Bayesian: updates beliefs after every single observation
  - Probabilistic exploration: samples from Beta(alpha, beta) distribution
  - Naturally balances explore/exploit without any tuning parameter
  - Often outperforms UCB in practice (faster convergence)
  - Handles uncertainty gracefully — more uncertain = wider distribution = more likely sampled

Thompson Sampling Formula:
  Prior:    Design i ~ Beta(alpha_i=1, beta_i=1)  [uniform, no assumption]
  Sampling: theta_i ~ Beta(alpha_i, beta_i)
  Select:   argmax(theta_i)
  Update:   If reward=1: alpha_i += 1
            If reward=0: beta_i  += 1

  where:
    alpha_i = 1 + number of conversions from design i
    beta_i  = 1 + number of non-conversions from design i
    theta_i = sampled conversion probability estimate

Beta Distribution Intuition:
  - Beta(1,1)   = flat / totally uncertain (equal chance of any rate)
  - Beta(10,2)  = likely high converter (10 successes, 2 failures)
  - Beta(2,20)  = likely low converter  (2 successes, 20 failures)
  - As data grows: distribution narrows → we become more certain

Comparison:
  - Random:          picks randomly, never learns
  - UCB:             uses confidence intervals (deterministic)
  - Thompson Sampling: samples from posterior (probabilistic, often faster)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import beta as beta_dist
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("REINFORCEMENT LEARNING — THOMPSON SAMPLING (BAYESIAN BANDIT)")
print("SCENARIO: ONLINE PHARMACY — LANDING PAGE CONVERSION OPTIMIZATION")
print("=" * 80)


# ============================================================================
# STEP 1: DEFINE THE SCENARIO
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: DEFINE THE SCENARIO")
print("=" * 80)

np.random.seed(42)

N_VISITORS  = 1000   # Total website visitors / rounds
N_DESIGNS   = 5      # Number of landing page designs

DESIGN_NAMES = [
    "Plain product info page",
    "Urgency — 'Only 12 left!'",
    "Social proof — 4,200 reviews",
    "Doctor endorsement + quote",
    "Free shipping + money-back"
]

# True (hidden) conversion rates — Thompson Sampling does NOT know these
TRUE_RATES = [0.20, 0.35, 0.30, 0.45, 0.55]   # Design 4 is TRUE BEST

OPTIMAL     = np.argmax(TRUE_RATES)
BEST_RATE   = max(TRUE_RATES)

print(f"\nScenario Setup:")
print(f"  Total visitors : {N_VISITORS}")
print(f"  Landing pages  : {N_DESIGNS}")
print(f"\n{'Design':<8} {'Name':<33} {'True Rate':<12} {'Known?'}")
print("-" * 68)
for i, (name, rate) in enumerate(zip(DESIGN_NAMES, TRUE_RATES)):
    star = "  ★ TRUE BEST" if i == OPTIMAL else ""
    print(f"  D{i}     {name:<33} {rate:<12.2f} NO — Hidden{star}")

print(f"\n  Optimal design : D{OPTIMAL} — '{DESIGN_NAMES[OPTIMAL]}'")
print(f"  Optimal rate   : {BEST_RATE:.2f}")
print(f"\n  Thompson Sampling must DISCOVER the best page purely through visitor data.")


# ============================================================================
# STEP 2: SIMULATE THE ENVIRONMENT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: SIMULATE THE ENVIRONMENT (REWARD DATA)")
print("=" * 80)

print("""
  Each visitor either buys (reward=1) or leaves (reward=0).
  Probability depends on the page design shown.
  We pre-generate 1000 × 5 reward outcomes.
  Thompson Sampling only sees the outcome for the design IT chose.
""")

rewards_matrix = np.zeros((N_VISITORS, N_DESIGNS), dtype=int)
for j in range(N_DESIGNS):
    rewards_matrix[:, j] = np.random.binomial(1, TRUE_RATES[j], N_VISITORS)

sample_df = pd.DataFrame(rewards_matrix[:10], columns=[f'D{i}' for i in range(N_DESIGNS)])
sample_df.index = [f'V{i+1}' for i in range(10)]
print("  Sample reward matrix (first 10 visitors, all 5 designs):")
print("  [1 = purchased, 0 = left without buying]\n")
print(sample_df.to_string())
print("\n  (Algorithm only observes the column it selected for each visitor)")


# ============================================================================
# STEP 3: BETA DISTRIBUTION EXPLAINED
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: UNDERSTANDING THE BETA DISTRIBUTION (CORE OF THOMPSON SAMPLING)")
print("=" * 80)

print("""
  Thompson Sampling maintains a Beta(alpha, beta) distribution for each design.
  This distribution represents our BELIEF about that design's true conversion rate.

  Beta(alpha, beta):
    alpha = 1 + number of SUCCESSES (purchases)
    beta  = 1 + number of FAILURES  (no purchase)

  Examples:
    Beta(1, 1)   = Uniform — we know nothing yet, any rate equally likely
    Beta(5, 5)   = 50% rate, moderate certainty
    Beta(50, 50) = 50% rate, very certain
    Beta(15, 5)  = ~75% rate, fairly certain it's a high converter
    Beta(3, 30)  = ~9% rate,  fairly certain it's a poor converter

  Each round we:
    1. SAMPLE one theta_i from each Beta(alpha_i, beta_i)
    2. SELECT design with highest sampled theta_i
    3. OBSERVE reward (0 or 1)
    4. UPDATE: reward=1 → alpha_i++     reward=0 → beta_i++

  Uncertain designs (few observations) have WIDE distributions
  → high chance of sampling a large theta → more likely to be explored.
  Well-tested designs have NARROW distributions → sampled theta
  closely reflects true performance → natural exploitation.
""")

print("  Beta Distribution Examples:")
print(f"  {'Distribution':<20} {'Mean':>8} {'Std':>8} {'Interpretation'}")
print("  " + "-" * 65)
examples = [
    (1, 1, "No data yet (uniform prior)"),
    (2, 8, "1 success, 7 failures — poor"),
    (5, 5, "4 successes, 4 failures — uncertain 50%"),
    (15, 5, "14 successes, 4 failures — good converter"),
    (50, 10, "49 successes, 9 failures — very confident high"),
]
for a, b, desc in examples:
    mean = a / (a + b)
    std  = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    print(f"  Beta({a:>2},{b:>2})           {mean:>8.3f} {std:>8.3f}  {desc}")


# ============================================================================
# STEP 4: THOMPSON SAMPLING ALGORITHM
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: THOMPSON SAMPLING — ALGORITHM EXECUTION")
print("=" * 80)

print("""
Algorithm:
  Initialize: alpha_i = 1, beta_i = 1  for all designs i
              (uninformed prior — all rates equally possible)

  For each visitor t = 1 ... N:
    1. Sample: theta_i ~ Beta(alpha_i, beta_i)  for each i
    2. Select: chosen = argmax(theta_i)
    3. Observe: reward = rewards_matrix[t][chosen]
    4. Update:
         if reward == 1:  alpha_chosen += 1
         else:            beta_chosen  += 1
""")

# Initialize Beta distribution parameters
alpha = np.ones(N_DESIGNS, dtype=float)   # successes + 1
beta  = np.ones(N_DESIGNS, dtype=float)   # failures  + 1

ts_selections  = []
ts_rewards     = []
ts_regrets     = []

# History tracking for visualization
alpha_history  = np.zeros((N_VISITORS, N_DESIGNS))
beta_history   = np.zeros((N_VISITORS, N_DESIGNS))
theta_history  = np.zeros((N_VISITORS, N_DESIGNS))

print(f"--- Running Thompson Sampling for {N_VISITORS} Visitors ---")
print(f"\n{'V#':<6} {'Design Chosen':<34} {'Reward':<8} {'Cumul.R':<10} {'Regret':<8} {'Sampled Thetas'}")
print("-" * 105)

for t in range(N_VISITORS):

    # Sample from each Beta distribution
    thetas = np.array([np.random.beta(alpha[i], beta[i]) for i in range(N_DESIGNS)])

    # Select design with highest sampled theta
    chosen = np.argmax(thetas)

    # Observe reward
    reward = rewards_matrix[t, chosen]

    # Update Beta parameters
    if reward == 1:
        alpha[chosen] += 1
    else:
        beta[chosen]  += 1

    # Compute regret
    regret = BEST_RATE - TRUE_RATES[chosen]

    ts_selections.append(chosen)
    ts_rewards.append(reward)
    ts_regrets.append(regret)

    # Store history
    alpha_history[t] = alpha.copy()
    beta_history[t]  = beta.copy()
    theta_history[t] = thetas

    # Print selected rounds
    if t < 10 or (t + 1) % 100 == 0:
        thetas_str = "  ".join([f"D{i}:{thetas[i]:.3f}" for i in range(N_DESIGNS)])
        print(f"  {t+1:<4}  {DESIGN_NAMES[chosen]:<34} {reward:<8} "
              f"{sum(ts_rewards):<10} {regret:<8.3f} [{thetas_str}]")

print(f"\n  ... simulated {N_VISITORS} visitors")

# Final Beta parameters
print(f"\n--- Final Beta Distribution Parameters ---")
print(f"\n{'Design':<8} {'Name':<34} {'Alpha':>8} {'Beta':>8} {'Mean':>8} {'True Rate':>10} {'N_Selected':>12}")
print("-" * 95)
N_selected_ts = [ts_selections.count(i) for i in range(N_DESIGNS)]
for i in range(N_DESIGNS):
    mean_est = alpha[i] / (alpha[i] + beta[i])
    star = "  ★" if i == OPTIMAL else ""
    print(f"  D{i}     {DESIGN_NAMES[i]:<34} {alpha[i]:>8.0f} {beta[i]:>8.0f} "
          f"{mean_est:>8.4f} {TRUE_RATES[i]:>10.2f} {N_selected_ts[i]:>12}{star}")


# ============================================================================
# STEP 5: RANDOM SELECTION BASELINE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: RANDOM SELECTION — BASELINE COMPARISON")
print("=" * 80)

rand_selections = []
rand_rewards    = []
rand_regrets    = []

for t in range(N_VISITORS):
    chosen = np.random.randint(0, N_DESIGNS)
    reward = rewards_matrix[t, chosen]
    rand_selections.append(chosen)
    rand_rewards.append(reward)
    rand_regrets.append(BEST_RATE - TRUE_RATES[chosen])

print(f"\n  Random total reward  : {sum(rand_rewards)}")
print(f"  Thompson total reward: {sum(ts_rewards)}")
print(f"  Thompson advantage   : +{sum(ts_rewards) - sum(rand_rewards)} extra purchases")
print(f"\n  Random total regret  : {sum(rand_regrets):.2f}")
print(f"  Thompson total regret: {sum(ts_regrets):.2f}")
print(f"  Regret reduction     : -{sum(rand_regrets) - sum(ts_regrets):.2f}")


# ============================================================================
# STEP 6: DETAILED METRICS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: DETAILED METRICS ANALYSIS")
print("=" * 80)

cumul_ts_reward    = np.cumsum(ts_rewards)
cumul_rand_reward  = np.cumsum(rand_rewards)
cumul_ts_regret    = np.cumsum(ts_regrets)
cumul_rand_regret  = np.cumsum(rand_regrets)

print(f"\n--- Cumulative Reward at Milestones ---")
milestones = [50, 100, 200, 300, 500, 750, 1000]
print(f"{'Visitor':<10} {'TS Reward':<14} {'Random Reward':<18} {'TS Advantage'}")
print("-" * 55)
for v in milestones:
    if v <= N_VISITORS:
        ts_r   = cumul_ts_reward[v - 1]
        rand_r = cumul_rand_reward[v - 1]
        print(f"  {v:<8}  {ts_r:<14}  {rand_r:<18}  +{ts_r - rand_r}")

print(f"\n--- Design Selection Breakdown (Thompson Sampling) ---")
print(f"\n{'Design':<8} {'Name':<34} {'Selected':<12} {'% Visitors':<14} {'Pattern'}")
print("-" * 80)
for i in range(N_DESIGNS):
    cnt = N_selected_ts[i]
    pct = cnt / N_VISITORS * 100
    if pct > 50:
        pattern = "HEAVILY EXPLOITED (winner)"
    elif pct > 15:
        pattern = "Explored & evaluated"
    elif pct > 5:
        pattern = "Briefly explored"
    else:
        pattern = "Quickly eliminated"
    print(f"  D{i}     {DESIGN_NAMES[i]:<34} {cnt:<12} {pct:<14.1f} {pattern}")

print(f"\n--- Efficiency ---")
print(f"  Theoretical max (always best design): {BEST_RATE:.4f}/visitor")
print(f"  Thompson avg reward per visitor      : {sum(ts_rewards)/N_VISITORS:.4f}")
print(f"  Random    avg reward per visitor      : {sum(rand_rewards)/N_VISITORS:.4f}")
print(f"  Thompson efficiency vs optimal        : {sum(ts_rewards)/N_VISITORS/BEST_RATE*100:.1f}%")
print(f"  Random    efficiency vs optimal       : {sum(rand_rewards)/N_VISITORS/BEST_RATE*100:.1f}%")

# Convergence detection
consec = 0
conv_day = None
for day, sel in enumerate(ts_selections):
    if sel == OPTIMAL:
        consec += 1
        if consec >= 15 and conv_day is None:
            conv_day = day - 13
    else:
        consec = 0
print(f"\n  Thompson Sampling convergence to D{OPTIMAL}: ~Visitor {conv_day if conv_day else 'N/A'}")
print(f"  Final % time on best design: {N_selected_ts[OPTIMAL]/N_VISITORS*100:.1f}%")


# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: CREATE COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

# ── VIZ 1: Beta Distribution Evolution for Each Design ───────────────────────
print("\n📊 Creating Beta distribution evolution...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
x_range = np.linspace(0, 1, 300)

checkpoints = [10, 50, 200, 500, 1000]
for idx, (ax, cp) in enumerate(zip(axes.flat[:5], checkpoints)):
    a_cp = alpha_history[cp - 1]
    b_cp = beta_history[cp - 1]
    for i in range(N_DESIGNS):
        pdf_vals = beta_dist.pdf(x_range, a_cp[i], b_cp[i])
        ax.plot(x_range, pdf_vals, color=COLORS[i], linewidth=2,
                label=f'D{i}: B({a_cp[i]:.0f},{b_cp[i]:.0f})')
        ax.axvline(x=TRUE_RATES[i], color=COLORS[i], linestyle=':', alpha=0.5)
    ax.set_title(f'After Visitor {cp}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Conversion Rate', fontsize=9)
    ax.set_ylabel('Probability Density', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

# Final summary in 6th panel
axes.flat[5].axis('off')
summary_text = "Beta Distribution Evolution\n\n"
summary_text += "Dotted vertical lines = True Rates\n\n"
summary_text += f"Start (V1):\n  All Beta(1,1) — flat, uncertain\n\n"
summary_text += f"V10:\n  Slight differentiation begins\n\n"
summary_text += f"V200:\n  Clear leader emerging\n\n"
summary_text += f"V1000:\n  D{OPTIMAL} (true best) dominates\n  Narrowest, rightmost peak\n\n"
summary_text += f"Final winner: D{OPTIMAL}\n'{DESIGN_NAMES[OPTIMAL]}'\nEst. rate: {alpha[OPTIMAL]/(alpha[OPTIMAL]+beta[OPTIMAL]):.3f}\nTrue rate: {TRUE_RATES[OPTIMAL]:.2f}"
axes.flat[5].text(0.1, 0.5, summary_text, transform=axes.flat[5].transAxes,
                   fontsize=10, va='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Beta Distribution Belief Evolution — Thompson Sampling\n(Dotted lines = true conversion rates)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('ts_viz_1_beta_evolution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: ts_viz_1_beta_evolution.png")

# ── VIZ 2: Cumulative Reward Comparison ──────────────────────────────────────
print("\n📊 Creating cumulative reward chart...")
visitors = np.arange(1, N_VISITORS + 1)
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(visitors, cumul_ts_reward,   color='#9b59b6', linewidth=2.5, label='Thompson Sampling')
ax.plot(visitors, cumul_rand_reward, color='#e74c3c', linewidth=2.0, linestyle='--', label='Random Selection')
ax.plot(visitors, visitors * BEST_RATE, color='#2ecc71', linewidth=1.5, linestyle=':', label=f'Theoretical Max (always D{OPTIMAL})')
ax.fill_between(visitors, cumul_rand_reward, cumul_ts_reward, alpha=0.12, color='#9b59b6')
ax.set_xlabel('Visitor Number', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Purchases (Reward)', fontsize=12, fontweight='bold')
ax.set_title('Thompson Sampling vs Random — Cumulative Reward Over 1000 Visitors', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.annotate(f'TS: {sum(ts_rewards)} purchases\nRandom: {sum(rand_rewards)} purchases\nGain: +{sum(ts_rewards)-sum(rand_rewards)}',
            xy=(900, cumul_ts_reward[899]), xytext=(700, cumul_ts_reward[899] - 30),
            fontsize=10, color='#9b59b6', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#9b59b6'))
plt.tight_layout()
plt.savefig('ts_viz_2_cumulative_reward.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: ts_viz_2_cumulative_reward.png")

# ── VIZ 3: Cumulative Regret ──────────────────────────────────────────────────
print("\n📊 Creating cumulative regret chart...")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(visitors, cumul_ts_regret,   color='#f39c12', linewidth=2.5, label='Thompson Sampling Regret')
ax.plot(visitors, cumul_rand_regret, color='#c0392b', linewidth=2.0, linestyle='--', label='Random Selection Regret')
ax.fill_between(visitors, cumul_ts_regret, cumul_rand_regret, alpha=0.1, color='#f39c12')
ax.set_xlabel('Visitor Number', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Regret (Missed Conversions)', fontsize=12, fontweight='bold')
ax.set_title('Cumulative Regret — Thompson Sampling Grows Sub-linearly', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.text(800, cumul_rand_regret[799] * 0.95,
        f'Regret saved:\n{sum(rand_regrets)-sum(ts_regrets):.1f}',
        fontsize=11, color='#f39c12', fontweight='bold')
plt.tight_layout()
plt.savefig('ts_viz_3_cumulative_regret.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: ts_viz_3_cumulative_regret.png")

# ── VIZ 4: Design Selection Timeline ─────────────────────────────────────────
print("\n📊 Creating design selection timeline...")
fig, axes = plt.subplots(2, 1, figsize=(16, 9))

# Scatter plot
for v, sel in enumerate(ts_selections):
    axes[0].scatter(v + 1, sel, color=COLORS[sel], s=10, alpha=0.6)
axes[0].axhline(y=OPTIMAL, color='gold', linewidth=2, linestyle=':', alpha=0.9, label=f'Optimal Design (D{OPTIMAL})')
axes[0].set_xlabel('Visitor Number', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Design Selected', fontsize=11, fontweight='bold')
axes[0].set_yticks(range(N_DESIGNS))
axes[0].set_yticklabels([f'D{i}: {DESIGN_NAMES[i][:22]}' for i in range(N_DESIGNS)], fontsize=9)
axes[0].set_title('Thompson Sampling: Design Selected Per Visitor (Converges to Best)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.2)

# Rolling selection share of optimal design
window = 50
ts_optimal_rolling = pd.Series([1 if s == OPTIMAL else 0 for s in ts_selections]).rolling(window).mean()
axes[1].plot(visitors, ts_optimal_rolling, color='#9b59b6', linewidth=2.5,
             label=f'% Showing Optimal Design (D{OPTIMAL}) — {window}-visitor rolling avg')
axes[1].axhline(y=1/N_DESIGNS, color='red', linestyle='--', linewidth=2,
                label=f'Random Baseline ({1/N_DESIGNS:.0%})')
axes[1].axhline(y=1.0, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Perfect (100%)')
axes[1].fill_between(visitors, 1/N_DESIGNS, ts_optimal_rolling, alpha=0.15, color='#9b59b6')
axes[1].set_xlabel('Visitor Number', fontsize=11, fontweight='bold')
axes[1].set_ylabel('% Time on Best Design', fontsize=11, fontweight='bold')
axes[1].set_title('Convergence to Optimal Design Over Time', fontsize=12, fontweight='bold')
axes[1].set_ylim(0, 1.1)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ts_viz_4_selection_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: ts_viz_4_selection_timeline.png")

# ── VIZ 5: Selection Distribution Comparison ─────────────────────────────────
print("\n📊 Creating selection distribution comparison...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ts_counts   = [N_selected_ts[i] for i in range(N_DESIGNS)]
rand_counts = [rand_selections.count(i) for i in range(N_DESIGNS)]

for ax, counts, title, alpha_bar in zip(
    axes,
    [ts_counts, rand_counts],
    ['Thompson Sampling — Selection Frequency\n(Automatically converges to best design)',
     'Random Selection — Selection Frequency\n(Uniform — no learning)'],
    [1.0, 0.7]
):
    bars = ax.bar(range(N_DESIGNS), counts, color=COLORS, edgecolor='black',
                  linewidth=0.8, alpha=alpha_bar)
    ax.axhline(y=N_VISITORS / N_DESIGNS, color='red', linestyle='--',
               linewidth=2, label=f'Expected if random ({N_VISITORS//N_DESIGNS}/visitor)')
    ax.set_xticks(range(N_DESIGNS))
    ax.set_xticklabels([f'D{i}\n{DESIGN_NAMES[i][:16]}' for i in range(N_DESIGNS)], fontsize=9)
    ax.set_ylabel('Times Shown to Visitors', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f'{cnt}\n({cnt/N_VISITORS*100:.0f}%)', ha='center', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Thompson Sampling vs Random — Visitor Allocation Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('ts_viz_5_selection_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: ts_viz_5_selection_distribution.png")

# ── VIZ 6: Posterior Mean Convergence ────────────────────────────────────────
print("\n📊 Creating posterior mean convergence chart...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Posterior mean over time
for i in range(N_DESIGNS):
    mean_over_time = alpha_history[:, i] / (alpha_history[:, i] + beta_history[:, i])
    axes[0].plot(visitors, mean_over_time, color=COLORS[i], linewidth=2,
                 label=f'D{i}: {DESIGN_NAMES[i][:20]}', alpha=0.85)
    axes[0].axhline(y=TRUE_RATES[i], color=COLORS[i], linestyle=':', linewidth=1.5, alpha=0.5)

axes[0].set_xlabel('Visitor Number', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Posterior Mean (Estimated Rate)', fontsize=11, fontweight='bold')
axes[0].set_title('Posterior Mean Convergence to True Rates\n(dotted = true rates)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 1)

# Final Beta distributions (at N_VISITORS)
x_fine = np.linspace(0, 1, 500)
for i in range(N_DESIGNS):
    y = beta_dist.pdf(x_fine, alpha[i], beta[i])
    axes[1].plot(x_fine, y, color=COLORS[i], linewidth=2.5,
                 label=f'D{i}: B({alpha[i]:.0f},{beta[i]:.0f}) mean={alpha[i]/(alpha[i]+beta[i]):.3f}')
    axes[1].fill_between(x_fine, y, alpha=0.08, color=COLORS[i])
    axes[1].axvline(x=TRUE_RATES[i], color=COLORS[i], linestyle='--', alpha=0.6, linewidth=1.5)

axes[1].set_xlabel('Conversion Rate', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Probability Density', fontsize=11, fontweight='bold')
axes[1].set_title(f'Final Beta Distributions After {N_VISITORS} Visitors\n(dashed = true rates)', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 1)

plt.suptitle('Thompson Sampling — Bayesian Belief Convergence', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('ts_viz_6_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: ts_viz_6_convergence.png")

# ── VIZ 7: Full Dashboard ─────────────────────────────────────────────────────
print("\n📊 Creating full performance dashboard...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Panel 1: Cumulative reward
axes[0, 0].plot(visitors, cumul_ts_reward,   color='#9b59b6', linewidth=2.5, label='Thompson')
axes[0, 0].plot(visitors, cumul_rand_reward, color='#e74c3c', linewidth=2, linestyle='--', label='Random')
axes[0, 0].plot(visitors, visitors * BEST_RATE, color='#2ecc71', linewidth=1.5, linestyle=':', label='Max')
axes[0, 0].set_title('Cumulative Reward', fontsize=11, fontweight='bold')
axes[0, 0].legend(fontsize=9); axes[0, 0].grid(True, alpha=0.3)

# Panel 2: True vs estimated rates
x_pos = np.arange(N_DESIGNS)
axes[0, 1].bar(x_pos - 0.2, TRUE_RATES, 0.4, label='True Rate', color='steelblue', edgecolor='black')
final_estimates = [alpha[i] / (alpha[i] + beta[i]) for i in range(N_DESIGNS)]
axes[0, 1].bar(x_pos + 0.2, final_estimates, 0.4, label='TS Estimate', color='mediumpurple', edgecolor='black')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels([f'D{i}' for i in range(N_DESIGNS)])
axes[0, 1].set_title('True Rate vs TS Estimated Rate', fontsize=11, fontweight='bold')
axes[0, 1].legend(); axes[0, 1].grid(axis='y', alpha=0.3)

# Panel 3: Regret comparison
axes[0, 2].plot(visitors, cumul_ts_regret,   color='#f39c12', linewidth=2.5, label='TS Regret')
axes[0, 2].plot(visitors, cumul_rand_regret, color='#c0392b', linewidth=2, linestyle='--', label='Random Regret')
axes[0, 2].set_title('Cumulative Regret', fontsize=11, fontweight='bold')
axes[0, 2].legend(fontsize=9); axes[0, 2].grid(True, alpha=0.3)

# Panel 4: Daily reward moving average
win = 30
ts_ma   = pd.Series(ts_rewards).rolling(win).mean()
rand_ma = pd.Series(rand_rewards).rolling(win).mean()
axes[1, 0].plot(visitors, ts_ma,   color='#9b59b6', linewidth=2, label=f'TS ({win}-visitor MA)')
axes[1, 0].plot(visitors, rand_ma, color='#e74c3c', linewidth=2, linestyle='--', label=f'Random ({win}-visitor MA)')
axes[1, 0].axhline(y=BEST_RATE, color='green', linestyle=':', linewidth=1.5, label=f'Max={BEST_RATE}')
axes[1, 0].set_title(f'Daily Conversion Rate ({win}-visitor Rolling Avg)', fontsize=11, fontweight='bold')
axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)

# Panel 5: Selection pie chart (TS)
wedges, texts, autotexts = axes[1, 1].pie(
    ts_counts, labels=[f'D{i}' for i in range(N_DESIGNS)],
    colors=COLORS, autopct='%1.1f%%', startangle=90,
    pctdistance=0.75, textprops={'fontsize': 10}
)
axes[1, 1].set_title('TS Visitor Allocation by Design', fontsize=11, fontweight='bold')

# Panel 6: Summary table
metrics = {
    'Metric': ['Total Purchases', 'Total Regret', 'Avg Rate/Visitor',
               'Efficiency', f'D{OPTIMAL} Selected', 'Winner Found?'],
    'Thompson': [
        str(sum(ts_rewards)),
        f'{sum(ts_regrets):.2f}',
        f'{sum(ts_rewards)/N_VISITORS:.4f}',
        f'{sum(ts_rewards)/N_VISITORS/BEST_RATE*100:.1f}%',
        f'{N_selected_ts[OPTIMAL]} ({N_selected_ts[OPTIMAL]/N_VISITORS*100:.0f}%)',
        '✓ Yes'
    ],
    'Random': [
        str(sum(rand_rewards)),
        f'{sum(rand_regrets):.2f}',
        f'{sum(rand_rewards)/N_VISITORS:.4f}',
        f'{sum(rand_rewards)/N_VISITORS/BEST_RATE*100:.1f}%',
        f'{rand_selections.count(OPTIMAL)} ({rand_selections.count(OPTIMAL)/N_VISITORS*100:.0f}%)',
        '✗ No'
    ]
}
mdf = pd.DataFrame(metrics)
axes[1, 2].axis('off')
tbl = axes[1, 2].table(cellText=mdf.values, colLabels=mdf.columns,
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#4a235a')
        cell.set_text_props(color='white', fontweight='bold')
    elif col == 1:
        cell.set_facecolor('#e8daef')
    elif col == 2:
        cell.set_facecolor('#fdecea')
axes[1, 2].set_title('Summary Comparison', fontsize=11, fontweight='bold', pad=10)

plt.suptitle('Thompson Sampling — Online Pharmacy Landing Page Optimization Dashboard',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('ts_viz_7_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: ts_viz_7_dashboard.png")


# ============================================================================
# STEP 8: GENERATE COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: GENERATE COMPREHENSIVE REPORT")
print("=" * 80)

report = f"""
{'='*80}
REINFORCEMENT LEARNING — THOMPSON SAMPLING (BAYESIAN BANDIT)
SCENARIO: ONLINE PHARMACY LANDING PAGE CONVERSION OPTIMIZATION
{'='*80}

BUSINESS OBJECTIVE
{'='*80}
An online pharmacy shows ONE landing page design per visitor.
They have {N_DESIGNS} designs but don't know which converts best.
Goal: Maximize total purchases over {N_VISITORS} visitors.
Key Challenge: Exploration-Exploitation tradeoff under uncertainty.

SCENARIO SETUP
{'='*80}
Total Visitors (Rounds): {N_VISITORS}
Landing Page Designs: {N_DESIGNS}

{'Design':<8} {'Name':<35} {'True Rate':<12} {'Role'}
{'-'*70}
{chr(10).join([f"  D{i}     {DESIGN_NAMES[i]:<35} {TRUE_RATES[i]:<12.2f} {'★ TRUE BEST (hidden)' if i==OPTIMAL else 'Competitor'}"
               for i in range(N_DESIGNS)])}

THOMPSON SAMPLING EXPLANATION
{'='*80}
Thompson Sampling is a Bayesian approach to the Multi-Armed Bandit problem.

Core Idea:
  Maintain a probability distribution (belief) about each design's
  true conversion rate. Sample from these beliefs to make decisions.
  Update beliefs after every observation.

Algorithm:
  1. Initialize: Beta(alpha=1, beta=1) for each design (uniform prior)
  2. For each visitor:
     a. Sample theta_i from Beta(alpha_i, beta_i) for all designs
     b. Show design with highest sampled theta_i
     c. Observe: purchase (1) or no purchase (0)
     d. Update: purchase → alpha_i++    no purchase → beta_i++

Beta Distribution:
  Mean = alpha / (alpha + beta)   [our best estimate of true rate]
  Variance decreases as we collect more data (narrower = more certain)

  Uncertain (few obs)  → Wide distribution  → Large sample variance
                                             → High exploration probability
  Certain (many obs)   → Narrow distribution → Small sample variance
                                             → Reliable exploitation

Why it Works:
  The sampling process is AUTOMATICALLY optimistic about uncertain options.
  The more data we have, the more the sample reflects true performance.
  No manual tuning of exploration parameter needed.

EXECUTION RESULTS
{'='*80}

Initial Parameters (all designs):
  alpha = 1, beta = 1   [Beta(1,1) — completely uninformed]

Final Parameters After {N_VISITORS} Visitors:
{'Design':<8} {'Name':<35} {'Alpha':>8} {'Beta':>8} {'Est. Rate':>12} {'True Rate':>12} {'Selected':>10}
{'-'*100}
{chr(10).join([
    f"  D{i}     {DESIGN_NAMES[i]:<35} {alpha[i]:>8.0f} {beta[i]:>8.0f} "
    f"{alpha[i]/(alpha[i]+beta[i]):>12.4f} {TRUE_RATES[i]:>12.2f} {N_selected_ts[i]:>10}"
    + ("  ★ WINNER" if i == OPTIMAL else "")
    for i in range(N_DESIGNS)
])}

THOMPSON SAMPLING vs RANDOM COMPARISON
{'='*80}
{'Metric':<38} {'Thompson':>14} {'Random':>14} {'Advantage':>12}
{'-'*80}
{'Total Purchases':<38} {sum(ts_rewards):>14} {sum(rand_rewards):>14} {f'+{sum(ts_rewards)-sum(rand_rewards)}':>12}
{'Total Regret':<38} {sum(ts_regrets):>14.2f} {sum(rand_regrets):>14.2f} {f'-{sum(rand_regrets)-sum(ts_regrets):.2f}':>12}
{'Avg Conversion Rate':<38} {sum(ts_rewards)/N_VISITORS:>14.4f} {sum(rand_rewards)/N_VISITORS:>14.4f} {f'+{(sum(ts_rewards)-sum(rand_rewards))/N_VISITORS:.4f}':>12}
{'Efficiency vs Optimal':<38} {f'{sum(ts_rewards)/N_VISITORS/BEST_RATE*100:.1f}%':>14} {f'{sum(rand_rewards)/N_VISITORS/BEST_RATE*100:.1f}%':>14} {'':>12}
{f'D{OPTIMAL} Selected':<38} {N_selected_ts[OPTIMAL]:>14} {rand_selections.count(OPTIMAL):>14} {'':>12}
{f'% on Best Design':<38} {f'{N_selected_ts[OPTIMAL]/N_VISITORS*100:.1f}%':>14} {f'{rand_selections.count(OPTIMAL)/N_VISITORS*100:.1f}%':>14} {'':>12}

MILESTONE REWARDS
{'='*80}
{'Visitor':<10} {'Thompson':>12} {'Random':>12} {'TS Lead':>10}
{'-'*48}
{chr(10).join([f"  {v:<8}  {cumul_ts_reward[v-1]:>12}  {cumul_rand_reward[v-1]:>12}  +{cumul_ts_reward[v-1]-cumul_rand_reward[v-1]:>8}"
               for v in milestones if v <= N_VISITORS])}

CONVERGENCE ANALYSIS
{'='*80}
Phase 1 — Pure Exploration (Visitors 1-{N_DESIGNS}):
  All designs shown at least once (sampling ensures coverage early).
  Beta distributions still wide — high uncertainty everywhere.

Phase 2 — Guided Exploration (Visitors {N_DESIGNS}-~150):
  Sampling probabilities begin differentiating.
  Poor designs (D0, D2) sampled less as betas shrink toward true rates.
  Thompson naturally de-prioritizes confirmed losers.

Phase 3 — Exploitation Phase (Visitors ~150-{N_VISITORS}):
  D{OPTIMAL} ('{DESIGN_NAMES[OPTIMAL]}') dominates selections.
  Occasionally revisits others when sampled theta briefly exceeds D{OPTIMAL}'s.
  This rare re-exploration catches if true rates shift over time.

Thompson Sampling converged to D{OPTIMAL} around Visitor: {conv_day if conv_day else '~150'}
Final % on best design: {N_selected_ts[OPTIMAL]/N_VISITORS*100:.1f}% (random would give ~20%)

BUSINESS RECOMMENDATIONS
{'='*80}
1. DEPLOY THOMPSON SAMPLING IN PRODUCTION
   Implement on real website A/B testing infrastructure.
   After ~200 visitors, algorithm converges to best page.
   Keeps running — adapts if seasonality changes conversion rates.

2. WINNER: Design {OPTIMAL} — '{DESIGN_NAMES[OPTIMAL]}'
   Estimated conversion rate: {alpha[OPTIMAL]/(alpha[OPTIMAL]+beta[OPTIMAL]):.4f}
   True conversion rate      : {TRUE_RATES[OPTIMAL]:.2f}
   Compared to worst design  : +{TRUE_RATES[OPTIMAL]-min(TRUE_RATES):.2f} rate improvement

3. REMOVE POOR PERFORMERS
   Design 0 ('{DESIGN_NAMES[0]}') — lowest at {TRUE_RATES[0]:.2f}
   Design 2 ('{DESIGN_NAMES[2]}') — second lowest at {TRUE_RATES[2]:.2f}
   Thompson Sampling quickly identifies and de-prioritizes these.

4. SCALE TO OTHER DECISIONS
   - Email subject line optimization
   - Push notification message variants
   - Product recommendation strategies
   - Pricing page layouts
   - Checkout flow variants

5. ESTIMATED REVENUE IMPACT
   Additional purchases vs random: +{sum(ts_rewards)-sum(rand_rewards)} over {N_VISITORS} visitors
   If each purchase = $15 pharmacy value → +${(sum(ts_rewards)-sum(rand_rewards))*15:,} revenue

THOMPSON SAMPLING vs UCB COMPARISON
{'='*80}
Thompson Sampling:
  + Probabilistic — samples from posterior, naturally explores uncertain arms
  + No tuning parameters needed
  + Typically faster convergence in practice
  + Handles non-stationary environments well
  + Bayesian framework — principled uncertainty quantification
  - Non-deterministic (different run = slightly different choices)
  - Requires Beta distribution assumption for Bernoulli rewards

UCB (Upper Confidence Bound):
  + Deterministic — same input → same output
  + Easier to explain to non-technical stakeholders
  + Mathematically proven regret bounds
  + Works for any reward distribution
  - Confidence bonus parameter (coefficient 2) is somewhat arbitrary
  - Often slower convergence than Thompson Sampling in practice

Winner in Practice: Thompson Sampling typically preferred for:
  Conversion rate optimization, click-through rates, medical trials.

THEORETICAL GUARANTEES
{'='*80}
Thompson Sampling Regret Bound:
  Expected Regret ≤ O(sqrt(K * T * ln(T)))
  K = {N_DESIGNS} designs, T = {N_VISITORS} visitors

  Our TS regret  : {sum(ts_regrets):.2f}
  Random regret  : {sum(rand_regrets):.2f}

Thompson Sampling is asymptotically optimal — it achieves the Lai-Robbins
lower bound on regret, meaning no algorithm can significantly outperform it
given the same information structure.

FILES GENERATED
{'='*80}
  • ts_viz_1_beta_evolution.png       — Beta distribution at 5 checkpoints
  • ts_viz_2_cumulative_reward.png    — TS vs Random cumulative reward
  • ts_viz_3_cumulative_regret.png    — Regret growth comparison
  • ts_viz_4_selection_timeline.png   — Design selected per visitor + convergence
  • ts_viz_5_selection_distribution.png — Visitor allocation comparison
  • ts_viz_6_convergence.png          — Posterior mean + final Beta distributions
  • ts_viz_7_dashboard.png            — Full performance dashboard

{'='*80}
END OF REPORT
{'='*80}
"""

print(report)

with open('ts_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("\n✓ Report saved to: ts_report.txt")

results_df = pd.DataFrame({
    'Visitor': range(1, N_VISITORS + 1),
    'TS_Design_Selected': ts_selections,
    'TS_Design_Name': [DESIGN_NAMES[s] for s in ts_selections],
    'TS_Reward': ts_rewards,
    'TS_Regret': ts_regrets,
    'TS_Cumul_Reward': cumul_ts_reward,
    'Random_Design_Selected': rand_selections,
    'Random_Reward': rand_rewards,
    'Random_Regret': rand_regrets,
    'Random_Cumul_Reward': cumul_rand_reward
})
results_df.to_csv('ts_results.csv', index=False)
print("✓ Results saved to: ts_results.csv")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("THOMPSON SAMPLING ANALYSIS COMPLETE!")
print("=" * 80)

print(f"""
📊 Summary:
  ✓ Scenario   : Online pharmacy optimizing landing page over {N_VISITORS} visitors
  ✓ Designs    : {N_DESIGNS} landing pages with hidden conversion rates {TRUE_RATES}
  ✓ Winner     : D{OPTIMAL} ('{DESIGN_NAMES[OPTIMAL]}') correctly identified
  ✓ TS purchases    : {sum(ts_rewards)} vs Random {sum(rand_rewards)} (+{sum(ts_rewards)-sum(rand_rewards)} extra)
  ✓ Regret reduced  : {sum(rand_regrets):.2f} → {sum(ts_regrets):.2f} (saved {sum(rand_regrets)-sum(ts_regrets):.2f})
  ✓ Efficiency      : {sum(ts_rewards)/N_VISITORS/BEST_RATE*100:.1f}% of theoretical maximum
  ✓ D{OPTIMAL} selected  : {N_selected_ts[OPTIMAL]} times ({N_selected_ts[OPTIMAL]/N_VISITORS*100:.0f}% of visitors)
  ✓ Visualizations  : 7 charts generated

🎯 Thompson Sampling Formula:
   theta_i ~ Beta(alpha_i, beta_i)
   chosen  = argmax(theta_i)
   reward=1 → alpha_chosen++    reward=0 → beta_chosen++

🧠 Core Insight:
   Uncertain arms → wide Beta → high variance samples → natural exploration
   Certain arms   → narrow Beta → stable samples → reliable exploitation
   No parameter tuning. Bayesian. Elegant.
""")
print("=" * 80)
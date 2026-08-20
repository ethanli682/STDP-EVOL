# Reward design references for cogames

Working notes for the discussion: how published multi-agent systems with mining / resource transfer / role specialization actually shape reward, and what each choice implies for our `composite_v4_role_gated` setup.

Each entry: **Reward formulation** (what they actually optimize) → **Setup / reasoning** (why that choice) → **Discussion hooks for our case** (what to argue about, what to try).

---

## 1. AlphaStar — DeepMind, *Nature* 2019

**Source:** [Nature paper](https://www.nature.com/articles/s41586-019-1724-z) · [supplementary PDF](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf) · [explainer](https://cyk1337.github.io/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/)

### Reward formulation
- **Terminal reward**: ±1 on win/loss (sparse).
- **Pseudo-rewards** conditioned on a strategy statistic `z` sampled from human replays:
  - `r_build_order` = negative edit distance between executed and target build order
  - `r_cumulative_stats` = negative Hamming distance between executed and target unit/structure counts
- **Game-score signals** available from the engine (used as auxiliary supervision in some heads): minerals collected, **mineral collection rate**, vespene collected, total value of units/structures, total destroyed value, idle worker time, total spent minerals.
- Each pseudo-reward channel feeds **its own value head**; they are not pre-summed into one scalar.

### Setup / reasoning
- Win/loss alone is too sparse for credit assignment over 10k+ step games.
- Imitation-learning prior makes `z` meaningful: pseudo-rewards keep the agent near a *plausible* strategy manifold.
- Multiple value heads let the optimizer attribute improvement to specific axes (economy vs. army vs. tech).

### Known failure mode (documented)
Edit-distance pseudo-rewards get **hacked**: agent produces the easy units in `z` and skips the critical hard ones. They needed league play + opponent diversity to suppress this.

### Discussion hooks for cogames
- Are we collapsing orthogonal signals (mining rate, transfer count, survival, role compliance) into one scalar in `composite_v4_role_gated`? If yes, that explains why CMA-ES sees a flat noisy landscape.
- Do we have anything analogous to `z` — a "what a good run should look like" template? E.g. expected mineral count by step T.
- Are any of our gating terms hackable (satisfiable by trivial behavior that doesn't advance the task)?

---

## 2. Craftax-Coop / Multi-Agent Craftax — *arXiv 2025*

**Source:** [arXiv paper](https://arxiv.org/html/2511.04904v1) · [code](https://github.com/ericyuxuanye/Multi-Agent-Craftax)

This is the closest analogue to cogames in published literature: heterogeneous roles, mining, resource transfer between agents.

### Reward formulation
- **Per-achievement reward**, fired the **first time per episode** each agent completes an achievement (chop wood, mine stone, smelt iron, craft pickaxe, drink water, kill zombie, …).
- **Trading is *not* directly rewarded.** Agents can trade base materials (ores, wood, stone) and consumables (food, water). Reward only comes from downstream achievements those traded resources enable.
- 9 progression levels; baselines reach only level 3.

### Roles (hard ability gates, not just reward gates)
- **Miner**: exclusively mines ores/stone, exclusively crafts pickaxes and torches, places stones for shelter.
- **Forager**: gathers food/water, hunts passive mobs, plants/harvests crops; larger food/water storage.
- **Warrior**: 2× base damage, exclusively crafts advanced swords, collects bows / crafts arrows.

### Setup / reasoning
- Once-per-episode firing prevents reward farming (no infinite loop of mining the same stone).
- Hard role abilities mean cooperation is *forced* — a Warrior literally cannot mine ore, so trade must happen for tech progression.
- Designers deliberately did *not* reward trades to avoid reward hacking (handing the same item back and forth).

### Documented failure modes
- **Credit assignment**: agents die from thirst/hunger many steps after the gathering decision that should have prevented it.
- **Exploration**: <2% of episodes reach level 3 even after 1B steps.
- **Cooperation**: MAPPO trades inconsistently; PQN shows zero meaningful resource sharing.
- Authors' verdict: PPO/MAPPO/QMIX **all fail** on Craftax-Coop without curriculum or intrinsic motivation.

### Discussion hooks for cogames
- Are our roles **gated by ability** or only by reward shaping? If only by shaping, agents can ignore the role and still get reward — which is what we'd see as noise.
- Do we reward the transfer act itself? Craftax-Coop deliberately doesn't — but they have downstream achievements to backstop. Do we?
- "Once per episode" fits our `ep_reward_mean ≈ 0.05` regime well: the reward should be a small set of meaningful event flags, not a continuous shaping signal.
- Their negative result is load-bearing: if standard MARL fails on the closest published benchmark, our flat fitness curve is the **expected baseline outcome** absent curriculum / intrinsic motivation.

---

## 3. SMAC — StarCraft Multi-Agent Challenge

**Source:** [GitHub](https://github.com/oxwhirl/smac)

### Reward formulation
Dense + sparse, summed:
- **Dense**: damage dealt − damage taken per step, plus a fixed bonus per enemy killed.
- **Sparse**: large terminal bonus on win.
- Both normalized so the dense term doesn't dominate in long episodes.

### Setup / reasoning
- Combat-only domain — no economy, no resource transfer.
- Dense shaping keeps gradient alive when win/loss is rare; sparse term anchors the actual objective.
- Reward is **global** (team-shared); credit assignment handled at the algorithm level (QMIX, COMA), not the reward.

### Discussion hooks for cogames
- The dense+sparse split is the simplest decomposition. Do we have a clean dense signal at all? `ep_reward_mean ≈ 0.05` suggests our "dense" signal is just shaping noise.
- SMAC works with global reward only because credit assignment is moved into the value function. If we use a flat-summed scalar with no value-decomposition method, we get the worst of both.

---

## 4. MATE — Multi-Agent Tracking Environment

**Source:** [OpenReview](https://openreview.net/pdf?id=SyoUVEyzJbE)

### Reward formulation
Target team reward: **`r(T) = F + B`**
- **F (Freight)**: fixed sparse reward on successful cargo delivery.
- **B (Bounty)**: dense per-cargo carrying reward.

### Setup / reasoning
- Explicit decomposition into "what we want" (delivery = F) and "how we keep gradient flowing" (bounty = B).
- Bounty value tuned so that pure greedy bounty-farming is dominated by actually-delivering.

### Discussion hooks for cogames
- Direct template for us: what is our `F`? (e.g. minerals delivered to base, technology tier reached.) What is our `B`? (e.g. minerals in inventory, distance to mine.)
- Does our composite have a clear dominance ordering between `F`-style and `B`-style terms, or are they on comparable scales (which would let bounty-farming win)?

---

## 5. Role-Specific Reward Design with LLM (StarCraft II, IEEE 2025)

**Source:** [IEEE Xplore PDF](https://ieeexplore.ieee.org/iel8/10887540/10887541/10890857.pdf)

### Reward formulation
- LLM generates per-role reward functions from a natural-language description of the role's responsibility.
- Each role gets a hand-tailored shaping term (e.g. "Tank: minimize teammate damage taken within 5 units").
- Per-role reward replaces a single global shaping term.

### Setup / reasoning
- Manual reward engineering across many roles is brittle; LLM-generated shaping searches the space faster.
- Reward functions are auditable Python, so failures are debuggable.

### Discussion hooks for cogames
- Our `composite_v4_role_gated` *is* the per-role-shaping pattern. Are the per-role weights learned, hand-tuned, or fixed? If fixed, we're optimizing in a brittle slice of the design space.
- Could we have an LLM (or just us) generate role-specific reward terms separately and ablate each?

---

## 6. Reward Decomposition: Lazy vs. Selfish — survey

**Source:** [Reward Design in Cooperative MARL for Packet Routing](https://openreview.net/forum?id=r15kjpHa-) · [Survey: communication in MARL](https://arxiv.org/pdf/2203.08975)

### Reward formulation options (and their pathologies)
| Scheme | Form | Pathology |
|---|---|---|
| **Pure global** | `r_i = R_team` ∀ i | Lazy agents, free-riders |
| **Pure local** | `r_i = f_i(s, a_i)` | Selfish agents, no transfer behavior |
| **Mixed** | `r_i = α·R_team + (1−α)·r_local` | Weight `α` is hard to tune |
| **Difference reward (COMA)** | `r_i = R_team − R_team(a_-i, a_i = noop)` | Counterfactual is expensive to compute |
| **Value decomposition (VDN/QMIX)** | `Q_team = Σ Q_i` | Reward stays global, credit assignment moves into Q |

### Discussion hooks for cogames
- Which scheme is `composite_v4_role_gated` actually implementing? It looks mixed — but with what `α`?
- If we want resource transfer between agents, pure global reward is fragile (transferer gets nothing locally) and pure local is fatal (no incentive to transfer). Mixed or difference-reward is the live option.
- COMA-style counterfactual rewards are tractable here because we can replay an episode with one agent's actions noop'd — worth prototyping.

---

## 7. MAPPO — *NeurIPS 2021*

**Source:** [BAIR blog](https://bair.berkeley.edu/blog/2021/07/14/mappo/) · [arXiv](https://ar5iv.labs.arxiv.org/html/2103.01955)

### Reward formulation
Uses whatever the environment provides — paper's contribution is on the algorithm side (centralized critic, decentralized actor), not reward design.

### Setup / reasoning
Argues the surprising result that **vanilla PPO with a centralized value function** matches or beats specialized MARL algorithms on SMAC and other benchmarks, **provided the reward signal is dense enough**.

### Discussion hooks for cogames
- The MAPPO result is a precondition check: if the reward is dense and informative, simple algorithms work. If a fancy MARL algorithm doesn't beat MAPPO, the bottleneck is the reward, not the algorithm. Same logic applies here for CMA-ES — we shouldn't blame the optimizer until we've validated the signal.

---

## Synthesis — candidate experiments for our setup

Ordered roughly by effort / blast radius:

1. **Decompose the logged fitness.** Log each component of `composite_v4_role_gated` separately per candidate and per role. Check whether any single component is moving under selection — if not, the signal is noise.
2. **Run a random-search baseline** at the same wallclock as CMA-ES. If CMA-ES does not beat it, we've confirmed the optimizer is ranking on noise.
3. **Try Craftax-Coop's "achievement once per episode"** pattern: identify a small set of meaningful one-shot events (first ore mined, first transfer, first tech crafted) and reward only those. Drop continuous shaping.
4. **Try the MATE F+B decomposition**: pick one delivery-style sparse reward and one inventory-style dense reward, weighted so dense cannot dominate. Optimize the sum *and* log them separately.
5. **Verify role gating is by ability, not just by reward.** If a Warrior can mine and get partial reward for it, the role gate is leaky and the agent will ignore the role.
6. **Consider COMA-style difference rewards** for the transfer behavior specifically: hard to reward locally, easy to compute counterfactually.

---

## Open questions to settle before changing reward

- What does `composite_v4_role_gated` actually compute, term by term?
- What is `ep_reward_mean` measuring — raw env reward, shaped, or the composite?
- How many distinct events fire in a typical episode? (If <10, the reward is effectively sparse and we should treat it as such.)
- Are seeds shared across candidates within an iteration? (Common-random-numbers vs. independent noise changes how many evals we need.)

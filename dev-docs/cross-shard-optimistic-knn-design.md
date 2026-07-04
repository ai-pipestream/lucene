<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# Cross-Shard Optimistic kNN — design

**Status:** draft v1 · **Date:** 2026-07-04
**Branch:** `feature/cross-shard-optimistic-knn` on `ai-pipestream/lucene` (cut from `apache/main` @ `8fe701c0873`)
**Worktree:** `/work/worktrees/lucene-cross-shard-optimistic`
**Supersedes:** the `CollaborativeKnnCollector` approach in apache/lucene#15676 (to be closed with a forward link once this lands as a new PR).

---

## 1. Problem statement

Sharded kNN today has a wasteful contract: **every shard returns a full top-k and the
coordinator reranks `k × shards` candidates to keep `k`.** At `k=10,000` over 16 shards that
is 160,000 candidates fetched, scored, serialized, and merged to keep 6% of them — and each
shard paid the full `ef=k` graph exploration to produce its discarded 94%.

Lucene already eliminated this waste *within* an index. The optimistic collection strategy
(#14226, #15063) searches each segment at a reduced `perLeafTopK` — the statistically
expected contribution of that segment to the global top-k, plus 3σ of a binomial — and then
re-enters only the segments whose results prove globally competitive:

```287:291:lucene/core/src/java/org/apache/lucene/search/AbstractKnnVectorQuery.java
  private static int perLeafTopKCalculation(int k, float leafProportion) {
    return (int)
        Math.max(
            1, k * leafProportion + LAMBDA * Math.sqrt(k * leafProportion * (1 - leafProportion)));
  }
```

Nothing in that logic is segment-specific. If shards are approximately random samples of the
corpus (the same assumption optimistic search already makes for segments), each of 16 shards
owes ~`k/16 + λσ` results, not `k`. For `k=10,000 / 16 shards` that is ~1,000 per shard
instead of 10,000 — a ~10× reduction in per-shard `ef`, result fan-in, and rerank cost
(§10.1), *before* any early-termination cleverness.

**Goal:** extend the optimistic contract across shards, with a shared, monotonically-rising
global floor (the merged k-th-best-so-far) replacing the blind "return everything and let the
coordinator sort it out" protocol. Deliver:

1. **Core Lucene PR** — a floor-aware collector that composes with the existing optimistic
   path across *leaves* (single machine, provable visited/recall Pareto win in luceneutil).
2. **Engine layer** (pipestream, out of Lucene scope) — the same floor object fed over gRPC
   across *shards*, plus an optional quantized in-memory **scout** that seeds the floor
   before any real shard has converged.

---

## 2. Post-mortem: what #15676 got wrong (and right)

Measured on the 247M-vector index (16 segments, K=100): stock 0.913 recall @ 41,749
visits/query; collaborative 0.561 sequential / 0.658 parallel. The old design doc blamed the
floor *value* (max of per-segment local k-th-bests instead of merged global k-th-best). That
analysis is wrong, and the correction drives this design:

- **Both floors are valid lower bounds of the final cutoff.** A segment's local k-th-best is
  the k-th best of a subset of all hits, so it never exceeds the final merged k-th-best.
  Switching to the merged-kth floor makes the bar *tighter* (merged-kth ≥ max of local kths),
  i.e. more aggressive — it cannot fix over-pruning.
- **In sequential execution the two floors are identical at the moment of damage:** after
  segment 1 completes, merged-kth of its published hits *is* its local k-th-best. Segments
  2..16 face the same guillotine either way.
- **The real defect is twofold:**
  1. **Comparison quantity.** `earlyTerminated()` compared the segment's best *found* score
     against the floor. During HNSW ascent the best-found is garbage; a converged segment's
     floor kills every still-climbing segment at `minVisits`.
  2. **No ascent protection.** *Any* global bar — found-based or frontier-based — applied to
     a segment that has not yet reached its query neighborhood terminates it at the entry
     point. The distributed Pi-cluster runs held 90% recall not because of the merged-heap
     floor but because of two confounders: 5× over-fetch (shards explored to ef=500) and
     network latency (the floor physically could not arrive during a shard's ascent).

What #15676 got right and we keep: the floor is a single monotonic scalar; sharing it is
cheap; `KnnCollectorManager` is the correct seam ("share global state across leaves" is its
documented purpose); and the visited/recall Pareto curve is the only honest metric
(benwtrent, first comment on the PR).

---

## 3. Prior art in-tree (use it, don't reinvent it)

| Mechanism | Where | What we take |
|---|---|---|
| **Optimistic collection** (#14226, #15063, #16324) | `AbstractKnnVectorQuery.OptimisticKnnCollectorManager` | `perLeafTopK` scaling, `isOptimistic()`/`newOptimisticCollector()` seam, re-entry phase. We compose with it, never bypass it. |
| **`MultiLeafKnnCollector` + `BlockingFloatHeap`** (removed by #15686, resurrectable at `75d47f09c45^`) | git history | The whole collector shape: global size-k heap folded into `minCompetitiveSimilarity()`, **local-k gate** (`kResultsCollected`), **greediness clamp** (non-competitive local queue of size `(1-g)·k`), **interval batching** (sync every 256 visits). Removal reasons we must answer: superseded by optimistic (→ we compose instead of competing) and non-deterministic results (→ §7). |
| **`MaxScoreAccumulator`** | `o.a.l.search` | Interval-batched (`0x3ff`) shared max with encoded score+doc, the concurrency pattern for the floor accumulator. |
| **`HnswQueueSaturationCollector` (patience)** | `o.a.l.search` | Existing single-JVM visit reducer; benchmark arm and a saturation signal we may later reuse as an alternative ascent gate. |
| **`KnnSearchStrategy.Seeded`** | `o.a.l.search.knn` | Entry-point seeding; future synergy with the scout (scout hits → seeds), out of scope for PR 1. |

Key consequence: **the core searcher (`HnswGraphSearcher`) needs zero changes.** The search
loop already re-derives `minAcceptedSimilarity` from `results.minCompetitiveSimilarity()`
after every accepted hit; a decorator that folds the global floor into that method lifts the
termination bar through existing plumbing. (The old branch's extra per-iteration re-read of
the collector bar — the hook msokolov flagged as dead code — is not needed and is dropped.)

---

## 4. Design — core Lucene

Two new classes in `o.a.l.search.knn`, both `@lucene.experimental`, plus zero changes to
`HnswGraphSearcher` and `AbstractKnnVectorQuery`:

### 4.1 `GlobalKnnFloor` — the shared per-query state

```java
/** Shared, monotonically tightening lower bound on the global top-k cutoff for one query.
 *  Thread-safe. Feedable from leaves in-process, or externally (remote shards, a scout). */
public final class GlobalKnnFloor {
  private final BlockingFloatHeap heap;       // size k: k best scores observed anywhere
  private volatile float floor = Float.NEGATIVE_INFINITY; // heap.peek() once full, else -inf

  public GlobalKnnFloor(int k) { ... }

  /** Batch-offer locally observed scores (ascending). Returns the current floor. */
  public float offer(float[] scores, int len) { ... }

  /** External feed, called from transport threads (gRPC handlers, scout callback).
   *  Monotonic max: values at or below the current floor are ignored, so duplicated
   *  or reordered remote deliveries need no special handling. */
  public void advertise(float kthBestLowerBound) { ... }

  /** Lock-free read of the current floor; NEGATIVE_INFINITY until k scores observed
   *  and nothing has been advertised. Never decreases. */
  public float floor() { ... }
}
```

**Why `advertise` carries no source id (implementation decision):** for scalar lower bounds
reduced by `max`, per-source slots are mathematically redundant — the max of everything ever
received equals the max over per-source maxima, and monotonicity makes conflating senders
harmless. Shard identity therefore stays at the transport layer (the coordinator already knows
which stream each hit came from), and a keyed API becomes necessary only if richer per-shard
summaries (top-j score vectors) are ever exchanged, which would be a new method rather than a
retrofit.

- Size-k min-heap of the best scores observed so far, across all feeders. Its min, once the
  heap is full, is a **valid lower bound of the final merged k-th-best** (subset argument,
  §2) — the invariant every safety claim rests on.
- `advertise()` is the entire distributed/scout surface area in core: a remote feeder pushes
  a *lower bound* scalar. Transport, calibration, topology live outside Lucene.
- **Remote update model.** Shards run in separate JVMs, so floor updates arrive on transport
  threads while search threads are mid-traversal. Safety comes from monotonicity, not
  locking discipline: each source's floor only rises and the reduction is `max`, so
  `advertise` is idempotent and commutative — safe under out-of-order or duplicated delivery
  with no sequence numbers, acks, or per-source bookkeeping (see the §4.1 note on why source
  ids are unnecessary for scalar bounds). Search threads observe updates via the collector's
  cached floor at their next sync boundary (≤ 256 visits later); staleness can only *delay*
  pruning, never make it unsafe (the floor is a lower bound of the final cutoff at all
  times). Source identity matters exactly at merge points: in star mode the coordinator is
  the merge point and already keys hits by stream; a future mesh mode reducing per-shard
  k-th-bests with `max()` is valid but looser than a true merged k-th
  (`max(local kths) ≤ merged kth`), so mesh-quality floors would need richer per-shard
  summaries (top-j score vectors) via a new, keyed method.
- Resurrect `BlockingFloatHeap` from `75d47f09c45^` (it has tests) rather than rewriting.
- Score-only semantics; termination uses strict `<` so score-ties are never pruned
  (tie-break on doc id then cannot drop a would-be winner; cheaper than encoding doc ids
  into the floor and consistent with `TopDocs.merge` keeping ties).

### 4.2 `FloorAwareKnnCollector` — the decorator

```java
public final class FloorAwareKnnCollector extends KnnCollector.Decorator {
  // all knobs package-private constants for PR 1; constructor params if reviewers want them
  private static final float GREEDINESS = 0.5f;   // conservative default, see note below
  private static final int   SYNC_INTERVAL = 0xff;

  private final GlobalKnnFloor globalFloor;
  private final FloatHeap nonCompetitiveQueue;    // size max(1, (1-g)*k): reduced local bar
  private final FloatHeap updatesQueue;           // local scores pending batch-offer
  private boolean kResultsCollected;              // the ascent gate
  private float cachedGlobalFloor = Float.NEGATIVE_INFINITY;

  @Override public boolean collect(int docId, float similarity) {
    boolean collected = super.collect(docId, similarity);
    nonCompetitiveQueue.offer(similarity);
    updatesQueue.offer(similarity);
    if (!kResultsCollected && numCollected() == k()) kResultsCollected = true;
    if (kResultsCollected && (justFilled || (visitedCount() & SYNC_INTERVAL) == 0)) {
      cachedGlobalFloor = globalFloor.offer(drainSorted(updatesQueue)); // batched, amortized
    }
    return collected;
  }

  @Override public float minCompetitiveSimilarity() {
    if (!kResultsCollected) return super.minCompetitiveSimilarity(); // ascent: local bar only
    return Math.max(
        super.minCompetitiveSimilarity(),
        Math.min(nonCompetitiveQueue.peek(), cachedGlobalFloor));    // greediness clamp
  }
}
```

Three load-bearing decisions, each fixing a measured failure of #15676:

1. **Ascent gate (local-k, not global-k).** No global influence until *this leaf* has
   collected its own k hits — i.e. it has provably escaped entry-point ascent. This replaces
   `minVisits=100` (a constant that cannot be right for both a 73k and a 247M index) and is
   the opposite polarity of the old "no floor until k *global* hits" gate, which does nothing
   in sequential execution.
2. **Greediness clamp instead of hard-stop.** A globally non-competitive leaf is not
   guillotined; its bar is lifted to `min(local non-competitive queue top, global floor)`, so
   it keeps exploring a thin frontier that preserves bridge paths. `g=0.9` would mean it retains a
   `(1-g)·k`-deep escape hatch. This subsumes the old `slack`/found-vs-reachable debate: the
   bar feeds the searcher's *existing* frontier-based break condition
   (`candidates.topScore() < minAcceptedSimilarity`), so the comparison quantity is the
   reachable bound by construction.
3. **Interval batching.** Local scores accumulate in a local heap and sync to the shared
   floor every 256 visits (plus once at fill). One volatile read between syncs. This is the
   answer to the memory-bus contention #15676 reported, and removes any need for
   striped/tree floors at realistic core counts.

### 4.3 `SharedFloorKnnCollectorManager` — composition with optimistic

```java
public final class SharedFloorKnnCollectorManager implements KnnCollectorManager {
  public SharedFloorKnnCollectorManager(int k) { ... }  // owns its floor (in-process queries)
  public SharedFloorKnnCollectorManager(int k, GlobalKnnFloor floor) { ... }  // externally fed
  public SharedFloorKnnCollectorManager(int k, GlobalKnnFloor floor, float greediness) { ... }

  public GlobalKnnFloor getGlobalFloor() { ... }  // for callers that feed or read the floor

  @Override public KnnCollector newCollector(int visitedLimit, KnnSearchStrategy s, LeafReaderContext ctx) {
    return new FloorAwareKnnCollector(new TopKnnCollector(k, visitedLimit, s), globalFloor, greediness);
  }
  @Override public KnnCollector newOptimisticCollector(int visitedLimit, KnnSearchStrategy s,
                                                       LeafReaderContext ctx, int perLeafK) {
    // gate fills at perLeafK (the leaf's owed contribution); the floor heap stays size k
    return new FloorAwareKnnCollector(new TopKnnCollector(perLeafK, visitedLimit, s), globalFloor, greediness);
  }
  @Override public boolean isOptimistic() { return true; }
}
```

One manager (and one floor) per query execution — both carry single-query state. Harness
usage: subclass `KnnFloatVectorQuery`, create the manager in the constructor, return it from
`getKnnCollectorManager` (it is called for both collection passes, which is what lets phase 2
start from the phase-1 floor).

- `isOptimistic() == true` means `AbstractKnnVectorQuery` gives every leaf the reduced
  `perLeafTopK` in phase 1 and runs its normal re-entry phase — **we inherit the optimistic
  baseline instead of competing against it** (the #15676 benchmark compared
  collaborative-without-optimistic vs stock-with-optimistic; this design cannot repeat that
  mistake because composition is structural).
- The floor keeps working during re-entry (phase-2 collectors share the same
  `GlobalKnnFloor`, already warm from phase 1), so re-entered leaves terminate as soon as
  they stop being competitive rather than re-earning full k. This is the expected source of
  the single-machine Pareto win: *optimistic search with a cheaper second phase.*
- PR 1 does **not** change the re-entry policy itself. A follow-up can use the floor to skip
  re-entry entirely when phase-1 bars prove completeness; keep it out of the first diff.

### 4.4 What the single-machine win looks like (expectation setting)

With the ascent gate, each leaf still pays entry-point descent + heap fill (`perLeafTopK`
collects). The floor can only shave the post-fill refinement tail and the re-entry phase.
That is the honest budget: meaningful (refinement + re-entry dominate at large k and many
leaves) but not the 50% seen on the Pi cluster, where over-fetch and latency changed the
regime. The acceptance bar (§8) is therefore recall parity with *current main* at strictly
fewer visits — any margin, consistently, across the sweep — not a headline percentage.

---

## 5. Design — engine layer (out of Lucene scope, same floor object)

The distributed deployment reuses `GlobalKnnFloor.advertise()` as its entire integration:

- **Coordinator:** owns the query; merges hit streams; maintains its own size-k heap of
  received hits; broadcasts `floor = heap.peek()` (once full) to shards over the existing
  gRPC `coordinate()` stream. Star topology, exactly today's `KnnResource.floorHeap` — no
  redesign needed, it was correct.
- **Duplicate-document precondition:** the floor-validity lemma (§10.3) assumes hits are
  distinct documents. Disjoint (round-robin) shards satisfy this for free. If shards can
  hold the same document (replicas, global-ID joined indexes), the coordinator must dedup
  by global doc id *before* heap insertion — duplicated hits would inflate the merged
  k-th-best above the true cutoff and break the lower-bound invariant.
- **Shard:** runs `SharedFloorKnnCollectorManager` with a `GlobalKnnFloor` whose `advertise()`
  is fed by coordinator messages. Local leaves and remote floor tighten the same object.
- **Per-shard k:** the coordinator applies `perLeafTopKCalculation(k, shardProportion)` to
  set each shard's request k — the 16-shard/k=10,000 case sends ~1,000 per shard, and the
  re-entry analogue is a second, targeted round-trip to shards whose worst hit beats the
  merged cutoff. This alone removes most of the `k × shards` rerank waste independent of
  floor-based early termination.
- **Over-fetch is retired as a correctness crutch** (it existed to mask the ascent bug) but
  a small factor may return as a tuning knob if the small-index sweep shows the last recall
  point needs it.

### 5.1 Scout shard (floor seeding)

A quantized (int8/binary) in-memory replica of a corpus sample, searched first (or
concurrently — it finishes in microseconds):

- Scout's k-th-best *calibrated* score seeds the floor via `advertise()` before any real
  shard fills its heap — closing the one gap pure floor-sharing has (the floor is `-inf`
  until somebody converges).
- **Calibration is mandatory:** quantized similarity is a biased estimate. The seed must be
  a lower bound with margin, e.g. `advertise(scoutKth - δ)` with δ from a held-out
  calibration set, or rescore the scout's top-k in full precision (k full-precision scores
  is trivially cheap) and advertise the exact value. An uncalibrated scout floor
  reintroduces the guillotine.
- Because the floor only ever tightens, a conservative seed is always safe; it just prunes
  less until real hits arrive.
- Scout stays entirely engine-side. In core Lucene terms it is just another `advertise()`
  caller — this is the argument for keeping that one method in the core API.

---

## 6. Recall-safety analysis (honest version)

Invariant: `floor(t)` = k-th best of a *subset* of all hits that will ever exist ≤ final
merged k-th-best. Therefore any node truly unable to beat `floor(t)` can never enter the
final top-k, and pruning on it is loss-free.

The gap between theory and implementation is the phrase "truly unable": HNSW estimates
reachability by its frontier top (`candidates.topScore()`), which is not an upper bound on
what deeper exploration could reach through low-scoring bridge nodes. Stock HNSW makes
exactly this approximation against its own local bar; we raise that bar toward the global
cutoff, which prunes exploration in the band `[local k-th, global k-th)` — where bridges
live. So the design is **not** "recall-safe by construction"; it is "the same greedy
approximation stock makes, at a tighter bar, with two explicit compensators":

1. the ascent gate guarantees the bar never applies before the leaf reaches its
   neighborhood (eliminating the entry-point guillotine, the dominant failure of #15676);
2. the greediness clamp bounds how far above the local bar the effective bar can rise,
   preserving a `(1-g)·k`-deep exploration frontier for bridges.

Consequently recall parity is an **empirical acceptance criterion** (§8), not a theorem.
`g` is the recall/visits dial: `g=0` is stock, `g=1` is hard-stop. 0.9 shipped in Lucene 9.x as
the `MultiLeafKnnCollector` default, but randomized small-k unit testing showed it costing
double-digit recall (worst observed: 0.96 → 0.71 at k=10 under the tightest valid bound). The
diagnosis: the clamp was *fractional* (`(1-g)·k` slots) while the protection HNSW needs is
*absolute* — a graph search needs some minimum number of below-bar candidates alive to route
through, regardless of k. Two mechanisms fix this, both implemented:

1. **Absolute slot minimum.** Clamp size = `max(16, (1-g)·k)` (`MIN_EXPLORATION_SLOTS`).
   Emergent property: at `k ≤ 16` the clamp is at least as wide as the local queue and the
   floor is *organically neutralized* — small-k search degrades to exactly stock behavior even
   under a hostile advertised bound (unit-tested with `advertise(Float.MAX_VALUE)`).
2. **Activation policy (the "when is collaboration acceptable" decision).** The manager engages
   the floor only when the query's k reaches `floorActivationK` (default 100); below it, it
   creates plain `TopKnnCollector`s — bit-identical to stock search, zero overhead, immune even
   to invalid floors (unit-tested). The threshold is justified by the §10.1 math from both
   directions: savings grow with k (`R = s/(1+λ√((s−1)/k))`, and with λ=16 the pro-rata padding
   swamps the share at small k, leaving nothing for a floor to cut), while bridge-loss risk
   concentrates at small k. Richer policies (segment count, corpus size, scout availability)
   live in the engine layer by deciding which manager to construct per query — the manager *is*
   the strategy seam; no new core interface is needed.

The default remains a conservative **g=0.5**; raising it is a measured, per-dataset trade (§11
sweeps). Unit tests pin the safe endpoint: at `g=0`, even the tightest valid advertised bound
(the exact final k-th best) must not cost recall versus stock.

## 7. Determinism

#15686's removal note cites "non-deterministic results due to race conditions." Position:

- The floor is monotonic and all updates commute (max/heap-insert), so the *floor sequence*
  is race-free; what varies with thread interleaving is *when* each leaf observes which
  floor value → visit counts always vary, result sets can vary where the greedy
  approximation binds.
- Sequential execution (no executor) is fully deterministic — assert it in tests.
- Parallel: measure result-set stability across repeated runs in the step-1 test (assert
  recall within ε and report set-difference rates). Document collaborative mode as opt-in
  and approximate, same class of trade as `HnswQueueSaturationCollector` patience mode.
- If reviewers require hard determinism, the fallback is floor updates only at leaf
  completion boundaries with a fixed reduction order — worse pruning, kept in reserve, not
  in PR 1.

## 8. Testing plan

(Quantitative rationale in §10; concrete harness wiring and execution matrices in §11.)

Acceptance bar per configuration: **recall ≥ current-main recall − 0.005 AND visits <
current-main visits**, where current-main = optimistic path as of `8fe701c0873` (includes
#16324). A config that cannot hold recall is rejected regardless of visit savings.

Arms, every run: (1) stock/optimistic (baseline); (2) resurrected `MultiLeafKnnCollector`
(`git revert 75d47f09c45`, the incumbent-that-was); (3) `FloorAware` composed with
optimistic (the candidate); (4) `HnswQueueSaturationCollector` patience (the in-tree
alternative visit-reducer); (5) old `CollaborativeKnnCollector` (the before picture, bench
branch only).

Ladder, cheap → expensive:

1. **Unit tests (in Lucene, every build).** Deterministic multi-segment index;
   1/2/4/8/16 segments × sequential/parallel; assert recall parity, sequential determinism,
   parallel stability (§7); ascent-gate regression (a converged segment must not prune a
   fresh one before it fills); floor-validity property test (floor(t) ≤ final k-th, always).
2. **Small-index sweep (73k local shards / small Cohere).** Tune `g`, sync interval,
   heap size (k vs k+fanout), optional over-fetch. All parameter search happens here.
   Include a **skewed-segment** config (tiered sizes) — 16 equal segments is the
   friendliest case and not representative.
3. **16-shard distributed rig.** Same arms over gRPC; adds per-shard-k reduction and scout
   seeding (with and without calibration, to demonstrate the guillotine and its fix).
   Primary demo: `k=10,000 × 16 shards` fan-in reduction.
4. **247M cold run (the deliverable).** Only after 1–3 pass. Fanout sweep {50,100,200,400,800},
   cold page cache, sequential and parallel-16. Output: visited/recall Pareto curves per arm
   — the artifact benwtrent asked for.

Standard-dataset leg: at least one luceneutil config on a public dataset (Cohere wiki 768)
so results are reproducible by reviewers without our shards.

Housekeeping before clean runs: remove `DEBUG q0` prints in `KnnGraphTester.java` (~1394).

## 9. PR strategy

1. Build and validate on `feature/cross-shard-optimistic-knn` (this branch, fork-only).
2. Open a fresh PR: title ~"Extend optimistic kNN collection with a shared global
   score floor across leaves"; body = §1 motivation, §3 prior-art table (explicitly:
   this resurrects the `MultiLeafKnnCollector` idea post-#15686, composed with optimistic
   collection, with the gate/greediness/batching rationale), Pareto curves from step 2/4.
3. Close #15676 with a comment linking forward: root-cause post-mortem (§2) + new PR.
4. Keep `bench/collab-plus-distinctdocs` for A/B history; never merge it anywhere.

Diff budget for PR 1: `GlobalKnnFloor`, `FloorAwareKnnCollector`,
`SharedFloorKnnCollectorManager`, resurrected `BlockingFloatHeap` (+ its old tests), new unit
tests, CHANGES entry. No `HnswGraphSearcher` changes, no `AbstractKnnVectorQuery` changes,
no new top-level strategy interface.

## 10. Quantitative model — why this wins

Notation: `N` corpus size, `s` shards/leaves (equal-sized unless noted), `k` requested
neighbors, `p_i = n_i/N` shard i's share, `s*` the final merged k-th-best score,
`floor(t)` the shared floor at time `t`, `g` greediness, `λ` Lucene's exploration constant.

### 10.1 Per-shard k under the random-sample assumption

If shards are random samples, the number of global top-k hits residing in shard i is
`X_i ~ Binomial(k, p_i)`, so `E[X_i] = k·p_i`, `σ_i = sqrt(k·p_i(1-p_i))`. Lucene's
optimistic path already sizes each leaf's quota as

```
perShardK(k, p) = k·p + λ·sqrt(k·p·(1−p)),    λ = 16   (LAMBDA in AbstractKnnVectorQuery)
```

For equal shards (`p = 1/s`) the total collected across shards telescopes to a closed form:

```
Σ perShardK = k + λ·sqrt(k·(s−1))
```

so the fan-in / rerank reduction versus today's "every shard returns k" contract is

```
R(k, s) = k·s / (k + λ·sqrt(k·(s−1))) = s / (1 + λ·sqrt((s−1)/k))
```

`R → s` as `k → ∞`: **the bigger k is, the closer we get to the full s-fold reduction —
the k=10,000 pain point is exactly where the math is most favorable.** Concretely at s=16:

| k | perShardK | Σ collected (vs k·s today) | fan-in reduction R |
|---|---|---|---|
| 100 | 45 | 720 vs 1,600 | 2.2× |
| 1,000 | 185 | 2,960 vs 16,000 | 5.4× |
| 10,000 | 1,012 | 16,197 vs 160,000 | **9.9×** |

Coverage: `λ=16` puts the per-shard quota 16 standard deviations above its mean — under the
sampling model the probability a shard truly owes more than its quota is negligible
(even λ=3 gives 0.13%/shard). Re-entry exists to repair *distribution skew* (non-random
placement), not sampling noise; its trigger rate is one of the things the harness measures.

### 10.2 Visit-cost model and where savings come from

Per-shard HNSW visits decompose as

```
V(ef, N) ≈ c₁·ln N            (layered greedy descent — ascent phase, cannot be pruned safely)
         + d_eff·ef           (layer-0 fill + refinement; d_eff = avg neighbors scored per accepted candidate)
```

Calibrating on the 247M measurement (stock: 41,749 visits / 16 segments ≈ 2,609 per segment
at `ef ≈ k+fanout = 100`, segments of ~15.4M vectors): the `ln N` term is tens of visits, so
`d_eff ≈ 26` and **the ef-proportional term dominates ≈ 99% of visits.** Two consequences:

1. Per-shard quota reduction (10.1) transfers almost linearly into visit reduction:
   `V_total ≈ s·c₁·ln N + d_eff·(k + λ√(k(s−1)))` versus stock `s·c₁·ln N + d_eff·s·k` —
   the same `R(k,s)` ratio applies to the dominant term.
2. The ascent gate costs us only the `c₁·ln N` term plus the local fill — which is work no
   correct scheme can skip anyway (a leaf cannot know it is non-competitive before reaching
   its neighborhood).

### 10.3 Floor safety lemma and the pruning band

**Lemma (floor validity).** `floor(t)` is the k-th best of the hits observed so far,
`H(t) ⊆ H(final)`; a k-th best over a subset can only be ≤ the k-th best over the superset,
so `floor(t) ≤ s*` for all `t`, and `floor` is monotone non-decreasing since `H(t)` only
grows. The same subset argument bounds every component of the effective bar:

```
bar_i(t) = max( localKth_i(t),  min( nonCompetitive_i(t), floor(t) ) )  ≤  s*
```

so a node whose reachable score is below `bar_i(t)` can never enter the final top-k —
pruning on it is loss-free, *modulo* the one approximation stock HNSW already makes:
`candidates.topScore()` (the frontier top) is treated as the reachable bound, though paths
through low-scoring bridge nodes can reach better clusters. Raising the bar from
`localKth_i` toward `s*` prunes exploration precisely in the band `[localKth_i, bar_i]`,
which is where bridges can be lost. The greediness clamp bounds the exposure:
`bar_i ≤ nonCompetitive_i.peek()` = the `((1−g)·k)`-th best similarity leaf i has seen, so a
`(1−g)·k`-deep exploration frontier always survives. `g` is therefore the recall↔visits
dial with hard endpoints: `g=0` ≡ stock (no global effect), `g=1` ≡ hard-stop (the #15676
failure mode).

### 10.4 Expected refinement savings from the floor

Stock leaf i refines until its frontier drops below its own `localKth_i` — a bar near the
global rank-`s·k` score (leaf i's k-th best ≈ the corpus (k·s)-th best under sampling). But
only ~`k/s` of its results matter. The floor lifts the stopping bar from rank-`sk` territory
toward rank-`k` territory (`s*`), so the ideal (fully-converged-floor) saving is the
exploration spent certifying the `k − k/s = k(s−1)/s` per-leaf results that can never
survive the merge — i.e. **up to `(s−1)/s` (94% at s=16) of refinement work**, realized only
to the extent the floor has risen by the time the leaf refines. Floor freshness is
timing-dependent (parallel interleaving, batching interval), which is why the harness logs
the floor trajectory (§11) instead of assuming the ideal. Note 10.1 and 10.4 overlap: quota
reduction already shrinks each leaf's certification duty; the floor mops up the remainder
(late phase-1 work and the re-entry phase).

### 10.5 Scout seeding

Let the scout hold a uniform ρ-sample of the corpus, quantized for speed, with its top-k
**rescored in full precision** before advertising. Rescored scout hits are true corpus hits,
so the lemma in 10.3 applies verbatim: `floor_scout` = k-th best of a hit subset ≤ `s*` —
**provably valid with zero calibration tuning** (this is why we prefer rescoring over a
margin δ). Tightness: the expected global rank of the scout's j-th best is `j/ρ`, so

```
floor_scout ≈ score at global rank k/ρ ;   ρ = 1/s  ⇒  floor_scout ≈ score@rank k·s
```

i.e. a scout holding one shard's worth of vectors seeds, at `t ≈ 0`, a floor as tight as
what a real shard advertises only *after full convergence*. The scout effectively teleports
every shard past the "wait for the first converged peer" phase — the exact gap (floor =
−∞ until somebody converges) that pure floor-sharing cannot close. Quantization error only
perturbs *which* candidates the scout surfaces (mild tightness loss), never validity.

### 10.6 Coordination cost

The floor rises at most k times; with monotone dedup and interval batching (τ=256 visits)
each shard performs ≤ `V/τ` floor syncs (~10 per query at V≈2,600) and the coordinator
broadcasts ≤ `min(k, ΣV/τ)` updates of one scalar each. Fan-in is bounded by Σ perShardK
(10.1). Coordination traffic is second-order relative to the hit streams it replaces; this
is also why the star topology needs no tree/striping escalation at realistic scales.

## 11. Harness strategy

### 11.1 Single-JVM: luceneutil `KnnGraphTester`

Existing machinery we reuse as-is: `ProfiledKnnFloatVectorQuery` overrides
`getKnnCollectorManager()` per arm and accumulates visits by summing `td.totalHits.value()`
across `mergeLeafResults` calls — under the optimistic path that method runs for both phases,
so **visits correctly include re-entry work**; recall comes from the existing exact-NN ground
truth and `checkResults`. Changes needed:

- Add a `maybeFloorSharedManager(k, GlobalKnnFloor)` reflection hook mirroring
  `maybeCollaborativeManager` (keeps the harness compiling against jars that lack the new
  classes; the old hook keeps working against the bench jar for the "before" arm).
- New flag `-floorShared`; keep `-collaborative` for the legacy arm.
- Instrumentation counters (static `LongAdder`s behind a `-stats` flag): floor syncs/query,
  floor value at each leaf's termination, re-entry count, and a sampled floor-vs-visits
  trajectory — this is the observable for the "realized fraction" in §10.4.
- Remove the leftover `DEBUG q0` prints (~line 1394) before any clean run.

Arms (same index, same seed, every run): **(1)** stock = current-main optimistic;
**(2)** resurrected `MultiLeafKnnCollector` (revert of #15686 — the incumbent-that-was);
**(3)** `FloorAware` composed with optimistic (the candidate); **(4)**
`HnswQueueSaturationCollector` patience (in-tree alternative); **(5)** old
`CollaborativeKnnCollector` via the bench jar (the "before" picture).

Execution matrix per arm: {sequential, parallel-16} × fanout {0, 50, 100, 200, 400, 800},
emitting one CSV row per point → Pareto curves of visits vs recall. Parameter sweeps
(`g ∈ {0.3, 0.5, 0.7, 0.9, 1.0}`, sync interval ∈ {63, 255, 1023}, gate-fill ∈
{perLeafTopK, k}, floor heap ∈ {k, k+fanout}) run **only** on the small index.

Decision gates before scaling up:
- Arm 3 recall ≥ arm 1 recall − 0.005 at every sweep point for some `g`; otherwise add the
  over-fetch knob (search `m·perLeafTopK`, return `perLeafTopK`) and re-sweep.
- Arm 3 visits < arm 1 visits by ≥5% at matched recall; if the margin is thinner, verify on
  a larger index before concluding (the refinement tail grows with ln N and ef) — do not
  burn the 247M run on a configuration that fails the small-index gate.
- Sequential determinism holds exactly; parallel result-set instability stays within the ε
  band across 10 repeats (§7 evidence for the PR).

### 11.2 16-shard distributed rig

Factorized arms so each mechanism's contribution is isolated — this is the experiment that
produces the §1 headline table:

| arm | per-shard k | floor | scout | isolates |
|---|---|---|---|---|
| D0 (today) | k | — | — | baseline |
| D1 | perShardK(k, 1/s) | — | — | optimistic contract alone (§10.1) |
| D2 | perShardK | gRPC-fed | — | floor contribution (§10.4) |
| D3 | perShardK | gRPC-fed | seeded | scout contribution (§10.5) |

Wiring: swap `CollaborativeKnnCollectorManager` → `SharedFloorKnnCollectorManager` in
`KnnNodeService` (the coordinator's `floorHeap`/broadcast loop in `KnnResource` is already
correct and stays); apply `perShardK` in the request builder; scout = one extra node holding
a rescoring int8 replica of a 1/16 sample, advertising once per query. Retire the 5×
over-fetch default (it was masking the ascent bug) — reintroduce only if 11.1 gates said so.

Metrics per (arm, k ∈ {100, 1000, 10000}): per-shard `nodesVisited` (already in
`SearchDebug`), hits emitted per shard (fan-in), floor broadcasts, wall-clock p50/p99,
recall vs merged brute-force truth, re-query (re-entry analogue) rate. The k=10,000 row of
D0 vs D3 — fan-in, visits, latency, recall — is the headline.

### 11.3 247M cold run (the deliverable)

Only after 11.1 gates pass and 11.2 confirms the distributed story. Arms 1/2/3 from 11.1,
fanout sweep, sequential and parallel-16, cold page cache (`drop_caches` between arms,
one arm per cache cycle), fixed query set (nq=100, matched top-100 truth). Output: the
visits-vs-recall Pareto curves for the PR, plus the floor-trajectory plot demonstrating
*why* the visits fell. Acceptance: recall ≥ stock − 0.005 with strictly fewer visits at
every sweep point; anything less and the PR does not go out.

## 12. Open questions

1. `advertise()` in core vs keeping `GlobalKnnFloor` sealed and engine subclassing — core
   method preferred (one line, makes the class useful beyond a single JVM without carrying
   any transport).
2. Should `newOptimisticCollector`'s gate fill at `perLeafTopK` (current choice — earlier
   global influence, matches what the leaf "owes") or at full `k` (later, safer)? Step-2
   sweep decides.
3. Greediness is exposed as a constructor knob (default 0.5); should the default instead scale
   with k so large-k searches keep a bounded absolute number of non-competitive slots? Decide
   from the §11 sweeps.
4. Scout calibration: fixed margin δ vs full-precision rescore of scout top-k. Lean rescore
   (exact, trivially cheap, no tuning).

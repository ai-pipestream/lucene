/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.search.join;

import org.apache.lucene.internal.hppc.IntFloatHashMap;
import org.apache.lucene.internal.hppc.IntFloatHashMap.IntFloatCursor;
import org.apache.lucene.internal.hppc.IntHashSet;
import org.apache.lucene.sandbox.search.knn.GlobalKnnFloor;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.hnsw.FloatHeap;

/**
 * A {@link KnnCollector.Decorator} that lets a {@link
 * DiversifyingNearestChildrenKnnCollector} prune against a {@link GlobalKnnFloor} shared with the
 * other searchers of the same query, the parent-scoring analog of {@link
 * org.apache.lucene.sandbox.search.knn.FloorAwareKnnCollector}. The ascent gate, greediness clamp
 * and batched synchronization of that class apply unchanged; what differs is the unit the floor
 * tracks, and that difference is dictated by {@link GlobalKnnFloor}'s distinct-document contract.
 *
 * <p>With max-style parent scoring, a parent's score is the best score among its children, and the
 * query collects the top {@code k} parents. The floor must therefore be a lower bound of the final
 * k-th best <em>parent</em> score, and the k-th best of any set of raw child scores is not one: k
 * high-scoring children can all belong to a single parent, so a floor over child scores can exceed
 * the true parent cutoff and over-prune. This collector instead tracks the best score observed so
 * far per parent and publishes each parent into the floor at most once, at the first
 * synchronization after the parent's first observation. The published value never exceeds the
 * parent's final score (a best-so-far can only rise), and each parent contributes at most one
 * entry, so the floor's heap is a pointwise lower bound of the final parent scores and its k-th
 * best is a valid bound of the merged cutoff. Two consequences of the insert-only heap:
 *
 * <ul>
 *   <li>A parent's later improvements are not republished: a second entry for the same parent
 *       would make the heap a multiset over one parent and could lift the floor above the true
 *       cutoff. The floor thus converges to the cutoff from below, slightly conservatively for
 *       parents published early.
 *   <li>The block-join layout guarantee, a parent and its children in one segment, is what makes
 *       per-leaf publication globally distinct: a parent is scored by exactly one leaf's collector
 *       per pass, and {@link SharedFloorDiversifyingKnnCollectorManager} suppresses publication on
 *       the optimistic strategy's second pass of the same leaf.
 * </ul>
 *
 * <p>The greediness clamp is fed with parent-best improvements rather than raw child scores, so
 * the clamp and the floor stay in the same unit and the clamp's absolute width keeps the meaning
 * documented on {@link org.apache.lucene.sandbox.search.knn.FloorAwareKnnCollector}.
 *
 * <p>Instances are confined to a single thread, like every {@link KnnCollector}; only the shared
 * {@link GlobalKnnFloor} is touched by multiple threads.
 *
 * @lucene.experimental
 */
final class FloorAwareDiversifyingChildrenKnnCollector extends KnnCollector.Decorator {

  private final DiversifyingNearestChildrenKnnCollector subCollector;
  private final GlobalKnnFloor globalFloor;
  private final BitSet parentBitSet;

  /** Number of locally collected parents after which the shared floor engages. */
  private final int gateK;

  /** Number of visited vectors between synchronizations with the shared floor. */
  private final int syncInterval;

  /**
   * Whether this collector publishes its observed parent scores into the shared floor. See {@link
   * SharedFloorDiversifyingKnnCollectorManager} for the publish-once-per-leaf discipline.
   */
  private final boolean publishToFloor;

  /**
   * The best {@code max(minExplorationSlots, (1 - greediness) * gateK)} parent-best similarities
   * seen by this collector; its minimum caps the effective bound, implementing the greediness
   * clamp.
   */
  private final FloatHeap nonCompetitiveQueue;

  /**
   * Best similarity observed so far per parent, for parents not yet flushed to the floor. Cleared
   * at each flush; a parent that improves after being flushed re-enters the map but is filtered
   * out by {@link #publishedParents}, so it is never offered twice.
   */
  private final IntFloatHashMap parentBest;

  /** Parents whose score has been offered to the floor; never cleared. */
  private final IntHashSet publishedParents;

  /**
   * Scratch buffer assembling the next publication batch; bounded at the floor's k, so an
   * oversized batch keeps only the best parent scores, which is all the floor can use.
   */
  private final FloatHeap updatesQueue;

  /** Scratch used to drain {@link #updatesQueue} in ascending order for batch publication. */
  private final float[] updatesScratch;

  private boolean gateOpened;

  /**
   * The visited count at or beyond which the next synchronization with the shared floor happens;
   * see {@link org.apache.lucene.sandbox.search.knn.FloorAwareKnnCollector} for why a threshold
   * is used instead of an exact boundary.
   */
  private long nextSyncAt;

  private float cachedGlobalFloor = Float.NEGATIVE_INFINITY;

  /**
   * Create a fully configured collector.
   *
   * @param subCollector the collector gathering this searcher's local top parents
   * @param globalFloor the floor shared by all searchers of this query, tracking parent scores
   * @param parentBitSet the leaf's parent bitset, the same instance the delegate joins with
   * @param greediness fraction of the search effort that follows the shared floor, in {@code [0,
   *     1]}; see {@link org.apache.lucene.sandbox.search.knn.FloorAwareKnnCollector}
   * @param minExplorationSlots smallest permitted size of the greediness clamp's queue; must be at
   *     least 1
   * @param syncInterval number of visited vectors between synchronizations with the shared floor;
   *     must be positive
   * @param gateK the number of locally collected parents after which the shared floor engages, in
   *     {@code [1, subCollector.k()]}
   * @param publishToFloor whether this collector feeds the floor or only reads it
   */
  FloorAwareDiversifyingChildrenKnnCollector(
      DiversifyingNearestChildrenKnnCollector subCollector,
      GlobalKnnFloor globalFloor,
      BitSet parentBitSet,
      float greediness,
      int minExplorationSlots,
      int syncInterval,
      int gateK,
      boolean publishToFloor) {
    super(subCollector);
    if (greediness < 0 || greediness > 1 || Float.isNaN(greediness)) {
      throw new IllegalArgumentException("greediness must be in [0,1], got: " + greediness);
    }
    if (minExplorationSlots < 1) {
      throw new IllegalArgumentException(
          "minExplorationSlots must be at least 1, got: " + minExplorationSlots);
    }
    if (syncInterval < 1) {
      throw new IllegalArgumentException("syncInterval must be positive, got: " + syncInterval);
    }
    if (gateK < 1 || gateK > subCollector.k()) {
      throw new IllegalArgumentException(
          "gateK must be in [1, subCollector.k()=" + subCollector.k() + "], got: " + gateK);
    }
    this.subCollector = subCollector;
    this.globalFloor = globalFloor;
    this.parentBitSet = parentBitSet;
    this.gateK = gateK;
    this.syncInterval = syncInterval;
    this.publishToFloor = publishToFloor;
    this.nonCompetitiveQueue =
        new FloatHeap(Math.max(minExplorationSlots, Math.round((1 - greediness) * gateK)));
    this.parentBest = new IntFloatHashMap();
    this.publishedParents = new IntHashSet();
    this.updatesQueue = publishToFloor ? new FloatHeap(globalFloor.k()) : null;
    this.updatesScratch = publishToFloor ? new float[globalFloor.k()] : null;
  }

  /** The ascent gate's threshold; package-private, for tests and the manager. */
  int gateK() {
    return gateK;
  }

  /** Whether this collector feeds the shared floor; package-private, for tests. */
  boolean publishesToFloor() {
    return publishToFloor;
  }

  @Override
  public boolean collect(int docId, float similarity) {
    boolean localSimUpdated = subCollector.collect(docId, similarity);
    boolean gateJustOpened = gateOpened == false && subCollector.numCollected() >= gateK;
    if (gateJustOpened) {
      gateOpened = true;
    }
    // Track the parent-unit best so the clamp and any later publication stay in parent scores.
    int parent = parentBitSet.nextSetBit(docId);
    boolean globalSimUpdated = false;
    float previousBest = parentBest.getOrDefault(parent, Float.NEGATIVE_INFINITY);
    if (similarity > previousBest) {
      parentBest.put(parent, similarity);
      globalSimUpdated = nonCompetitiveQueue.offer(similarity);
    }

    if (gateOpened && (gateJustOpened || visitedCount() >= nextSyncAt)) {
      nextSyncAt = visitedCount() + syncInterval;
      boolean published = false;
      if (publishToFloor) {
        int len = drainUnpublishedParents();
        if (len > 0) {
          // offer() returns the floor after the batch, siblings' contributions included.
          cachedGlobalFloor = globalFloor.offer(updatesScratch, len);
          globalSimUpdated = true;
          published = true;
        }
      }
      if (published == false) {
        // A non-publishing collector only reads; a publishing one whose batch was empty may also
        // have fallen behind its siblings. A stale floor can only delay termination, never cause
        // a wrong result.
        float refreshed = globalFloor.floor();
        if (refreshed > cachedGlobalFloor) {
          cachedGlobalFloor = refreshed;
          globalSimUpdated = true;
        }
      }
    }
    return localSimUpdated || globalSimUpdated;
  }

  /**
   * Offer the current best of every parent observed since the last flush that has never been
   * published, and return the batch length. Each parent enters the floor exactly once, with a
   * value that cannot exceed its final score; later improvements stay local (see the class
   * comment).
   */
  private int drainUnpublishedParents() {
    if (parentBest.isEmpty()) {
      return 0;
    }
    for (IntFloatCursor entry : parentBest) {
      if (publishedParents.add(entry.key)) {
        updatesQueue.offer(entry.value);
      }
    }
    parentBest.clear();
    int len = updatesQueue.size();
    if (len > 0) {
      for (int i = 0; i < len; i++) {
        updatesScratch[i] = updatesQueue.poll();
      }
      assert updatesQueue.size() == 0;
    }
    return len;
  }

  @Override
  public float minCompetitiveSimilarity() {
    if (gateOpened == false) {
      return subCollector.minCompetitiveSimilarity();
    }
    // nextDown keeps the bound strictly below the floor so exact score ties at the cutoff remain
    // reachable; nextDown of NEGATIVE_INFINITY is NEGATIVE_INFINITY, so an undefined floor is a
    // no-op here.
    return Math.max(
        subCollector.minCompetitiveSimilarity(),
        Math.min(nonCompetitiveQueue.peek(), Math.nextDown(cachedGlobalFloor)));
  }

  @Override
  public String toString() {
    return "FloorAwareDiversifyingChildrenKnnCollector[subCollector=" + subCollector + "]";
  }
}

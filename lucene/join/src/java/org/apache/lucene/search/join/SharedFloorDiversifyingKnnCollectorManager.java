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

import java.io.IOException;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.sandbox.search.knn.FloorAwareKnnCollector;
import org.apache.lucene.sandbox.search.knn.GlobalKnnFloor;
import org.apache.lucene.sandbox.search.knn.SharedFloorKnnCollectorManager;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.util.BitSet;

/**
 * A {@link KnnCollectorManager} that creates {@link DiversifyingNearestChildrenKnnCollector}
 * instances whose segment searches prune against a {@link GlobalKnnFloor} shared by every searcher
 * of the same query, composing block-join parent diversification with the shared-floor mechanism
 * of {@link SharedFloorKnnCollectorManager}. Where that class bounds each segment by the merged
 * top-k document cutoff, this one bounds each segment by the merged top-k <em>parent</em> cutoff:
 * with max-style parent scoring, a parent's score is its best child's score, and the floor tracks
 * the k-th best parent score observed so far across all searchers. {@link
 * FloorAwareDiversifyingChildrenKnnCollector} documents why raw child scores cannot feed that
 * floor and how per-parent publication keeps the bound valid.
 *
 * <p>Everything {@link SharedFloorKnnCollectorManager} documents about its own mechanics applies
 * here unchanged: the manager reports {@link #isOptimistic()} so the kNN machinery runs its
 * pro-rata collection strategy on top; the ascent gate opens at each leaf's statistically expected
 * contribution to the merged top-k, derived from the leaf's share and the index's {@code
 * globalShare} via {@link SharedFloorKnnCollectorManager#perShardGate(int, double)}; each leaf's
 * scores are published at most once per query, so collectors created for a leaf's second pass read
 * the floor without feeding it; and floor sharing engages only when the query's k reaches {@code
 * floorActivationK}, below which this manager creates plain, undecorated collectors and the search
 * is exactly stock block-join search.
 *
 * <p>The floor may also be fed from outside this process through {@link
 * GlobalKnnFloor#advertise(float)}, with the same caller responsibilities {@link
 * SharedFloorKnnCollectorManager} defines, one unit change aside: the advertised bound must be a
 * lower bound of the final k-th best <em>parent</em> score. A remote shard's converged top-parents
 * cutoff qualifies; any k-th best of raw child scores does not.
 *
 * <p>A manager, like the floor it holds, carries the state of a single query execution: create one
 * per query, and never share one across queries. Both collector-creation methods may be called
 * concurrently, as segments are searched in parallel.
 *
 * @lucene.experimental
 */
public final class SharedFloorDiversifyingKnnCollectorManager implements KnnCollectorManager {

  private final int k;
  private final BitSetProducer parentsFilter;
  private final GlobalKnnFloor globalFloor;
  private final float greediness;
  private final int floorActivationK;
  private final int minExplorationSlots;
  private final int syncInterval;
  private final float globalShare;

  /**
   * The leaves whose parent scores have been claimed for publication into the shared floor, so
   * that a second search of the same leaf gets a non-publishing collector; see {@link
   * SharedFloorKnnCollectorManager}.
   */
  private final Set<LeafReaderContext> publishedLeaves = ConcurrentHashMap.newKeySet();

  /**
   * Create a manager with its own floor and default tuning, for queries whose searchers all live
   * in this process.
   *
   * @param k the number of top parents the query collects
   * @param parentsFilter filter identifying the parent documents
   */
  public SharedFloorDiversifyingKnnCollectorManager(int k, BitSetProducer parentsFilter) {
    this(k, parentsFilter, new GlobalKnnFloor(k));
  }

  /**
   * Create a manager around an externally provided floor with default tuning, for queries whose
   * floor is also fed by searchers outside this process.
   *
   * @param k the number of top parents the query collects
   * @param parentsFilter filter identifying the parent documents
   * @param globalFloor the floor shared by all searchers of this query; its {@link
   *     GlobalKnnFloor#k()} must equal {@code k}, since a floor tracking the wrong result-set size
   *     is not a valid bound
   */
  public SharedFloorDiversifyingKnnCollectorManager(
      int k, BitSetProducer parentsFilter, GlobalKnnFloor globalFloor) {
    this(k, parentsFilter, globalFloor, FloorAwareKnnCollector.DEFAULT_GREEDINESS);
  }

  /**
   * Create a manager with an explicit greediness, applying the {@link
   * SharedFloorKnnCollectorManager#DEFAULT_FLOOR_ACTIVATION_K default activation threshold}.
   *
   * @param k the number of top parents the query collects
   * @param parentsFilter filter identifying the parent documents
   * @param globalFloor the floor shared by all searchers of this query; its {@link
   *     GlobalKnnFloor#k()} must equal {@code k}
   * @param greediness fraction of each segment's search effort that follows the shared floor, in
   *     {@code [0, 1]}; see {@link FloorAwareKnnCollector}
   */
  public SharedFloorDiversifyingKnnCollectorManager(
      int k, BitSetProducer parentsFilter, GlobalKnnFloor globalFloor, float greediness) {
    this(
        k,
        parentsFilter,
        globalFloor,
        greediness,
        SharedFloorKnnCollectorManager.DEFAULT_FLOOR_ACTIVATION_K);
  }

  /**
   * Create a manager with an explicit greediness and activation threshold, applying the default
   * slot minimum and sync interval.
   *
   * @param k the number of top parents the query collects
   * @param parentsFilter filter identifying the parent documents
   * @param globalFloor the floor shared by all searchers of this query; its {@link
   *     GlobalKnnFloor#k()} must equal {@code k}
   * @param greediness fraction of each segment's search effort that follows the shared floor, in
   *     {@code [0, 1]}; see {@link FloorAwareKnnCollector}
   * @param floorActivationK the smallest k at which floor sharing engages; for smaller k this
   *     manager creates plain collectors and the search is exactly stock search
   */
  public SharedFloorDiversifyingKnnCollectorManager(
      int k,
      BitSetProducer parentsFilter,
      GlobalKnnFloor globalFloor,
      float greediness,
      int floorActivationK) {
    this(
        k,
        parentsFilter,
        globalFloor,
        greediness,
        floorActivationK,
        FloorAwareKnnCollector.DEFAULT_MIN_EXPLORATION_SLOTS,
        FloorAwareKnnCollector.DEFAULT_SYNC_INTERVAL);
  }

  /**
   * Create a fully configured manager for an index holding the whole corpus. Equivalent to the
   * full constructor with {@code globalShare = 1}.
   *
   * @param k the number of top parents the query collects
   * @param parentsFilter filter identifying the parent documents
   * @param globalFloor the floor shared by all searchers of this query; its {@link
   *     GlobalKnnFloor#k()} must equal {@code k}
   * @param greediness fraction of each segment's search effort that follows the shared floor, in
   *     {@code [0, 1]}
   * @param floorActivationK the smallest k at which floor sharing engages; for smaller k this
   *     manager creates plain collectors and the search is exactly stock search
   * @param minExplorationSlots smallest permitted size of each collector's greediness clamp queue;
   *     must be at least 1
   * @param syncInterval number of visited vectors between each collector's synchronizations with
   *     the shared floor; must be positive
   */
  public SharedFloorDiversifyingKnnCollectorManager(
      int k,
      BitSetProducer parentsFilter,
      GlobalKnnFloor globalFloor,
      float greediness,
      int floorActivationK,
      int minExplorationSlots,
      int syncInterval) {
    this(
        k,
        parentsFilter,
        globalFloor,
        greediness,
        floorActivationK,
        minExplorationSlots,
        syncInterval,
        1f);
  }

  /**
   * Create a fully configured manager. Every tuning value the mechanism has is a parameter here;
   * the shorter constructors exist only to supply defaults.
   *
   * @param k the number of top parents the query collects
   * @param parentsFilter filter identifying the parent documents
   * @param globalFloor the floor shared by all searchers of this query; its {@link
   *     GlobalKnnFloor#k()} must equal {@code k}
   * @param greediness fraction of each segment's search effort that follows the shared floor, in
   *     {@code [0, 1]}. When {@code globalShare} is below 1, derive this value from the
   *     exploration width the graph needs via {@link FloorAwareKnnCollector#greedinessForClamp(int,
   *     int)} (with {@code gateK} from {@link SharedFloorKnnCollectorManager#perShardGate(int,
   *     double)})
   *     rather than choosing a constant; see the greediness clamp caution on that class.
   * @param floorActivationK the smallest k at which floor sharing engages; for smaller k this
   *     manager creates plain collectors and the search is exactly stock search
   * @param minExplorationSlots smallest permitted size of each collector's greediness clamp queue;
   *     must be at least 1
   * @param syncInterval number of visited vectors between each collector's synchronizations with
   *     the shared floor; must be positive
   * @param globalShare the fraction of the whole corpus held by the index this manager searches,
   *     in {@code (0, 1]}. Pass a value below 1 when this index is one shard of a sharded corpus
   *     whose searchers share the floor across processes; each collector's ascent gate then opens
   *     at the shard's expected contribution to the merged top-k instead of at its full local
   *     queue. See {@link SharedFloorKnnCollectorManager}.
   */
  public SharedFloorDiversifyingKnnCollectorManager(
      int k,
      BitSetProducer parentsFilter,
      GlobalKnnFloor globalFloor,
      float greediness,
      int floorActivationK,
      int minExplorationSlots,
      int syncInterval,
      float globalShare) {
    if (k < 1) {
      throw new IllegalArgumentException("k must be at least 1, got: " + k);
    }
    Objects.requireNonNull(parentsFilter, "parentsFilter");
    Objects.requireNonNull(globalFloor, "globalFloor");
    if (globalFloor.k() != k) {
      throw new IllegalArgumentException(
          "the floor must track the same result-set size as the query: floor k="
              + globalFloor.k()
              + ", query k="
              + k);
    }
    if (greediness < 0 || greediness > 1 || Float.isNaN(greediness)) {
      throw new IllegalArgumentException("greediness must be in [0,1], got: " + greediness);
    }
    if (floorActivationK < 1) {
      throw new IllegalArgumentException(
          "floorActivationK must be at least 1, got: " + floorActivationK);
    }
    if (minExplorationSlots < 1) {
      throw new IllegalArgumentException(
          "minExplorationSlots must be at least 1, got: " + minExplorationSlots);
    }
    if (syncInterval < 1) {
      throw new IllegalArgumentException("syncInterval must be positive, got: " + syncInterval);
    }
    if (globalShare <= 0 || globalShare > 1 || Float.isNaN(globalShare)) {
      throw new IllegalArgumentException("globalShare must be in (0,1], got: " + globalShare);
    }
    this.k = k;
    this.parentsFilter = parentsFilter;
    this.globalFloor = globalFloor;
    this.greediness = greediness;
    this.floorActivationK = floorActivationK;
    this.minExplorationSlots = minExplorationSlots;
    this.syncInterval = syncInterval;
    this.globalShare = globalShare;
  }

  /** Return the floor shared by this manager's collectors, so that callers may feed or read it. */
  public GlobalKnnFloor getGlobalFloor() {
    return globalFloor;
  }

  @Override
  public KnnCollector newCollector(
      int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context)
      throws IOException {
    BitSet parentBitSet = parentsFilter.getBitSet(context);
    if (parentBitSet == null) {
      return null;
    }
    DiversifyingNearestChildrenKnnCollector collector =
        new DiversifyingNearestChildrenKnnCollector(k, visitedLimit, searchStrategy, parentBitSet);
    if (k < floorActivationK) {
      return collector;
    }
    return floorAware(collector, context, parentBitSet);
  }

  @Override
  public KnnCollector newOptimisticCollector(
      int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context, int perLeafK)
      throws IOException {
    BitSet parentBitSet = parentsFilter.getBitSet(context);
    if (parentBitSet == null) {
      return null;
    }
    // The local queue is sized to the segment's pro-rata share of the top parents; the floor
    // itself always tracks the full k best parents across the query. Whether floor sharing
    // engages is decided by the query's k against floorActivationK, never by this segment's
    // perLeafK, so that every segment of the same query makes the same engage decision.
    DiversifyingNearestChildrenKnnCollector collector =
        new DiversifyingNearestChildrenKnnCollector(
            perLeafK, visitedLimit, searchStrategy, parentBitSet);
    if (k < floorActivationK) {
      return collector;
    }
    return floorAware(collector, context, parentBitSet);
  }

  private FloorAwareDiversifyingChildrenKnnCollector floorAware(
      DiversifyingNearestChildrenKnnCollector collector,
      LeafReaderContext context,
      BitSet parentBitSet) {
    // Publish each leaf's parent scores at most once per query; a second search of the same leaf
    // re-collects the same parents, and republishing them would violate the floor's
    // distinct-document contract. See the class comment.
    boolean publish = context == null || publishedLeaves.add(context);
    return new FloorAwareDiversifyingChildrenKnnCollector(
        collector,
        globalFloor,
        parentBitSet,
        greediness,
        minExplorationSlots,
        syncInterval,
        gateFor(context, collector.k()),
        publish);
  }

  /**
   * The ascent gate for a collector over {@code context}: the leaf's statistically expected
   * contribution to the merged top-k parents, never more than the local queue it collects into.
   * See {@link SharedFloorKnnCollectorManager} for the derivation.
   */
  private int gateFor(LeafReaderContext context, int queueSize) {
    double leafGlobalShare = globalShare;
    if (context != null && context.parent != null) {
      leafGlobalShare *= context.reader().maxDoc() / (double) context.parent.reader().maxDoc();
    }
    if (leafGlobalShare <= 0 || leafGlobalShare >= 1) {
      return queueSize;
    }
    return Math.min(queueSize, SharedFloorKnnCollectorManager.perShardGate(k, leafGlobalShare));
  }

  @Override
  public boolean isOptimistic() {
    return true;
  }
}

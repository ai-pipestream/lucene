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

package org.apache.lucene.search.knn;

import java.io.IOException;
import java.util.Objects;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopKnnCollector;

/**
 * A {@link KnnCollectorManager} whose collectors share a {@link GlobalKnnFloor}, so that every
 * segment of a query prunes against a lower bound of the query's final merged top-k cutoff rather
 * than only against its own local top-k.
 *
 * <p>This manager reports {@link #isOptimistic()} and provides {@link #newOptimisticCollector},
 * which means the kNN query machinery runs its regular pro-rata collection strategy on top of it:
 * each segment initially collects only its statistically expected share of the global top-k, and
 * segments whose results prove globally competitive are searched again. The shared floor
 * complements that strategy rather than replacing it. During the first pass it lets segments stop
 * refining results that the merged cutoff has already outrun, and because the floor object is
 * shared across both passes, segments searched again start with the bound the first pass
 * established instead of rediscovering it.
 *
 * <p>The floor may also be fed from outside this process through {@link
 * GlobalKnnFloor#advertise(float)}, allowing several searchers of disjoint indexes, each holding a
 * manager wired to the same query, to bound each other's work. Coordinating who advertises what,
 * and deduplicating documents that may appear in more than one index, is the caller's
 * responsibility; see {@link GlobalKnnFloor} for the exact contract.
 *
 * <p>A manager, like the floor it holds, carries the state of a single query execution: create one
 * per query, and never share one across queries. Both collector-creation methods may be called
 * concurrently, as segments are searched in parallel.
 *
 * @lucene.experimental
 */
public final class SharedFloorKnnCollectorManager implements KnnCollectorManager {

  private final int k;
  private final GlobalKnnFloor globalFloor;
  private final float greediness;

  /**
   * Create a manager with its own floor and the {@link FloorAwareKnnCollector#DEFAULT_GREEDINESS
   * default greediness}, for queries whose searchers all live in this process.
   *
   * @param k the number of neighbors the query collects
   */
  public SharedFloorKnnCollectorManager(int k) {
    this(k, new GlobalKnnFloor(k), FloorAwareKnnCollector.DEFAULT_GREEDINESS);
  }

  /**
   * Create a manager around an externally provided floor, for queries whose floor is also fed by
   * searchers outside this process.
   *
   * @param k the number of neighbors the query collects
   * @param globalFloor the floor shared by all searchers of this query; its {@link
   *     GlobalKnnFloor#k()} must equal {@code k}, since a floor tracking the wrong result-set size
   *     is not a valid bound
   */
  public SharedFloorKnnCollectorManager(int k, GlobalKnnFloor globalFloor) {
    this(k, globalFloor, FloorAwareKnnCollector.DEFAULT_GREEDINESS);
  }

  /**
   * Create a manager around an externally provided floor with an explicit greediness.
   *
   * @param k the number of neighbors the query collects
   * @param globalFloor the floor shared by all searchers of this query; its {@link
   *     GlobalKnnFloor#k()} must equal {@code k}
   * @param greediness fraction of each segment's search effort that follows the shared floor, in
   *     {@code [0, 1]}; see {@link FloorAwareKnnCollector}
   */
  public SharedFloorKnnCollectorManager(int k, GlobalKnnFloor globalFloor, float greediness) {
    if (k < 1) {
      throw new IllegalArgumentException("k must be at least 1, got: " + k);
    }
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
    this.k = k;
    this.globalFloor = globalFloor;
    this.greediness = greediness;
  }

  /** Return the floor shared by this manager's collectors, so that callers may feed or read it. */
  public GlobalKnnFloor getGlobalFloor() {
    return globalFloor;
  }

  @Override
  public KnnCollector newCollector(
      int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context)
      throws IOException {
    return new FloorAwareKnnCollector(
        new TopKnnCollector(k, visitedLimit, searchStrategy), globalFloor, greediness);
  }

  @Override
  public KnnCollector newOptimisticCollector(
      int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context, int perLeafK)
      throws IOException {
    // The local queue, and with it the ascent gate, is sized to the segment's pro-rata share; the
    // floor itself always tracks the full k best across the query.
    return new FloorAwareKnnCollector(
        new TopKnnCollector(perLeafK, visitedLimit, searchStrategy), globalFloor, greediness);
  }

  @Override
  public boolean isOptimistic() {
    return true;
  }
}

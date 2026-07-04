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

import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestFloorAwareKnnCollector extends LuceneTestCase {

  public void testInvalidGreediness() {
    GlobalKnnFloor floor = new GlobalKnnFloor(4);
    TopKnnCollector delegate = new TopKnnCollector(4, Integer.MAX_VALUE);
    expectThrows(
        IllegalArgumentException.class, () -> new FloorAwareKnnCollector(delegate, floor, -0.1f));
    expectThrows(
        IllegalArgumentException.class, () -> new FloorAwareKnnCollector(delegate, floor, 1.1f));
    expectThrows(
        IllegalArgumentException.class,
        () -> new FloorAwareKnnCollector(delegate, floor, Float.NaN));
  }

  public void testAscentGateIgnoresFloorUntilLocalQueueFills() {
    int k = 5;
    GlobalKnnFloor floor = new GlobalKnnFloor(k);
    // A sibling searcher has already converged and established a high floor. A correct collector
    // must not expose it before this searcher has escaped its own ascent, otherwise the graph
    // search would be terminated at its entry point before finding anything.
    floor.advertise(10f);
    FloorAwareKnnCollector collector =
        new FloorAwareKnnCollector(new TopKnnCollector(k, Integer.MAX_VALUE), floor);

    for (int doc = 0; doc < k - 1; doc++) {
      collector.incVisitedCount(1);
      collector.collect(doc, 0.1f * (doc + 1));
      assertEquals(
          "the shared floor must be invisible while the local queue is filling",
          Float.NEGATIVE_INFINITY,
          collector.minCompetitiveSimilarity(),
          0.0f);
    }

    collector.incVisitedCount(1);
    collector.collect(k - 1, 0.1f * k);
    assertTrue(
        "once the local queue is full, the shared floor must start binding",
        collector.minCompetitiveSimilarity() > Float.NEGATIVE_INFINITY);
  }

  public void testGreedinessClampCapsTheSharedFloor() {
    int k = 4;
    // greediness 0.5 keeps a non-competitive queue of (1 - 0.5) * 4 = 2 entries, so the effective
    // bound may never exceed the second-best similarity this collector has seen.
    float greediness = 0.5f;
    GlobalKnnFloor floor = new GlobalKnnFloor(k);
    floor.advertise(100f);
    FloorAwareKnnCollector collector =
        new FloorAwareKnnCollector(new TopKnnCollector(k, Integer.MAX_VALUE), floor, greediness);

    collector.incVisitedCount(1);
    collector.collect(0, 1f);
    collector.incVisitedCount(1);
    collector.collect(1, 2f);
    collector.incVisitedCount(1);
    collector.collect(2, 3f);
    collector.incVisitedCount(1);
    collector.collect(3, 4f);

    // Local k-th best is 1, the second-best seen is 3, and the floor is 100. The clamp must win.
    assertEquals(
        "the bound must be capped by the (1-greediness)*k-th best local similarity, not jump to "
            + "the shared floor",
        3f,
        collector.minCompetitiveSimilarity(),
        0.0f);
  }

  public void testSharedFloorBindsOneUlpBelowItsValue() {
    int localK = 4;
    // Size the floor for a larger result set so the four local scores cannot define it: the only
    // floor source in this test is the advertised bound.
    GlobalKnnFloor floor = new GlobalKnnFloor(100);
    floor.advertise(2.5f);
    // greediness 1 collapses the clamp to the single best local similarity, letting the floor
    // term be observed directly.
    FloorAwareKnnCollector collector =
        new FloorAwareKnnCollector(new TopKnnCollector(localK, Integer.MAX_VALUE), floor, 1f);

    collector.incVisitedCount(1);
    collector.collect(0, 1f);
    collector.incVisitedCount(1);
    collector.collect(1, 1.2f);
    collector.incVisitedCount(1);
    collector.collect(2, 1.4f);
    collector.incVisitedCount(1);
    collector.collect(3, 4f);

    // Local k-th best is 1 and the best-seen is 4, so min(bestSeen, nextDown(floor)) selects the
    // floor term. The bound must sit strictly below the floor: a hit scoring exactly at the floor
    // may still win the merged tie-break and must remain findable.
    assertEquals(Math.nextDown(2.5f), collector.minCompetitiveSimilarity(), 0.0f);
    assertTrue(collector.minCompetitiveSimilarity() < 2.5f);
  }

  public void testCollectReportsSharedFloorUpdates() {
    int k = 2;
    GlobalKnnFloor floor = new GlobalKnnFloor(k);
    // greediness 0 keeps a non-competitive queue of k entries; it mirrors the local queue, so a
    // score rejected by both cannot report an update through either local structure.
    FloorAwareKnnCollector collector =
        new FloorAwareKnnCollector(new TopKnnCollector(k, Integer.MAX_VALUE), floor, 0f);

    collector.incVisitedCount(1);
    assertTrue("a locally accepted hit must report an update", collector.collect(0, 5f));
    collector.incVisitedCount(1);
    assertTrue(collector.collect(1, 6f));

    // Both queues hold {5, 6}. A worse score away from a synchronization boundary changes
    // nothing and must say so, otherwise the searcher would re-derive its bound for no reason.
    collector.incVisitedCount(1);
    assertFalse(
        "a rejected hit between synchronizations must not report an update",
        collector.collect(2, 1f));

    // Advance to the next synchronization boundary: even a rejected hit must report an update
    // there, because the re-read of the shared floor may have moved the effective bound.
    collector.incVisitedCount(253);
    assertEquals(0, collector.visitedCount() & 0xff);
    assertTrue(
        "a hit on a synchronization boundary must report an update after the floor re-read",
        collector.collect(3, 1f));
  }

  public void testDelegationOfCollectorPlumbing() {
    GlobalKnnFloor floor = new GlobalKnnFloor(3);
    TopKnnCollector delegate = new TopKnnCollector(3, 17);
    FloorAwareKnnCollector collector = new FloorAwareKnnCollector(delegate, floor);
    assertEquals(3, collector.k());
    assertEquals(17, collector.visitLimit());
    collector.incVisitedCount(5);
    assertEquals(5, collector.visitedCount());
    assertEquals(5, delegate.visitedCount());
    assertFalse(collector.earlyTerminated());
    collector.incVisitedCount(12);
    assertTrue(
        "the visit limit must keep terminating through the decorator", collector.earlyTerminated());
  }
}

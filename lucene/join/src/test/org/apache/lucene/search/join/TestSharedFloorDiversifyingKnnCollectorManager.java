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
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.sandbox.search.knn.GlobalKnnFloor;
import org.apache.lucene.sandbox.search.knn.SharedFloorKnnCollectorManager;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.FixedBitSet;

/**
 * Tests of {@link SharedFloorDiversifyingKnnCollectorManager}: its collector-creation policy, the
 * per-parent publication discipline that keeps the shared floor a valid bound of the merged
 * top-parents cutoff, and end-to-end recall of managed searches against stock block-join kNN.
 */
public class TestSharedFloorDiversifyingKnnCollectorManager extends LuceneTestCase {

  private static final String FIELD = "vector";
  private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.EUCLIDEAN;

  /** The floor must never receive a second score for the same parent. */
  public void testParentImprovementsAreNotRepublished() {
    // Two parents: bit 2 joins children 0,1 and bit 5 joins children 3,4.
    FixedBitSet parentBitSet = new FixedBitSet(8);
    parentBitSet.set(2);
    parentBitSet.set(5);
    GlobalKnnFloor floor = new GlobalKnnFloor(2);
    FloorAwareDiversifyingChildrenKnnCollector collector =
        new FloorAwareDiversifyingChildrenKnnCollector(
            new DiversifyingNearestChildrenKnnCollector(2, Integer.MAX_VALUE, null, parentBitSet),
            floor,
            parentBitSet,
            0.5f,
            2,
            1,
            1,
            true);

    // First parent observed at 0.9 and published at the gate-opening sync; the floor needs one
    // more parent before it is defined.
    collector.collect(1, 0.9f);
    assertEquals(Float.NEGATIVE_INFINITY, floor.floor(), 0.0f);

    // The same parent improves to 0.95. If improvements were republished, the floor's heap would
    // hold {0.9, 0.95} and define a floor of 0.9, a bound two entries of one parent cannot
    // justify. The floor must stay undefined.
    collector.incVisitedCount(1);
    collector.collect(0, 0.95f);
    assertEquals(
        "an improved score of an already published parent must not enter the floor",
        Float.NEGATIVE_INFINITY,
        floor.floor(),
        0.0f);

    // A second parent at 0.5 completes the floor: the k-th best of {0.9, 0.5} is 0.5.
    collector.incVisitedCount(1);
    collector.collect(4, 0.5f);
    assertEquals(0.5f, floor.floor(), 0.0f);

    // The delegate's local heap also holds both parents at {0.95, 0.5}, so the effective bound
    // agrees at 0.5.
    assertEquals(0.5f, collector.minCompetitiveSimilarity(), 0.0f);
  }

  /** A non-publishing collector reads the floor without feeding it. */
  public void testNonPublishingCollectorNeverFeedsTheFloor() {
    FixedBitSet parentBitSet = new FixedBitSet(8);
    parentBitSet.set(2);
    GlobalKnnFloor floor = new GlobalKnnFloor(1);
    FloorAwareDiversifyingChildrenKnnCollector collector =
        new FloorAwareDiversifyingChildrenKnnCollector(
            new DiversifyingNearestChildrenKnnCollector(1, Integer.MAX_VALUE, null, parentBitSet),
            floor,
            parentBitSet,
            0.5f,
            2,
            1,
            1,
            false);

    collector.collect(1, 0.9f);
    collector.incVisitedCount(1);
    collector.collect(0, 0.95f);
    assertEquals(Float.NEGATIVE_INFINITY, floor.floor(), 0.0f);
  }

  /** The manager's creation policy: activation threshold, per-leaf gates, publish-once. */
  public void testCollectorCreationPolicy() throws IOException {
    int k = 1000;
    try (Directory dir = newDirectory()) {
      indexFamilies(dir, 12, 2, 8, 3);
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        assertEquals("test setup: segment count", 3, reader.leaves().size());
        BitSetProducer parentsFilter = parentsFilter(reader);
        int totalDocs = reader.maxDoc();

        // Below the activation threshold the search is exactly stock: a plain collector.
        SharedFloorDiversifyingKnnCollectorManager inactive =
            new SharedFloorDiversifyingKnnCollectorManager(
                5, parentsFilter, new GlobalKnnFloor(5), 0.9f, 10);
        KnnCollector plain = inactive.newCollector(Integer.MAX_VALUE, null, reader.leaves().get(0));
        assertNotNull(plain);
        assertFalse(plain instanceof FloorAwareDiversifyingChildrenKnnCollector);

        SharedFloorDiversifyingKnnCollectorManager manager =
            new SharedFloorDiversifyingKnnCollectorManager(
                k, parentsFilter, new GlobalKnnFloor(k), 0.9f, 1, 16, 256);

        // Each leaf's collector gates at the leaf's pro-rata share of the merged top-k.
        for (LeafReaderContext context : reader.leaves()) {
          KnnCollector collector = manager.newCollector(Integer.MAX_VALUE, null, context);
          assertTrue(collector instanceof FloorAwareDiversifyingChildrenKnnCollector);
          FloorAwareDiversifyingChildrenKnnCollector decorated =
              (FloorAwareDiversifyingChildrenKnnCollector) collector;
          double leafShare = context.reader().maxDoc() / (double) totalDocs;
          assertEquals(
              "leaf " + context.ord + " must gate at its pro-rata share",
              SharedFloorKnnCollectorManager.perShardGate(k, leafShare),
              decorated.gateK());
          assertTrue(decorated.gateK() < k);
        }

        // A leaf is published at most once per query: the optimistic first pass publishes, the
        // second pass of the same leaf reads only, and a different leaf still publishes. A fresh
        // manager: the gate loop above already claimed this manager's leaves.
        SharedFloorDiversifyingKnnCollectorManager publishManager =
            new SharedFloorDiversifyingKnnCollectorManager(
                k, parentsFilter, new GlobalKnnFloor(k), 0.9f, 1, 16, 256);
        LeafReaderContext first = reader.leaves().get(0);
        LeafReaderContext second = reader.leaves().get(1);
        FloorAwareDiversifyingChildrenKnnCollector pass1 =
            (FloorAwareDiversifyingChildrenKnnCollector)
                publishManager.newOptimisticCollector(Integer.MAX_VALUE, null, first, 100);
        assertTrue("the first search of a leaf must publish", pass1.publishesToFloor());
        FloorAwareDiversifyingChildrenKnnCollector pass2 =
            (FloorAwareDiversifyingChildrenKnnCollector)
                publishManager.newCollector(Integer.MAX_VALUE, null, first);
        assertFalse(
            "a second search of the same leaf must not publish", pass2.publishesToFloor());
        FloorAwareDiversifyingChildrenKnnCollector otherLeaf =
            (FloorAwareDiversifyingChildrenKnnCollector)
                publishManager.newCollector(Integer.MAX_VALUE, null, second);
        assertTrue(
            "the first search of a different leaf must publish", otherLeaf.publishesToFloor());
      }
    }
  }

  /** A segment holding no parents yields no collector, matching the stock manager's contract. */
  public void testSegmentWithoutParentsYieldsNoCollector() throws IOException {
    try (Directory dir = newDirectory()) {
      try (IndexWriter writer =
          new IndexWriter(
              dir, new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
        for (int i = 0; i < 3; i++) {
          Document doc = new Document();
          doc.add(new KnnFloatVectorField(FIELD, randomVector(8), SIMILARITY));
          writer.addDocument(doc);
        }
        writer.commit();
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        BitSetProducer parentsFilter =
            new QueryBitSetProducer(new TermQuery(new Term("docType", "_parent")));
        SharedFloorDiversifyingKnnCollectorManager manager =
            new SharedFloorDiversifyingKnnCollectorManager(10, parentsFilter);
        assertNull(manager.newCollector(Integer.MAX_VALUE, null, reader.leaves().get(0)));
        assertNull(
            manager.newOptimisticCollector(Integer.MAX_VALUE, null, reader.leaves().get(0), 5));
      }
    }
  }

  /**
   * End to end: a managed block-join kNN search must keep stock recall, and the floor it leaves
   * behind must be a valid bound of the exact top-parents cutoff. The floor comparison is exact:
   * the publication discipline publishes each parent at most once with a value that cannot exceed
   * the parent's final score, so the floor can never exceed the true cutoff.
   */
  public void testManagedSearchKeepsRecallAndTheFloorStaysValid() throws IOException {
    int dim = 16;
    int families = 120;
    int k = 10;
    try (Directory dir = newDirectory()) {
      indexFamilies(dir, families, 3, dim, 4);
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        BitSetProducer parentsFilter = parentsFilter(reader);
        IndexSearcher searcher = new IndexSearcher(reader);

        double stockRecallSum = 0;
        double flooredRecallSum = 0;
        int queries = 5;
        for (int i = 0; i < queries; i++) {
          float[] query = randomVector(dim);
          Map<Integer, Float> truth = exactParentScores(reader, parentsFilter, query);
          float cutoff = kthBest(truth, k);

          TopDocs stock =
              searcher.search(
                  new DiversifyingChildrenFloatKnnVectorQuery(FIELD, query, null, k, parentsFilter),
                  k);
          stockRecallSum += parentRecall(stock, reader, parentsFilter, truth, k);

          GlobalKnnFloor floor = new GlobalKnnFloor(k);
          SharedFloorDiversifyingKnnCollectorManager manager =
              new SharedFloorDiversifyingKnnCollectorManager(
                  k, parentsFilter, floor, 0.9f, 1, 16, 256);
          TopDocs floored =
              searcher.search(
                  new SharedFloorParentJoinQuery(FIELD, query, k, parentsFilter, manager), k);
          flooredRecallSum += parentRecall(floored, reader, parentsFilter, truth, k);

          assertTrue(
              "the floor must engage once k parents have been observed",
              floor.floor() > Float.NEGATIVE_INFINITY);
          assertTrue(
              "floor " + floor.floor() + " must not exceed the true cutoff " + cutoff,
              floor.floor() <= cutoff);
        }
        double stockRecall = stockRecallSum / queries;
        double flooredRecall = flooredRecallSum / queries;
        assertTrue(
            "floor sharing lost recall: stock=" + stockRecall + " floored=" + flooredRecall,
            flooredRecall >= stockRecall - 0.05);
      }
    }
  }

  /**
   * The distributed wiring in miniature: two disjoint block-join indexes standing in for two
   * shards of one corpus, each searched through its own manager declaring half the corpus, both
   * wired to one shared floor. Merged recall must hold against stock search and the floor must
   * stay a valid bound of the merged top-parents cutoff.
   */
  public void testTwoShardsSharingOneFloor() throws IOException {
    int dim = 16;
    int k = 10;
    try (Directory dirA = newDirectory();
        Directory dirB = newDirectory()) {
      indexFamilies(dirA, 60, 3, dim, 2);
      indexFamilies(dirB, 60, 3, dim, 2);
      try (DirectoryReader readerA = DirectoryReader.open(dirA);
          DirectoryReader readerB = DirectoryReader.open(dirB)) {
        BitSetProducer filterA = parentsFilter(readerA);
        BitSetProducer filterB = parentsFilter(readerB);
        IndexSearcher searcherA = new IndexSearcher(readerA);
        IndexSearcher searcherB = new IndexSearcher(readerB);

        double stockRecallSum = 0;
        double flooredRecallSum = 0;
        int queries = 5;
        for (int i = 0; i < queries; i++) {
          float[] query = randomVector(dim);
          Map<Integer, Float> truth = exactParentScores(readerA, filterA, query);
          exactParentScores(readerB, filterB, query, 1 << 20, truth);
          float cutoff = kthBest(truth, k);

          TopDocs stockA =
              searcherA.search(
                  new DiversifyingChildrenFloatKnnVectorQuery(FIELD, query, null, k, filterA), k);
          TopDocs stockB =
              searcherB.search(
                  new DiversifyingChildrenFloatKnnVectorQuery(FIELD, query, null, k, filterB), k);
          stockRecallSum +=
              mergedParentRecall(stockA, stockB, readerA, filterA, readerB, filterB, truth, k);

          // One floor for the query; each shard's manager knows its share of the whole corpus.
          GlobalKnnFloor floor = new GlobalKnnFloor(k);
          TopDocs flooredA =
              searcherA.search(
                  new SharedFloorParentJoinQuery(
                      FIELD,
                      query,
                      k,
                      filterA,
                      new SharedFloorDiversifyingKnnCollectorManager(
                          k, filterA, floor, 0.9f, 1, 16, 256, 0.5f)),
                  k);
          TopDocs flooredB =
              searcherB.search(
                  new SharedFloorParentJoinQuery(
                      FIELD,
                      query,
                      k,
                      filterB,
                      new SharedFloorDiversifyingKnnCollectorManager(
                          k, filterB, floor, 0.9f, 1, 16, 256, 0.5f)),
                  k);
          flooredRecallSum +=
              mergedParentRecall(flooredA, flooredB, readerA, filterA, readerB, filterB, truth, k);

          assertTrue(floor.floor() > Float.NEGATIVE_INFINITY);
          assertTrue(
              "floor " + floor.floor() + " must not exceed the true cutoff " + cutoff,
              floor.floor() <= cutoff);
        }
        double stockRecall = stockRecallSum / queries;
        double flooredRecall = flooredRecallSum / queries;
        assertTrue(
            "cross-shard floor sharing lost recall: stock="
                + stockRecall
                + " floored="
                + flooredRecall,
            flooredRecall >= stockRecall - 0.05);
      }
    }
  }

  /**
   * A {@link DiversifyingChildrenFloatKnnVectorQuery} whose collector manager is wired to a shared
   * floor. A query instance carries single-execution state and must not be reused.
   */
  private static class SharedFloorParentJoinQuery extends DiversifyingChildrenFloatKnnVectorQuery {
    private final SharedFloorDiversifyingKnnCollectorManager manager;

    SharedFloorParentJoinQuery(
        String field,
        float[] target,
        int k,
        BitSetProducer parentsFilter,
        SharedFloorDiversifyingKnnCollectorManager manager) {
      super(field, target, null, k, parentsFilter);
      this.manager = manager;
    }

    @Override
    protected KnnCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
      return manager;
    }
  }

  private static BitSetProducer parentsFilter(IndexReader reader) throws IOException {
    BitSetProducer parentsFilter =
        new QueryBitSetProducer(new TermQuery(new Term("docType", "_parent")));
    CheckJoinIndex.check(reader, parentsFilter);
    return parentsFilter;
  }

  private float[] randomVector(int dim) {
    float[] vector = new float[dim];
    for (int i = 0; i < dim; i++) {
      vector[i] = (float) random().nextGaussian();
    }
    return vector;
  }

  /**
   * Index {@code families} block-join families of one parent with {@code childrenPerFamily}
   * vector-bearing children each, spread over {@code segments} segments.
   */
  private void indexFamilies(
      Directory dir, int families, int childrenPerFamily, int dim, int segments)
      throws IOException {
    try (IndexWriter writer =
        new IndexWriter(dir, new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
      int familiesPerSegment = (families + segments - 1) / segments;
      for (int family = 0; family < families; family++) {
        List<Document> block = new ArrayList<>();
        for (int child = 0; child < childrenPerFamily; child++) {
          Document doc = new Document();
          doc.add(new KnnFloatVectorField(FIELD, randomVector(dim), SIMILARITY));
          doc.add(new StoredField("family", family));
          block.add(doc);
        }
        Document parent = new Document();
        parent.add(new StringField("docType", "_parent", Field.Store.NO));
        parent.add(new StoredField("family", family));
        block.add(parent);
        writer.addDocuments(block);
        if ((family + 1) % familiesPerSegment == 0) {
          writer.flush();
        }
      }
      writer.commit();
    }
  }

  /** Exact per-parent max child similarity over the whole index, keyed by global parent id. */
  private Map<Integer, Float> exactParentScores(
      IndexReader reader, BitSetProducer parentsFilter, float[] query) throws IOException {
    return exactParentScores(reader, parentsFilter, query, 0, new HashMap<>());
  }

  private Map<Integer, Float> exactParentScores(
      IndexReader reader, BitSetProducer parentsFilter, float[] query, int keyOffset)
      throws IOException {
    return exactParentScores(reader, parentsFilter, query, keyOffset, new HashMap<>());
  }

  private Map<Integer, Float> exactParentScores(
      IndexReader reader,
      BitSetProducer parentsFilter,
      float[] query,
      int keyOffset,
      Map<Integer, Float> out)
      throws IOException {
    for (LeafReaderContext ctx : reader.leaves()) {
      FloatVectorValues values = ctx.reader().getFloatVectorValues(FIELD);
      if (values == null) {
        continue;
      }
      BitSet parentBitSet = parentsFilter.getBitSet(ctx);
      assertNotNull(parentBitSet);
      KnnVectorValues.DocIndexIterator iterator = values.iterator();
      for (int doc = iterator.nextDoc();
          doc != DocIdSetIterator.NO_MORE_DOCS;
          doc = iterator.nextDoc()) {
        float score = SIMILARITY.compare(query, values.vectorValue(iterator.index()));
        int parentKey = keyOffset + ctx.docBase + parentBitSet.nextSetBit(doc);
        out.merge(parentKey, score, Math::max);
      }
    }
    return out;
  }

  /** The k-th best score of the exact per-parent maxima; the true merged cutoff. */
  private static float kthBest(Map<Integer, Float> parentScores, int k) {
    List<Float> scores = new ArrayList<>(parentScores.values());
    scores.sort(Comparator.reverseOrder());
    return scores.get(Math.min(k, scores.size()) - 1);
  }

  /** The global parent ids of the exact top-k. */
  private static Set<Integer> topKParents(Map<Integer, Float> parentScores, int k) {
    List<Map.Entry<Integer, Float>> entries = new ArrayList<>(parentScores.entrySet());
    entries.sort(
        Map.Entry.<Integer, Float>comparingByValue(Comparator.reverseOrder())
            .thenComparingInt(Map.Entry::getKey));
    Set<Integer> top = new HashSet<>();
    for (int i = 0; i < k && i < entries.size(); i++) {
      top.add(entries.get(i).getKey());
    }
    return top;
  }

  /** Recall of one index's result against the exact top-k parents of the whole corpus. */
  private double parentRecall(
      TopDocs docs,
      IndexReader reader,
      BitSetProducer parentsFilter,
      Map<Integer, Float> truth,
      int k)
      throws IOException {
    Set<Integer> truthTop = topKParents(truth, k);
    int hits = 0;
    for (ParentScore parent : parentKeys(docs, reader, parentsFilter, 0)) {
      if (truthTop.contains(parent.parentKey())) {
        hits++;
      }
    }
    return hits / (double) k;
  }

  /** Merge two shards' results by score, keep the top k parents, and score recall against truth. */
  private double mergedParentRecall(
      TopDocs docsA,
      TopDocs docsB,
      IndexReader readerA,
      BitSetProducer filterA,
      IndexReader readerB,
      BitSetProducer filterB,
      Map<Integer, Float> truth,
      int k)
      throws IOException {
    Set<Integer> truthTop = topKParents(truth, k);
    List<ParentScore> merged = new ArrayList<>();
    merged.addAll(parentKeys(docsA, readerA, filterA, 0));
    merged.addAll(parentKeys(docsB, readerB, filterB, 1 << 20));
    // Keep at most k distinct parents by score; the collector already emits one child per parent,
    // so distinctness comes free, but the merge across shards must re-rank.
    merged.sort(Comparator.comparingDouble(ParentScore::score).reversed());
    int hits = 0;
    Set<Integer> seen = new HashSet<>();
    for (ParentScore parent : merged) {
      if (seen.add(parent.parentKey()) == false) {
        continue;
      }
      if (seen.size() > k) {
        break;
      }
      if (truthTop.contains(parent.parentKey())) {
        hits++;
      }
    }
    return hits / (double) k;
  }

  private record ParentScore(int parentKey, float score) {}

  /**
   * The global parent ids of a search result and their scores, mapping each returned best child
   * to its parent.
   */
  private List<ParentScore> parentKeys(
      TopDocs docs, IndexReader reader, BitSetProducer parentsFilter, int keyOffset)
      throws IOException {
    List<LeafReaderContext> leaves = reader.leaves();
    List<ParentScore> keys = new ArrayList<>();
    for (ScoreDoc scoreDoc : docs.scoreDocs) {
      int leafOrd = ReaderUtil.subIndex(scoreDoc.doc, leaves);
      LeafReaderContext ctx = leaves.get(leafOrd);
      BitSet parentBitSet = parentsFilter.getBitSet(ctx);
      assertNotNull(parentBitSet);
      int parentKey = keyOffset + ctx.docBase + parentBitSet.nextSetBit(scoreDoc.doc - ctx.docBase);
      keys.add(new ParentScore(parentKey, scoreDoc.score));
    }
    return keys;
  }
}

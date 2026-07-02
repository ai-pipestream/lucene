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

package org.apache.lucene.util.hnsw;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.util.ArrayUtil;
import org.junit.Before;

/** Tests for ResumableHnswGraphSearcher */
public class TestResumableHnswSearch extends HnswGraphTestCase<float[]> {

  private float[] target;

  @Before
  public void setup() {
    similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
    target = randomVector(16);
  }

  @Override
  VectorEncoding getVectorEncoding() {
    return VectorEncoding.FLOAT32;
  }

  @Override
  Query knnQuery(String field, float[] vector, int k) {
    return new org.apache.lucene.search.KnnFloatVectorQuery(field, vector, k);
  }

  @Override
  float[] randomVector(int dim) {
    return randomVector(random(), dim);
  }

  @Override
  float[] getTargetVector() {
    return target;
  }

  @Override
  KnnVectorValues vectorValues(int size, int dimension) {
    return MockVectorValues.fromValues(createRandomFloatVectors(size, dimension, random()));
  }

  @Override
  KnnVectorValues vectorValues(float[][] values) {
    return MockVectorValues.fromValues(values);
  }

  @Override
  KnnVectorValues vectorValues(LeafReader reader, String fieldName) throws IOException {
    FloatVectorValues vectorValues = reader.getFloatVectorValues(fieldName);
    float[][] vectors = new float[reader.maxDoc()][];
    for (int i = 0; i < vectorValues.size(); i++) {
      vectors[vectorValues.ordToDoc(i)] =
          ArrayUtil.copyOfSubArray(vectorValues.vectorValue(i), 0, vectorValues.dimension());
    }
    return MockVectorValues.fromValues(vectors);
  }

  @Override
  KnnVectorValues vectorValues(int size, int dimension, KnnVectorValues pregeneratedVectorValues, int pregeneratedOffset) {
    float[][] vectors = createRandomFloatVectors(size, dimension, random());
    for (int i = 0; i < pregeneratedVectorValues.size(); i++) {
        try {
            System.arraycopy(((FloatVectorValues)pregeneratedVectorValues).vectorValue(i), 0, vectors[i + pregeneratedOffset], 0, dimension);
        } catch (IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }
    return MockVectorValues.fromValues(vectors);
  }

  @Override
  KnnVectorValues circularVectorValues(int nDoc) {
    int dim = 16;
    float[][] vectors = new float[nDoc][dim];
    for (int i = 0; i < nDoc; i++) {
        vectors[i][i % dim] = 1.0f;
    }
    return MockVectorValues.fromValues(vectors);
  }

  @Override
  org.apache.lucene.document.Field knnVectorField(String name, float[] vector, VectorSimilarityFunction similarityFunction) {
    return new org.apache.lucene.document.KnnFloatVectorField(name, vector, similarityFunction);
  }

  public void testResumableSearchConsistency() throws IOException {
    int dim = 16;
    int n = 1000;
    float[][] vectors = createRandomFloatVectors(n, dim, random());
    KnnVectorValues v = vectorValues(vectors);
    HnswGraph graph = createGraph(v, 16, 100);
    
    float[] queryVector = randomVector(dim);
    RandomVectorScorer scorer = DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(similarityFunction, v, queryVector);
    
    // 1. Standard search for K=50
    int K = 50;
    TopKnnCollector standardCollector = new TopKnnCollector(K, Integer.MAX_VALUE);
    HnswGraphSearcher.search(scorer, standardCollector, graph, null);
    TopDocs standardResults = standardCollector.topDocs();
    
    // 2. Resumable search for K=50, but we request 10 at a time
    ResumableHnswGraphSearcher resumableSearcher = new ResumableHnswGraphSearcher(
        new NeighborQueue(K, true), HnswGraphSearcher.createBitSet(K, n));
    
    ResumableHnswGraphSearcher.SearchContext context = new ResumableHnswGraphSearcher.SearchContext(
        K, n, new int[] { graph.entryNode() }, 0);
    
    TopKnnCollector resumableCollector = new TopKnnCollector(K, Integer.MAX_VALUE);
    
    // Resume 5 times to get to 50
    for (int i = 0; i < 5; i++) {
        resumableSearcher.search(resumableCollector, scorer, graph, null, context);
    }
    TopDocs resumableResults = resumableCollector.topDocs();
    
    // Verify results are IDENTICAL
    assertEquals("Total results count should match", standardResults.scoreDocs.length, resumableResults.scoreDocs.length);
    for (int i = 0; i < standardResults.scoreDocs.length; i++) {
        assertEquals("Doc at index " + i + " should match", standardResults.scoreDocs[i].doc, resumableResults.scoreDocs[i].doc);
        assertEquals("Score at index " + i + " should match", standardResults.scoreDocs[i].score, resumableResults.scoreDocs[i].score, 0.00001f);
    }
  }

  private HnswGraph createGraph(KnnVectorValues v, int M, int beamWidth) throws IOException {
    RandomVectorScorerSupplier scorerSupplier = DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorerSupplier(similarityFunction, v);
    HnswGraphBuilder builder = HnswGraphBuilder.create(scorerSupplier, M, beamWidth, random().nextLong());
    return builder.build(v.size());
  }
}

package org.apache.lucene.util.hnsw;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;

/**
 * A resumable version of {@link HnswGraphSearcher} that maintains search state
 * to allow incremental Top-K retrieval (Streaming K).
 */
public class ResumableHnswGraphSearcher extends HnswGraphSearcher {

  /**
   * State object for a resumable search.
   */
  public static class SearchContext {
    final NeighborQueue candidates;
    final BitSet visited;
    final int[] eps;
    final int level;
    boolean initialized = false;

    public SearchContext(int k, int graphSize, int[] eps, int level) {
      this.candidates = new NeighborQueue(k, true);
      this.visited = HnswGraphSearcher.createBitSet(k, graphSize);
      this.eps = eps;
      this.level = level;
    }
    
    public void clear() {
        candidates.clear();
        visited.clear();
        initialized = false;
    }
  }

  public ResumableHnswGraphSearcher(NeighborQueue candidates, BitSet visited) {
    super(candidates, visited);
  }

  /**
   * Resumes a search using the provided context.
   */
  public void search(
      KnnCollector results,
      RandomVectorScorer scorer,
      HnswGraph graph,
      Bits acceptOrds,
      SearchContext context)
      throws IOException {

    int size = getGraphSize(graph);
    
    // Use context's state instead of scratch state
    if (this.bulkNodes == null || this.bulkNodes.length < graph.maxConn() * 2) {
        this.bulkNodes = new int[graph.maxConn() * 2];
        this.bulkScores = new float[graph.maxConn() * 2];
    }

    if (!context.initialized) {
        scoreEntryPoints(results, scorer, context.visited, context.eps, acceptOrds, context.candidates, new float[context.eps.length]);
        context.initialized = true;
    }

    if (results.earlyTerminated()) {
      return;
    }

    float minAcceptedSimilarity = Math.nextUp(results.minCompetitiveSimilarity());
    boolean shouldExploreMinSim = true;

    while (context.candidates.size() > 0 && results.earlyTerminated() == false) {
      float liveMinSimilarity = results.minCompetitiveSimilarity();
      if (liveMinSimilarity > minAcceptedSimilarity) {
        minAcceptedSimilarity = liveMinSimilarity;
        shouldExploreMinSim = true;
      }

      float topCandidateSimilarity = context.candidates.topScore();
      if (topCandidateSimilarity < minAcceptedSimilarity) {
        if (shouldExploreMinSim && Math.nextUp(topCandidateSimilarity) == minAcceptedSimilarity) {
          shouldExploreMinSim = false;
        } else {
          // Break but do NOT clear context; we are "paused"
          break;
        }
      }

      int topCandidateNode = context.candidates.pop();
      graphSeek(graph, context.level, topCandidateNode);
      int friendOrd;
      int numNodes = 0;
      while ((friendOrd = graphNextNeighbor(graph)) != NO_MORE_DOCS) {
        if (context.visited.getAndSet(friendOrd)) {
          continue;
        }
        if (results.earlyTerminated()) {
          break;
        }
        bulkNodes[numNodes++] = friendOrd;
      }

      numNodes = (int) Math.min(numNodes, results.visitLimit() - results.visitedCount());
      results.incVisitedCount(numNodes);
      
      if (numNodes > 0 && scorer.bulkScore(bulkNodes, bulkScores, numNodes) > results.minCompetitiveSimilarity()) {
        for (int i = 0; i < numNodes; i++) {
          int node = bulkNodes[i];
          float score = bulkScores[i];
          if (score >= minAcceptedSimilarity) {
            context.candidates.add(node, score);
            if (acceptOrds == null || acceptOrds.get(node)) {
              if (results.collect(node, score)) {
                float oldMinAcceptedSimilarity = minAcceptedSimilarity;
                minAcceptedSimilarity = Math.nextUp(results.minCompetitiveSimilarity());
                if (minAcceptedSimilarity > oldMinAcceptedSimilarity) {
                  shouldExploreMinSim = true;
                }
              }
            }
          }
        }
      }
    }
  }
}

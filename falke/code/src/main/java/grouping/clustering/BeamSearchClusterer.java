package grouping.clustering;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import com.carrotsearch.hppc.ObjectDoubleMap;
import com.carrotsearch.hppc.cursors.ObjectDoubleCursor;

import model.CPair;
import model.Concept;
import util.Beam;
import util.UnionFind;

/**
 * improves initial solution (connected components) by removing edges and finds
 * best modified solution with beam search
 * 
 * @author xxx
 *
 */
public class BeamSearchClusterer extends AbstractConceptClusterer {

	protected double posThreshold;
	protected int beamSize;
	protected int maxDepth;
	protected boolean parallel;

	protected CPair[] allPairs;
	protected double[] predictions;
	protected double fixedScore;
	protected Set<Concept> singleConcepts;
	protected Set<Concept> concepts;
	protected Set<CPair> allEdges;

	protected Beam<Solution> beam;
	protected Solution bestSolution;

	public BeamSearchClusterer(int beamSize, int maxDepth, boolean parallel) {
		this(0.5, beamSize, maxDepth, parallel);
	}

	public BeamSearchClusterer(double posThreshold, int beamSize, int maxDepth, boolean parallel) {
		this.posThreshold = posThreshold;
		this.beamSize = beamSize;
		this.maxDepth = maxDepth;
		this.parallel = parallel;
	}

	@Override
	public Set<List<Concept>> createClusters(Set<Concept> concepts, ObjectDoubleMap<CPair> predictions) {

		this.concepts = concepts;

		// initial solution
		Set<CPair> allEdges = this.getPositivePairs(predictions, this.posThreshold);
		this.allEdges = this.transReduction(concepts, allEdges);
		System.out.println("beam: branching factor " + this.allEdges.size());

		Solution init = new Solution();
		Set<List<Concept>> clusters = this.buildCluster(init);
		this.precomputeScoring(predictions, clusters);

		init.score = this.scoreClustering(clusters);
		this.bestSolution = init;
		System.out.println("beam: init " + init.score);

		// local beam search
		this.beam = new Beam<Solution>(this.beamSize);
		this.beam.add(init);
		int d = 0;
		do {
			System.out.println("beam: queue " + this.beam.size() + " depth " + d + " best " + this.bestSolution.score);
			List<Solution> neighbours = new ArrayList<Solution>();
			for (Solution sol : this.beam.pollAll()) {
				if (sol.removedEdges.size() < this.maxDepth)
					neighbours.addAll(this.generateNeighbours(sol));
			}
			if (neighbours.size() > 0) {
				Solution bestNeighbour = Collections.min(neighbours); // reverse
				if (bestNeighbour.score > this.bestSolution.score)
					this.bestSolution = bestNeighbour;
				this.beam.addAll(neighbours);
			}
			d++;
		} while (!this.beam.isEmpty());

		// final solution
		clusters = this.buildCluster(this.bestSolution);

		System.out.println("beam: best " + this.bestSolution.score);

		return clusters;

	}

	protected List<Solution> generateNeighbours(Solution prevSolution) {

		List<Solution> sols;
		Set<CPair> nextEdges = new HashSet<CPair>(this.allEdges);
		nextEdges.removeAll(prevSolution.removedEdges);

		if (parallel) {
			sols = nextEdges.parallelStream().map(e -> this.buildNeighbour(prevSolution, e))
					.collect(Collectors.toList());
		} else {
			sols = nextEdges.stream().map(e -> this.buildNeighbour(prevSolution, e)).collect(Collectors.toList());
		}
		return sols;
	}

	protected Solution buildNeighbour(Solution prevSolution, CPair edgeToRemove) {
		Solution newSolution = new Solution(prevSolution, edgeToRemove);
		Set<List<Concept>> clusters = this.buildCluster(newSolution);
		newSolution.score = this.scoreClustering(clusters);
		return newSolution;
	}

	protected Set<List<Concept>> buildCluster(Solution sol) {
		Set<CPair> reducedEdges = new HashSet<CPair>(allEdges);
		reducedEdges.removeAll(sol.removedEdges);
		Set<List<Concept>> clusters = this.buildTransClosureClusters(concepts, reducedEdges);
		return clusters;
	}

	protected Set<CPair> transReduction(Set<Concept> concepts, Set<CPair> pairs) {
		Set<CPair> reduction = new HashSet<CPair>(pairs);
		UnionFind<Concept> unionFind = new UnionFind<Concept>(concepts);
		for (CPair pair : pairs) {
			Concept c1 = unionFind.find(pair.c1);
			Concept c2 = unionFind.find(pair.c2);
			if (c1 == c2)
				reduction.remove(pair);
			else
				unionFind.union(pair.c1, pair.c2);
		}
		return reduction;
	}

	// precompute objective function for pairs that will never change
	// -> single concept clusters
	protected void precomputeScoring(ObjectDoubleMap<CPair> predictions, Set<List<Concept>> clusters) {

		this.singleConcepts = new HashSet<Concept>();
		for (List<Concept> cluster : clusters)
			if (cluster.size() == 1)
				this.singleConcepts.add(cluster.get(0));

		this.fixedScore = 0;
		List<CPair> pairs = new ArrayList<CPair>();
		for (ObjectDoubleCursor<CPair> p : predictions) {
			if (this.singleConcepts.contains(p.key.c1) || this.singleConcepts.contains(p.key.c2)) {
				this.fixedScore += 1 - p.value;
			} else {
				pairs.add(p.key);
			}
		}

		this.allPairs = new CPair[pairs.size()];
		this.predictions = new double[pairs.size()];
		for (int i = 0; i < pairs.size(); i++) {
			this.allPairs[i] = pairs.get(i);
			this.predictions[i] = predictions.get(this.allPairs[i]);
		}
	}

	protected double scoreClustering(Set<List<Concept>> clusters) {
		double score = this.fixedScore;
		Set<CPair> pairs = this.convertClusters(clusters);
		for (int i = 0; i < allPairs.length; i++) {
			if (pairs.contains(allPairs[i]))
				score += predictions[i];
			else
				score += (1 - predictions[i]);
		}
		return score;
	}

	private class Solution implements Comparable<Solution> {
		public Set<CPair> removedEdges;
		public double score;

		public Solution() {
			this.removedEdges = new HashSet<CPair>();
		}

		public Solution(Solution other, CPair newEdge) {
			this.removedEdges = new HashSet<CPair>(other.removedEdges);
			this.removedEdges.add(newEdge);
		}

		@Override
		public int compareTo(Solution o) {
			return Double.compare(o.score, this.score);
		}
	}
}

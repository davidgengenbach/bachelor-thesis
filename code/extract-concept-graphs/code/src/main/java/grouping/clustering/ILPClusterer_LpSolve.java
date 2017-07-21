package grouping.clustering;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

import com.carrotsearch.hppc.ObjectDoubleMap;
import com.carrotsearch.hppc.ObjectIntHashMap;
import com.carrotsearch.hppc.ObjectIntMap;
import com.carrotsearch.hppc.cursors.ObjectCursor;

import lpsolve.LpSolve;
import lpsolve.LpSolveException;
import model.CPair;
import model.Concept;

/**
 * creates best clustering given the classifications
 * 
 * finds most probable relation that is an equivalence relation over the
 * concepts
 * 
 * @author xxx
 *
 */
public class ILPClusterer_LpSolve extends AbstractConceptClusterer {

	private ObjectIntMap<CPair> varMap;

	@Override
	public Set<List<Concept>> createClusters(Set<Concept> concepts, ObjectDoubleMap<CPair> predictions) {

		// init
		this.varMap = this.initVarMap(predictions);
		LpSolve problem = null;
		try {
			problem = this.createILP(concepts, predictions);
		} catch (LpSolveException e) {
			System.err.println("Error during ILP creation");
			e.printStackTrace();
			return null;
		}

		// solve
		double[] sol = null;
		try {
			problem.setVerbose(3); // 3 = quiet
			int res = problem.solve();
			if (res != 0) {
				System.err.println("lp_solve failed, running again");
				problem.writeLp("data/grouping/failed.lp");
			}
			// System.out.println("ILP solved");

			// this is now 0-based!
			sol = problem.getPtrVariables();

		} catch (LpSolveException e) {
			System.err.println("Error solving ILP");
			e.printStackTrace();
			return null;
		}

		// TODO: clusters look fine now, but this still fails!
		// boolean trans = this.isTransitive(sol, concepts.size());
		// if (!trans)
		// System.err.println("Error solving ILP: not transitive!");

		// create corresponding clusters
		Set<CPair> pairsToMerge = new HashSet<CPair>();
		for (ObjectCursor<CPair> p : predictions.keys()) {
			double merge = sol[varMap.get(p.value) - 1];
			if (merge > 0)
				pairsToMerge.add(p.value);
		}

		Set<List<Concept>> clusters = this.buildTransClosureClusters(concepts, pairsToMerge);

		return clusters;
	}

	/**
	 * creates a mapping between concept pairs and indices used in the ILP
	 * 
	 * @param predictions
	 */
	private ObjectIntMap<CPair> initVarMap(ObjectDoubleMap<CPair> predictions) {
		// lp_solve uses 1-based indices!
		ObjectIntMap<CPair> varMap = new ObjectIntHashMap<CPair>();
		int x = 1;
		for (ObjectCursor<CPair> p : predictions.keys()) {
			varMap.put(p.value, x);
			x++;
		}
		return varMap;
	}

	/**
	 * defines the ILP for the given concepts and predictions
	 * 
	 * @param concepts
	 *            Set of concepts to be clustered
	 * @param predictions
	 *            Predictions for all pairs of concepts
	 * @return ILP
	 * @throws LpSolveException
	 */
	private LpSolve createILP(Set<Concept> concepts, ObjectDoubleMap<CPair> predictions) throws LpSolveException {

		// define ILP
		int nbVars = varMap.size(); // primary only
		LpSolve problem = lpsolve.LpSolve.makeLp(0, 2 * nbVars);
		problem.setMaxim();

		/*
		 * variables (all binary)
		 * primary: x_p if pair p used
		 * aux: x_p+n if pair p is not used
		 */
		for (int i = 1; i <= 2 * nbVars; i++)
			problem.setBinary(i, true);

		/*
		 * objective function
		 * \sum_pairs s(p) * x_p + (1 - s(p)) (1 - x_p)
		 * (1 - x_p) => x_p+n
		 */
		double[] objCoeff = new double[2 * nbVars + 1];
		for (ObjectCursor<CPair> p : predictions.keys()) {
			int i = varMap.get(p.value);
			double v = predictions.get(p.value);
			if (Double.isNaN(v))
				System.out.println("ERROR");
			objCoeff[i] = v;
			objCoeff[i + nbVars] = 1 - v;
		}
		problem.setObjFn(objCoeff);

		/*
		 * constraints - aux variables
		 * -> one and only one of them has to be 1
		 */
		for (ObjectCursor<CPair> p : predictions.keys()) {
			double[] consCoeff = new double[2 * nbVars + 1];
			int i = varMap.get(p.value);
			consCoeff[i] = 1;
			consCoeff[i + nbVars] = 1;
			problem.addConstraint(consCoeff, LpSolve.EQ, 1);
		}

		/*
		 * constraints - transitivity
		 * -> selection of pairs has to be transitive
		 */
		for (Concept ci : concepts) {
			for (Concept cj : concepts) {
				for (Concept ck : concepts) {
					if (ci != cj && ci != ck && cj != ck) {

						try {
							Set<String> triple = new HashSet<String>();
							triple.add(ci.name);
							triple.add(cj.name);
							triple.add(ck.name);

							int pair_ij = varMap.get(new CPair(ci, cj));
							int pair_ik = varMap.get(new CPair(ci, ck));
							int pair_jk = varMap.get(new CPair(cj, ck));

							double[] consCoeff = new double[2 * nbVars + 1];
							consCoeff[pair_ij] = -1;
							consCoeff[pair_jk] = -1;
							consCoeff[pair_ik] = 1;
							problem.addConstraint(consCoeff, LpSolve.GE, -1);

							consCoeff = new double[2 * nbVars + 1];
							consCoeff[pair_ij] = -1;
							consCoeff[pair_jk] = 1;
							consCoeff[pair_ik] = -1;
							problem.addConstraint(consCoeff, LpSolve.GE, -1);

							consCoeff = new double[2 * nbVars + 1];
							consCoeff[pair_ij] = 1;
							consCoeff[pair_jk] = -1;
							consCoeff[pair_ik] = -1;
							problem.addConstraint(consCoeff, LpSolve.GE, -1);

						} catch (NullPointerException e) {
							// wrong pair ordering
						}
					}
				}
			}
		}

		return problem;
	}

	private boolean isTransitive(double[] ilpSolution, int n) {

		double[][] adjacencyMatrix = new double[n][n];
		int k = 0;
		for (int i = 0; i < n; i++) {
			adjacencyMatrix[i][i] = 1;
			for (int j = i + 1; j < n; j++) {
				adjacencyMatrix[i][j] = ilpSolution[k];
				adjacencyMatrix[j][i] = ilpSolution[k];
				k++;
			}
		}

		INDArray m = new NDArray(adjacencyMatrix);
		INDArray m2 = m.mmul(m);

		System.out.println(m);

		for (int i = 0; i < m.rows(); i++) {
			for (int j = 0; j < m.columns(); j++) {
				if (m2.getDouble(i, j) > 0 && m.getDouble(i, j) == 0) {
					System.out.println(i + " " + j + " " + m2.getDouble(i, j) + " " + m.getDouble(i, j));
					return false;
				}
			}
		}

		return true;
	}
}

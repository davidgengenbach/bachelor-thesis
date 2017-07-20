package grouping.exp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.Sets;

import edu.stanford.nlp.util.Triple;
import grouping.clf.FeatureExtractor;
import grouping.clf.ThresholdClassifier;
import model.Concept;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;

public class RunExperimentsClassification {

	public static String dataFile = "data/grouping/wiki_training.tsv";
	public static String featureFile = "data/grouping/wiki_training.arff";

	public static void main(String[] args) throws Exception {

		// load data
		List<Triple<Concept, Concept, Boolean>> conceptPairs = loadData(dataFile);
		Instances data = createFeatures(conceptPairs, null, featureFile);

		printDataStatistics(data);
		System.exit(0);

		testSimThresholdClassifier(data, conceptPairs);

		// test classifier
		// Logistic clfLogistic = (Logistic)
		// SerializationHelper.read("models/grouping_Logistic.model");
		//
		// Evaluation eval = new Evaluation(data);
		// eval.evaluateModel(clfLogistic, data); // training data!
		//
		// System.out.println(eval.toClassDetailsString());
		// System.out.println(eval.toMatrixString());
	}

	public static void testSimThresholdClassifier(Instances data, List<Triple<Concept, Concept, Boolean>> conceptPairs)
			throws Exception {

		Map<ThresholdClassifier, Evaluation> evals = new HashMap<ThresholdClassifier, Evaluation>();
		for (int i = 0; i < 9; i++) {

			ThresholdClassifier clf = new ThresholdClassifier();
			clf.buildClassifier(data, i);

			Evaluation eval = new Evaluation(data);
			eval.evaluateModel(clf, data);
			evals.put(clf, eval);

			NumberFormat f = new DecimalFormat("#00.00");
			System.out.println(f.format(eval.precision(1) * 100) + "\t" + f.format(eval.recall(1) * 100) + "\t"
					+ f.format(eval.fMeasure(1) * 100) + "\t" + clf);

			// printPredications(eval, data, conceptPairs, 1, 0);
		}

		compareClassifier(evals, data, conceptPairs);
	}

	public static void compareClassifier(Map<ThresholdClassifier, Evaluation> evals, Instances data,
			List<Triple<Concept, Concept, Boolean>> pairs) {

		System.out.println();
		System.out.println("pairs in eval:\t" + data.numInstances());
		int pos = data.attributeStats(data.classIndex()).nominalCounts[1];
		System.out.println("- positive: \t" + pos);
		System.out.println();

		// missed by all
		Set<Integer> missedByAll = null;
		for (Evaluation eval : evals.values()) {
			Set<Integer> missed = getIdsForLabel(eval.predictions(), 1, 0);
			if (missed.size() > 0) {
				if (missedByAll == null)
					missedByAll = missed;
				else
					missedByAll = Sets.intersection(missedByAll, missed);
			}
		}
		System.out.println("missed by all: \t" + missedByAll.size());

		// correct by at least one
		Set<Integer> correctByOne = null;
		Set<Integer> correctByAll = null;
		for (Evaluation eval : evals.values()) {
			Set<Integer> correct = getIdsForLabel(eval.predictions(), 1, 1);
			if (correct.size() < pos) {
				if (correctByOne == null)
					correctByOne = correct;
				else
					correctByOne = Sets.union(correctByOne, correct);
				if (correctByAll == null)
					correctByAll = correct;
				else
					correctByAll = Sets.intersection(correctByAll, correct);
			}
		}
		System.out.println("correct one: \t" + correctByOne.size());
		System.out.println("correct all: \t" + correctByAll.size());

		printPredictions(data, pairs, missedByAll);

	}

	public static Set<Integer> getIdsForLabel(ArrayList<Prediction> predictions, double trueLabel, double predLabel) {
		Set<Integer> missed = new HashSet<Integer>();
		for (int i = 0; i < predictions.size(); i++) {
			NominalPrediction p = (NominalPrediction) predictions.get(i);
			if (p.actual() == trueLabel && p.predicted() == predLabel)
				missed.add(i);
		}
		return missed;
	}

	public static void printPredictions(Instances data, List<Triple<Concept, Concept, Boolean>> pairs,
			Set<Integer> ids) {
		for (Integer i : ids) {
			System.out.println(data.instance(i) + "\t" + pairs.get(i).first() + " <-> " + pairs.get(i).second());
		}
	}

	public static void printDataStatistics(Instances data) {
		System.out.println();
		System.out.println(data.toSummaryString());
		System.out.println();
		System.out.println(data.attribute(data.classIndex()).name());
		System.out.println(data.attributeStats(data.classIndex()));
		System.out.println();
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			// System.out.println(data.attribute(i).name());
			// System.out.println(data.attributeStats(i).numericStats);
		}
	}

	// create features for each pair
	// can either load data from arff file or compute from scratch
	public static Instances createFeatures(List<Triple<Concept, Concept, Boolean>> pairs, String readFromFile,
			String saveToFile) throws Exception {

		Instances data = null;
		FeatureExtractor ex = null;

		// read from file if it exists
		if (readFromFile != null) {

			BufferedReader reader = new BufferedReader(new FileReader(readFromFile));
			ArffReader arff = new ArffReader(reader);
			data = arff.getData();
			data.setClassIndex(data.numAttributes() - 1);

			ObjectInputStream in = new ObjectInputStream(new FileInputStream(readFromFile.replace("arff", "ser")));
			ex = (FeatureExtractor) in.readObject();
			in.close();

		} else {

			ex = new FeatureExtractor(pairs, "exp");
			data = ex.computeFeatures(null, "exp");

			// save features to file
			if (saveToFile != null) {

				ArffSaver saver = new ArffSaver();
				saver.setInstances(data);
				saver.setFile(new File(saveToFile));
				saver.writeBatch();

				ObjectOutputStream out = new ObjectOutputStream(
						new FileOutputStream(saveToFile.replace("arff", "ser")));
				out.writeObject(ex);
				out.close();
			}
		}

		return data;
	}

	public static List<Triple<Concept, Concept, Boolean>> loadData(String fileName) throws IOException {
		List<Triple<Concept, Concept, Boolean>> pairs = new ArrayList<Triple<Concept, Concept, Boolean>>();
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		String line = null;
		while ((line = reader.readLine()) != null) {
			String[] cols = line.split("\t");
			if (cols[0].equals("topic")) // header
				continue;
			Concept c1 = new Concept(cols[3]);
			Concept c2 = new Concept(cols[4]);
			boolean label = cols[5].equals("1") ? true : false;
			Triple<Concept, Concept, Boolean> pair = new Triple<Concept, Concept, Boolean>(c1, c2, label);
			pairs.add(pair);
		}
		reader.close();
		return pairs;
	}

}

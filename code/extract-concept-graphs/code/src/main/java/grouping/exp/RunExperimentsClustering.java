package grouping.exp;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import com.carrotsearch.hppc.ObjectDoubleHashMap;
import com.carrotsearch.hppc.ObjectDoubleMap;
import com.carrotsearch.hppc.cursors.ObjectCursor;

import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Triple;
import grouping.clustering.CompClusterer;
import grouping.clustering.AbstractConceptClusterer;
import model.CPair;
import model.Concept;
import util.AdjustedRandIndex;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader.ArffReader;

public class RunExperimentsClustering {

	public static String dataFile = "data/grouping/training.tsv";
	public static String featureFile = "data/grouping/training.arff";

	public static List<Triple<Concept, Concept, Boolean>> conceptPairs;
	public static Map<String, Map<Concept, Integer>> conceptClustering;

	public static void main(String[] args) throws Exception {

		Scanner in = new Scanner(System.in);
		in.nextLine();
		in.close();
		System.out.println("starting");

		// load data
		loadData(dataFile);
		Instances data = createFeatures(featureFile);

		// classifiers
		List<Classifier> classifiers = new ArrayList<Classifier>();
		classifiers.add((Classifier) SerializationHelper.read("models/grouping_Logistic.model"));
		// classifiers.add((Classifier)
		// SerializationHelper.read("models/grouping_RandomForest.model"));

		for (Classifier clf : classifiers) {

			// apply classifier
			Map<String, ObjectDoubleMap<CPair>> predictionsByTopic = applyClassifier(clf, data, false);

			// clustering per topic
			List<AbstractConceptClusterer> clusterers = new ArrayList<AbstractConceptClusterer>();
			clusterers.add(new CompClusterer(0.5));
			// clusterers.add(new BeamSearchClusterer(0.5, 10, 20));

			for (AbstractConceptClusterer clusterer : clusterers) {

				double avgARI = 0;
				for (String topic : predictionsByTopic.keySet()) {

					Set<Concept> concepts = new HashSet<Concept>();
					for (ObjectCursor<CPair> p : predictionsByTopic.get(topic).keys()) {
						concepts.add(p.value.c1);
						concepts.add(p.value.c2);
					}

					// apply clustering algorithm
					Set<List<Concept>> clusters = clusterer.createClusters(concepts, predictionsByTopic.get(topic));
					Map<Concept, Integer> predClustering = convertToGroupNumbers(clusters);

					// compare to gold
					Map<Concept, Integer> goldClustering = conceptClustering.get(topic);

					List<Concept> conceptsOrdered = new ArrayList<Concept>(concepts);
					int[] goldArray = convertToArray(conceptsOrdered, goldClustering);
					int[] predArray = convertToArray(conceptsOrdered, predClustering);

					double ari = AdjustedRandIndex.compute(goldArray, predArray);
					avgARI += ari;
				}
				avgARI /= predictionsByTopic.size();

				NumberFormat f = new DecimalFormat("#0.0000");
				String[] params = { clf.getClass().getSimpleName(), clusterer.getClass().getSimpleName(),
						f.format(avgARI) };
				System.out.println(StringUtils.join(params, "\t"));
			}
		}

	}

	public static int[] convertToArray(List<Concept> concepts, Map<Concept, Integer> groupAssignments) {
		int[] array = new int[concepts.size()];
		for (int i = 0; i < concepts.size(); i++)
			array[i] = groupAssignments.get(concepts.get(i));
		return array;
	}

	public static Map<Concept, Integer> convertToGroupNumbers(Set<List<Concept>> clusters) {
		Map<Concept, Integer> groupNumbers = new HashMap<Concept, Integer>();
		int i = 0;
		for (List<Concept> cluster : clusters) {
			for (Concept c : cluster)
				groupNumbers.put(c, i);
			i++;
		}
		return groupNumbers;
	}

	public static Map<String, ObjectDoubleMap<CPair>> applyClassifier(Classifier clf, Instances data, boolean verbose) {

		Evaluation eval = null;
		try {
			eval = new Evaluation(data);
			eval.evaluateModel(clf, data);
			if (verbose) {
				System.out.println(clf);
				System.out.println(eval.toClassDetailsString());
				System.out.println(eval.toMatrixString());
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		Map<String, ObjectDoubleMap<CPair>> predictionsByTopic = new HashMap<String, ObjectDoubleMap<CPair>>();
		for (int i = 0; i < conceptPairs.size(); i++) {
			NominalPrediction p = (NominalPrediction) eval.predictions().get(i);
			String topic = conceptPairs.get(i).first().type.substring(0, 4);
			if (!predictionsByTopic.keySet().contains(topic))
				predictionsByTopic.put(topic, new ObjectDoubleHashMap<CPair>());
			predictionsByTopic.get(topic).put(new CPair(conceptPairs.get(i).first(), conceptPairs.get(i).second()),
					p.distribution()[1]);
		}
		return predictionsByTopic;
	}

	// load features from arff file
	public static Instances createFeatures(String readFromFile) throws IOException {

		if (readFromFile != null) {
			BufferedReader reader = new BufferedReader(new FileReader(readFromFile));
			ArffReader arff = new ArffReader(reader);
			Instances data = arff.getData();
			data.setClassIndex(data.numAttributes() - 1);
			return data;
		}

		return null;
	}

	public static void loadData(String fileName) throws IOException {

		conceptPairs = new ArrayList<Triple<Concept, Concept, Boolean>>();
		conceptClustering = new HashMap<String, Map<Concept, Integer>>();

		Map<String, Concept> concepts = null;
		String lastTopic = null;
		Map<Concept, Integer> clustering = null;

		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		String line = null;
		while ((line = reader.readLine()) != null) {

			String[] cols = line.split("\t");
			if (cols[0].equals("topic")) // header
				continue;

			if (lastTopic == null || !lastTopic.equals(cols[0])) {
				if (clustering != null)
					conceptClustering.put(lastTopic, clustering);
				clustering = new HashMap<Concept, Integer>();
				concepts = new HashMap<String, Concept>();
			}
			lastTopic = cols[0];

			Concept c1 = concepts.get(cols[1]);
			if (c1 == null) {
				c1 = new Concept(cols[3]);
				c1.type = cols[0] + "|" + cols[1];
				concepts.put(cols[1], c1);
			}
			Concept c2 = concepts.get(cols[2]);
			if (c2 == null) {
				c2 = new Concept(cols[4]);
				c2.type = cols[0] + "|" + cols[2];
				concepts.put(cols[2], c2);
			}
			boolean label = cols[5].equals("1") ? true : false;
			int groupC1 = Integer.parseInt(cols[6]);
			int groupC2 = Integer.parseInt(cols[7]);

			Triple<Concept, Concept, Boolean> pair = new Triple<Concept, Concept, Boolean>(c1, c2, label);
			conceptPairs.add(pair);

			clustering.put(c1, groupC1);
			clustering.put(c2, groupC2);
		}
		if (clustering != null)
			conceptClustering.put(lastTopic, clustering);

		reader.close();

	}

}

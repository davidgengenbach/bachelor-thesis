package scoring.concepts.exp;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.deeplearning4j.berkeley.StringUtils;

import grouping.clf.sim.ConceptSimilarityMeasure;
import grouping.clf.sim.JaccardDistance;
import model.Concept;
import model.ConceptMap;
import model.ExtractionResult;
import model.Proposition;
import model.io.ConceptMapReader;
import model.io.Format;
import scoring.concepts.features.FeatureExtractor;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class CreateTrainingData {

	public static final String folderName = "data/WikiCMaps/test_system";
	public static final String goldFolderName = "data/WikiCMaps/test";
	public static final String name = "grouping-noun10conjarg2-sim5log-gopt";
	public static final boolean test = true;

	public static Map<Integer, List<String[]>> annotations;

	public static void main(String[] args) throws IOException, ClassNotFoundException {

		Logger.getRootLogger().setLevel(Level.INFO);

		FeatureExtractor ex = null;
		if (!test) {
			ex = new FeatureExtractor();
		} else {
			String fileName = "data/selection/" + name + ".fe.ser";
			if (folderName.contains("Wiki"))
				fileName = fileName.replace("selection/", "selection/wiki_");
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));
			ex = (FeatureExtractor) in.readObject();
			in.close();
		}
		ex.init(folderName, name + ".graph_features.tsv");

		File folder = new File(folderName);
		for (File clusterFolder : folder.listFiles()) {
			if (clusterFolder.isDirectory()) {

				int topic = Integer.parseInt(clusterFolder.getName());

				// load concepts
				String serFileName = folderName + "/" + clusterFolder.getName() + "/" + name + ".groups.ser";
				ExtractionResult res = ExtractionResult.load(serFileName);
				System.out.println(clusterFolder.getName() + " " + res.concepts.size());

				// extract features
				ex.collectFeatures(res.groupedConcepts, topic);

				// create gold labels
				Map<Concept, Boolean> labelsBinary = getBinaryLabels(res.concepts, topic, 0.9);
				ex.addLabels(labelsBinary, "label_binary");
				// Map<Concept, Double> labelsLikert = getLikertLabels(res,
				// topic, 12);
				// Map<Concept, Double> labels = mergeLabels(labelsBinary,
				// labelsLikert);
				// ex.addLabels(labels, "label_likert");
			}
		}

		Instances data = ex.getFeatures();
		System.out.println(data.toSummaryString());

		// save data
		String arffName = "data/selection/" + name + ".arff";
		if (test)
			arffName = "data/selection/test_" + name + ".arff";
		if (folderName.contains("Wiki"))
			arffName = arffName.replace("selection/", "selection/wiki_");
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File(arffName));
		saver.writeBatch();

		if (!test) {
			ex.startTest();
			String fileName = "data/selection/" + name + ".fe.ser";
			if (folderName.contains("Wiki"))
				fileName = fileName.replace("selection/", "selection/wiki_");
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName));
			out.writeObject(ex);
			out.close();
		}

	}

	private static Map<Concept, Double> mergeLabels(Map<Concept, Boolean> binaryLabels,
			Map<Concept, Double> likertLabels) {
		Map<Concept, Double> labels = new HashMap<Concept, Double>();
		int count = 0;
		for (Concept c : likertLabels.keySet()) {
			if (binaryLabels.get(c))
				labels.put(c, 5.0);
			else
				labels.put(c, likertLabels.get(c));
			if (labels.get(c) > 0)
				count++;
		}
		System.out.println("non-zero: " + count + " (" + (count / (float) labels.size()) + ")");
		return labels;
	}

	private static Map<Concept, Boolean> getBinaryLabels(List<Concept> concepts, int topic, double threshold) {

		File goldFile = new File(goldFolderName + "/" + topic + "/" + topic + ".cmap");
		ConceptMap goldMap = ConceptMapReader.readFromFile(goldFile, Format.TSV);

		Map<Concept, Boolean> labels = new HashMap<Concept, Boolean>();
		ConceptSimilarityMeasure sim = new JaccardDistance();

		int posLabels = 0;
		for (Concept c : concepts) {
			boolean label = false;
			for (Concept g : goldMap.getConcepts()) {
				int goldToken = g.name.split("\\s+").length;
				if (sim.computeSimilarity(c, g) > threshold && c.tokenList.size() <= 2 * goldToken) {
					label = true;
					posLabels++;
					break;
				}
			}
			labels.put(c, label);
		}

		System.out.println("pos: " + posLabels + ", total: " + concepts.size());

		return labels;
	}

	private static Map<Concept, Double> getLikertLabels(ExtractionResult res, int topic, int max_dist) {

		loadAnnotations("data/selection/amt_scores.tsv");

		int count = 0;
		Map<Concept, Set<Double>> scores = new HashMap<Concept, Set<Double>>();
		for (Proposition p : res.propositions) {
			String pText = p.sourceConcept.name + " " + p.relationPhrase + " " + p.targetConcept.name;
			int bestMatch = max_dist + 1;
			String[] bestProp = null;
			for (String[] amtProp : annotations.get(topic)) {
				String amtPropText = amtProp[1] + " " + amtProp[2] + " " + amtProp[3];
				int dist = StringUtils.editDistance(pText, amtPropText.toLowerCase());
				if (dist < bestMatch) {
					bestMatch = dist;
					bestProp = amtProp;
				}
				if (dist == 0)
					break;
			}
			double score = 0;
			if (bestProp != null) {
				score = Double.parseDouble(bestProp[4]);
				count += 2;
			}
			scores.putIfAbsent(p.sourceConcept, new HashSet<Double>());
			scores.get(p.sourceConcept).add(score);
			scores.putIfAbsent(p.targetConcept, new HashSet<Double>());
			scores.get(p.targetConcept).add(score);
		}

		System.out.println("pos: " + count + ", total: " + res.concepts.size());

		Map<Concept, Double> labels = new HashMap<Concept, Double>();
		for (Entry<Concept, Set<Double>> e : scores.entrySet())
			labels.put(e.getKey(), Collections.max(e.getValue()));

		return labels;
	}

	private static Map<Integer, List<String[]>> loadAnnotations(String fileName) {
		if (annotations != null)
			return annotations;
		annotations = new HashMap<Integer, List<String[]>>();
		try {
			for (String line : FileUtils.readLines(new File(fileName), Charsets.UTF_8)) {
				String[] cols = line.split("\t");
				int topic = Integer.parseInt(cols[0]);
				annotations.putIfAbsent(topic, new ArrayList<String[]>());
				annotations.get(topic).add(cols);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return annotations;
	}

}

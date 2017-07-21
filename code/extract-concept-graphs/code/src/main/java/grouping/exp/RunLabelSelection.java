package grouping.exp;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

import eval.Result;
import eval.RunEvaluation;
import eval.matcher.Match;
import eval.matcher.StemSWMatch;
import eval.metrics.ConceptMatchMetric;
import grouping.LabelSelector;
import grouping.LabelSelector.Strategy;
import model.Concept;
import model.ConceptMap;
import model.ExtractionResult;
import model.PToken;
import model.Proposition;
import model.io.ConceptMapReader;
import model.io.ConceptMapWriter;
import model.io.Format;
import preprocessing.NonUIMAPreprocessor;
import util.features.Feature;
import util.features.FeatureContainer;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class RunLabelSelection {

	public static String folderName = "data/CMapSummaries/train_system";
	public static String name = "grouping-noun10conjarg2-sim5log-gopt-agg";

	public static void main(String[] args) throws Exception {

		File folder = new File(folderName);
		for (File topicFolder : folder.listFiles()) {

			System.out.println();
			System.out.println(topicFolder.getName());
			System.out.println();

			// load
			String path = folderName + "/" + topicFolder.getName() + "/" + name;
			ExtractionResult res = ExtractionResult.load(path + ".groups.ser");

			// load gold map
			ConceptMap goldMap = ConceptMapReader.readFromFile(new File(folderName.replace("train_system", "train")
					+ "/" + topicFolder.getName() + "/" + topicFolder.getName() + ".cmap"), Format.TSV);

			evalRecall(res, "baseline", goldMap);

			// select label
			ExtractionResult resSelect = selectClusterLabel(res,
					new LabelSelector(Strategy.LONGEST, .3, Strategy.FREQ, Strategy.SHORTEST));
			// resSelect = mergeDuplicateClusters(resSelect,
			// RunLabelSelection::getLemmaKey);
			// resSelect = mergeDuplicateClusters(resSelect,
			// RunLabelSelection::getLabelKey);
			// resSelect = selectClusterLabel(resSelect, new
			// LabelSelector(Strategy.SHORTEST));
			evalRecall(resSelect, ".3", goldMap);

			// analyze again

			writeExtractionResults(resSelect, path);
			writePropsFile(resSelect, path + "-props.cmap");
		}
	}

	private static ExtractionResult selectClusterLabel(ExtractionResult res, LabelSelector sel) {

		List<Concept> concepts = new ArrayList<Concept>();
		Map<Concept, Concept> changes = new HashMap<Concept, Concept>();
		for (List<Concept> cluster : res.groupedConcepts) {
			Concept label = sel.select(cluster);
			concepts.add(label);
			if (label != cluster.get(0)) {
				changes.put(cluster.get(0), label);
				cluster.remove(label);
				cluster.add(0, label);
			}
		}

		List<Proposition> props = new ArrayList<Proposition>();
		for (Proposition p : res.propositions) {
			if (changes.containsKey(p.sourceConcept))
				p.sourceConcept = changes.get(p.sourceConcept);
			if (changes.containsKey(p.targetConcept))
				p.targetConcept = changes.get(p.targetConcept);
			if (p.sourceConcept != p.targetConcept)
				props.add(p);
		}

		ExtractionResult newRes = new ExtractionResult(concepts, props, res.groupedConcepts);
		return newRes;
	}

	private static ExtractionResult selectClusterLabelModel(ExtractionResult res, String modelPath) throws Exception {

		Classifier clf = (Classifier) SerializationHelper.read(modelPath);

		List<Concept> concepts = new ArrayList<Concept>();
		Map<Concept, Concept> changes = new HashMap<Concept, Concept>();
		for (List<Concept> cluster : res.groupedConcepts) {

			Concept label = null;
			List<Concept> candidates = cluster.stream().filter(c -> c.weight > 0).collect(Collectors.toList());

			if (candidates.size() == 1)
				label = candidates.get(0);
			else {

				// features
				FeatureContainer<Concept> cont = new FeatureContainer<Concept>();

				double maxFreq = candidates.stream().mapToDouble(c -> c.weight).max().getAsDouble();
				Set<Concept> mostFreq = candidates.stream().filter(c -> c.weight == maxFreq)
						.collect(Collectors.toSet());

				int maxLength = candidates.stream().mapToInt(c -> c.name.length()).max().getAsInt();
				Set<Concept> longest = candidates.stream().filter(c -> c.name.length() == maxLength)
						.filter(c -> c.weight > 0).collect(Collectors.toSet());

				int minLength = candidates.stream().mapToInt(c -> c.name.length()).min().getAsInt();
				Set<Concept> shortest = candidates.stream().filter(c -> c.name.length() == minLength)
						.filter(c -> c.weight > 0).collect(Collectors.toSet());

				for (Concept c : candidates) {
					cont.add(c, new Feature<>("~label", false));
					cont.add(c, new Feature<Boolean>("is_freq", mostFreq.contains(c)));
					cont.add(c, new Feature<Boolean>("is_long", longest.contains(c)));
					cont.add(c, new Feature<Boolean>("is_short", shortest.contains(c)));
					cont.add(c, new Feature<Integer>("cluster_size", cluster.size()));
					cont.add(c, new Feature<Double>("freq", c.weight));
					cont.add(c, new Feature<Double>("freq_rel", c.weight / cluster.size()));
					cont.add(c, new Feature<Double>("most_freq_rel", maxFreq / cluster.size()));
					cont.add(c, new Feature<Integer>("length", c.name.length()));
				}
				Instances features = cont.createInstances(cont.getFeatures());
				features.setClassIndex(features.numAttributes() - 1);

				// predict
				double max = 0;
				for (Concept c : candidates) {
					Instance feat = cont.createInstance(cont.getFeatures(), c);
					feat.setDataset(features);
					double[] pred = clf.distributionForInstance(feat);
					if (pred[1] > max) {
						max = pred[1];
						label = c;
					}
				}
			}

			// update results
			concepts.add(label);
			if (label != cluster.get(0)) {
				changes.put(cluster.get(0), label);
				cluster.remove(label);
				cluster.add(0, label);
			}

		}

		List<Proposition> props = new ArrayList<Proposition>();
		for (Proposition p : res.propositions) {
			if (changes.containsKey(p.sourceConcept))
				p.sourceConcept = changes.get(p.sourceConcept);
			if (changes.containsKey(p.targetConcept))
				p.targetConcept = changes.get(p.targetConcept);
			props.add(p);
		}

		ExtractionResult newRes = new ExtractionResult(concepts, props, res.groupedConcepts);
		return newRes;
	}

	private static Map<Concept, List<Concept>> evalRecall(ExtractionResult res, String name, ConceptMap goldMap) {

		InputStream sw = RunEvaluation.class.getResourceAsStream("/lists/stopwords_en_eval.txt");
		Match match = new StemSWMatch(sw);
		ConceptMatchMetric metric = new ConceptMatchMetric(match);

		ConceptMap map = new ConceptMap(name);
		for (Concept c : res.concepts)
			map.addConcept(c);
		for (Proposition p : res.propositions)
			map.addProposition(p);

		Result result = metric.compare(map, goldMap);
		System.out.println(result + "\t" + name);

		// map gold concepts to cluster
		Map<Concept, List<Concept>> matched = new HashMap<Concept, List<Concept>>();
		for (Concept cg : goldMap.getConcepts()) {
			List<Concept> matchingCluster = null;
			for (List<Concept> cluster : res.groupedConcepts) {
				for (Concept c : cluster) {
					if (match.isMatch(cg.name.toLowerCase(), c.name.toLowerCase())) {
						if (matchingCluster == null || cluster.size() > matchingCluster.size())
							matchingCluster = cluster;
					}
				}
			}
			if (matchingCluster != null)
				matched.put(cg, matchingCluster);
		}

		return matched;
	}

	// create map after label selection
	private static void writeExtractionResults(ExtractionResult res, String fileName) {
		ExtractionResult.save(res, fileName + ".groups.ser");
		ConceptMap map = new ConceptMap("");
		for (Concept c : res.concepts) {
			map.addConcept(c);
			map.addProposition(new Proposition(c, c, "dummy"));
		}
		ConceptMapWriter.writeToFile(map, new File(fileName + ".cmap"), Format.TSV);
	}

	// create props file
	private static void writePropsFile(ExtractionResult res, String fileName) {
		ConceptMap map = new ConceptMap("");
		for (Proposition p : res.propositions) {
			map.addConcept(p.sourceConcept);
			map.addConcept(p.targetConcept);
			map.addProposition(p);
		}
		ConceptMapWriter.writeToFile(map, new File(fileName), Format.TSV);
	}

	private static ExtractionResult mergeDuplicateClusters(ExtractionResult res, Function<Concept, String> keyFn) {

		Map<String, Set<List<Concept>>> groupedClusters = new HashMap<String, Set<List<Concept>>>();
		for (List<Concept> cluster : res.groupedConcepts) {
			Concept rep = cluster.get(0);
			String key = keyFn.apply(rep);
			groupedClusters.putIfAbsent(key, new HashSet<List<Concept>>());
			groupedClusters.get(key).add(cluster);
		}

		Set<List<Concept>> clusters = new HashSet<List<Concept>>();
		List<Concept> concepts = new ArrayList<Concept>();
		Map<Concept, Concept> changes = new HashMap<Concept, Concept>();
		for (Set<List<Concept>> groupedCluster : groupedClusters.values()) {
			if (groupedCluster.size() > 1) {
				List<Concept> main = groupedCluster.iterator().next();
				Concept rep = main.get(0);
				for (List<Concept> other : groupedCluster) {
					if (other != main) {
						main.addAll(other);
						changes.put(other.get(0), rep);
					}
				}
				clusters.add(main);
				concepts.add(rep);
			} else {
				List<Concept> cluster = groupedCluster.iterator().next();
				clusters.add(cluster);
				concepts.add(cluster.get(0));
			}
		}

		List<Proposition> props = new ArrayList<Proposition>();
		for (Proposition p : res.propositions) {
			if (changes.containsKey(p.sourceConcept))
				p.sourceConcept = changes.get(p.sourceConcept);
			if (changes.containsKey(p.targetConcept))
				p.targetConcept = changes.get(p.targetConcept);
			props.add(p);
		}

		ExtractionResult newRes = new ExtractionResult(concepts, props, clusters);
		return newRes;
	}

	private static String getLemmaKey(Concept c) {
		String key = "";
		for (PToken t : c.tokenList) {
			if (t.pos.startsWith("N") && t.pos.contains("P")) {
				t = NonUIMAPreprocessor.getInstance().lemmatize(t);
			}
			key += t.lemma.toLowerCase() + " ";
		}
		return key;
	}

	private static String getLabelKey(Concept c) {
		return c.name.toLowerCase().trim();
	}

}

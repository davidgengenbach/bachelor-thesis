package scoring.concepts.exp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import eval.Evaluation;
import eval.matcher.StemSWMatch;
import eval.metrics.ConceptMatchMetric;
import eval.metrics.Metric;
import mapbuilding.MapBuilderBase;
import mapbuilding.MapBuilderILP;
import model.Concept;
import model.ConceptMap;
import model.ExtractionResult;
import model.Proposition;
import model.io.ConceptMapReader;
import model.io.Format;
import pipeline.ConceptMapMining;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.instance.RemoveRange;
import weka.filters.unsupervised.instance.RemoveWithValues;

/**
 * Leave-one-out cross-validation for concept selection regression
 * 
 * @author xxx
 *
 */
public class RunCVRegression {

	public static final String goldFolderName = "data/CMapSummaries/train";
	public static final String serFolderName = "data/CMapSummaries/train_system";
	public static final String name = "grouping-noun10conjarg2-sim5log-gopt-agg";
	public static final String arffFileName = "data/selection/grouping-noun10conjarg2-sim5log-gopt-agg.arff";

	public static AbstractClassifier clf = new RandomForest();

	public static boolean evalWithILP = true;
	public static int ilpTimeout = 120;

	public static void main(String[] args) throws Exception {

		AbstractClassifier[] clfs = { new ZeroR(), new LinearRegression(), new RandomForest() };

		for (AbstractClassifier x : clfs) {
			clf = x;

			// 1) load arff
			ArffReader arff = new ArffReader(new BufferedReader(new FileReader(arffFileName)));
			Instances data = arff.getData();

			Map<String, Integer> dataMap = new HashMap<String, Integer>(70000);
			for (int i = 0; i < data.numInstances(); i++) {
				Instance ins = data.instance(i);
				String key = ((int) ins.value(1)) + "_" + ((int) ins.value(0));
				dataMap.put(key, i);
			}

			Remove r = new Remove();
			r.setAttributeIndices("1-2,223"); // id, topic, binary label
			r.setInputFormat(data);
			data = Filter.useFilter(data, r);
			data.setClassIndex(data.numAttributes() - 1);

			// 1.5) load candidates
			Map<String, ExtractionResult> candidates = new HashMap<String, ExtractionResult>();
			List<String> topics = new ArrayList<String>();
			File folder = new File(serFolderName);
			for (File clusterFolder : folder.listFiles()) {
				if (clusterFolder.isDirectory()) {
					topics.add(clusterFolder.getName());
					String serFileName = serFolderName + "/" + clusterFolder.getName() + "/" + name + ".groups.ser";
					ExtractionResult res = ExtractionResult.load(serFileName);
					candidates.put(clusterFolder.getName(), res);
				}
			}

			// 2) leave-one-out cross-validation
			List<Evaluation> evals = new ArrayList<Evaluation>();
			Metric m = new ConceptMatchMetric(
					new StemSWMatch(RunCVRegression.class.getResourceAsStream("/lists/stopwords_en_eval.txt")));
			Metric[] metrics = { m };

			// f = test-topic
			for (int f = 0; f < topics.size(); f++) {
				String testTopic = topics.get(f);
				List<Concept> testConcepts = candidates.get(testTopic).concepts;
				System.out.println("cv test-fold " + f + " " + topics.get(f) + " " + testConcepts.size());

				// 3) compile training data -> topics w/o f
				String range = ""; // -> 1-based!
				for (Concept c : candidates.get(testTopic).concepts)
					range += (dataMap.get(testTopic + "_" + c.id) + 1) + ",";
				RemoveRange remRange = new RemoveRange();
				remRange.setInstancesIndices(range.substring(0, range.length() - 1));
				remRange.setInputFormat(data);
				Instances dataTrain = Filter.useFilter(data, remRange);

				RemoveWithValues removeZero = new RemoveWithValues();
				removeZero.setAttributeIndex("last");
				removeZero.setSplitPoint(0.00001);
				removeZero.setMatchMissingValues(true);
				removeZero.setInputFormat(dataTrain);
				dataTrain = Filter.useFilter(dataTrain, removeZero);

				ReplaceMissingValues repMissing = new ReplaceMissingValues();
				repMissing.setInputFormat(dataTrain);
				dataTrain = Filter.useFilter(dataTrain, repMissing);

				// 4) train model

				clf.buildClassifier(dataTrain);

				weka.classifiers.evaluation.Evaluation clfEval = new weka.classifiers.evaluation.Evaluation(dataTrain);
				clfEval.evaluateModel(clf, dataTrain);

				// 3) evaluate on test fold

				File goldFile = new File(goldFolderName + "/" + testTopic + "/" + testTopic + ".cmap");
				ConceptMap goldMap = ConceptMapReader.readFromFile(goldFile, Format.TSV);
				Evaluation eval = new Evaluation(testTopic, goldMap);
				eval.addMetrics(metrics);

				// apply classifier
				for (Concept c : testConcepts) {
					Instance ins = data.instance(dataMap.get(testTopic + "_" + c.id));
					double score = clf.classifyInstance(ins);
					c.weight = score;
				}

				// derive ranking
				Collections.sort(testConcepts);
				int[] topks = { 25, 50 };
				for (int topk : topks) {
					ConceptMap map = new ConceptMap("top-" + topk);
					map.addConcept(testConcepts.subList(0, topk));
					for (Concept c : map.getConcepts())
						map.addProposition(new Proposition(c, c, "dummy"));
					eval.addConceptMap(map);
				}

				// derive concept map
				if (evalWithILP) {
					MapBuilderILP.ilpTimeout = ilpTimeout;
					MapBuilderBase builder = new MapBuilderILP();
					builder.setParent(new ConceptMapMining() {
						@Override
						public String getName() {
							return "ilp";
						}

						@Override
						public int getMaxConcepts() {
							return 25;
						}
					});
					builder.setData(candidates.get(testTopic));
					builder.buildMap();
					eval.addConceptMap(builder.getConceptMap());
				}

				// run evaluation
				eval.run();
				evals.add(eval);
			}

			Evaluation topicEval = Evaluation.getAvgResults(evals, false).get(0);
			System.out.println(clf.getClass().getSimpleName());
			System.out.println(topicEval.printResults());
		}
	}

}

package grouping.clf;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.stanford.nlp.util.Triple;
import grouping.clf.sim.ConceptSimilarityMeasure;
import grouping.clf.sim.EditDistance;
import grouping.clf.sim.JaccardDistance;
import grouping.clf.sim.SemilarSentenceMeasure;
import grouping.clf.sim.StemMatch;
import grouping.clf.sim.StringMatch;
import grouping.clf.sim.WordBasedMeasure;
import grouping.clf.sim.WordEmbeddingDistance;
import grouping.clf.sim.WordEmbeddingDistance.EmbeddingType;
import model.Concept;
import model.PToken;
import preprocessing.NonUIMAPreprocessor;
import semilar.config.ConfigManager;
import semilar.sentencemetrics.CorleyMihalceaComparer;
import semilar.tools.semantic.WordNetSimilarity.WNSimMeasure;
import semilar.wordmetrics.LSAWordMetric;
import util.Muter;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * computes features for concept coreference classification
 * 
 * @author xxx
 *
 */
public class FeatureExtractor implements Serializable {

	private static final long serialVersionUID = 1L;
	private static final boolean LOG = true;

	private static final String cacheFile = "data/grouping/grouping-features.$topic$.cache.ser";
	private static Map<String, Instance> cache;

	public static List<String> classes;
	public static WordEmbeddingDistance glove;
	public static WordEmbeddingDistance word2vec;

	private Instances data;
	private ReplaceMissingValues repMissing;
	private Discretize discretize;
	private RemoveUseless removeUseless;

	public FeatureExtractor(List<Triple<Concept, Concept, Boolean>> pairs, String topic) throws Exception {
		SingletonFeatureExtractor.setFeatureExtractor(this);

		classes = new ArrayList<String>();
		classes.add("no merge");
		classes.add("merge");

		this.data = this.computeFeatures(pairs, topic);
	}

	public Instances computeFeatures(List<Triple<Concept, Concept, Boolean>> pairs, String topic) throws Exception {

		if (pairs == null)
			return this.data;
		if (cache == null)
			cache = this.loadCache(cacheFile.replace("$topic$", topic));

		if (LOG)
			Logger.getLogger(this.getClass().getName()).log(Level.INFO, "Initializing feature extractor");

		// initialize similarity features
		List<ConceptSimilarityMeasure> measures = new ArrayList<ConceptSimilarityMeasure>();
		measures.add(new StringMatch());
		measures.add(new StemMatch());
		measures.add(new EditDistance());
		measures.add(new JaccardDistance());

		measures.add(new WordBasedMeasure(WNSimMeasure.RES));
		ConfigManager.setSemilarDataRootFolder("../semilar/resources/");
		measures.add(new WordBasedMeasure(new LSAWordMetric("LSA-MODEL-TASA-LEMMATIZED-DIM300")));
		measures.add(new SemilarSentenceMeasure(new CorleyMihalceaComparer(0.3f, false, "NONE", "par")));

		// if (glove == null)
		// glove = new WordEmbeddingDistance(EmbeddingType.GLOVE, 300, true);
		// measures.add(glove);
		if (word2vec == null)
			word2vec = new WordEmbeddingDistance(EmbeddingType.WORD2VEC, 300, true);
		measures.add(word2vec);

		if (LOG)
			Logger.getLogger(this.getClass().getName()).log(Level.INFO, "Initializing feature extractor - done");

		// compute features per instance
		if (LOG)
			Logger.getLogger(this.getClass().getName()).log(Level.INFO, "Extracting features");

		Instances data = this.createDataSet(measures);

		if (LOG)
			Logger.getLogger(this.getClass().getName()).log(Level.INFO,
					"nb of features: " + (data.numAttributes() - 1));
		for (Triple<Concept, Concept, Boolean> pair : pairs) {
			data.add(this.createInstance(pair.first(), pair.second(), pair.third(), measures));
		}
		if (LOG)
			Logger.getLogger(this.getClass().getName()).log(Level.INFO, "Extracting features - done");

		this.saveCache(cache, cacheFile.replace("$topic$", topic));

		// apply filter
		if (LOG)
			Logger.getLogger(this.getClass().getName()).log(Level.INFO, "Appyling filter");
		if (this.repMissing == null) {
			repMissing = new ReplaceMissingValues();
			repMissing.setInputFormat(data);
		}
		data = Filter.useFilter(data, repMissing);

		// if (this.discretize == null) {
		// discretize = new Discretize();
		// String[] discOpt = { "-D", "-precision", "6" };
		// discretize.setOptions(discOpt);
		// discretize.setInputFormat(data);
		// }
		// data = Filter.useFilter(data, discretize);

		if (this.removeUseless == null) {
			removeUseless = new RemoveUseless();
			removeUseless.setInputFormat(data);
		}
		data = Filter.useFilter(data, removeUseless);

		if (LOG)
			Logger.getLogger(this.getClass().getName()).log(Level.INFO, "Appyling filter - done");

		return data;

	}

	private Instance createInstance(Concept c1, Concept c2, Boolean label, List<ConceptSimilarityMeasure> measures) {

		// cache lookup
		Instance instance = this.getFromCache(c1, c2);
		if (instance != null)
			return instance;

		// preprocess (for standalone training data only)
		if (c1.tokenList == null || c1.tokenList.size() == 0)
			NonUIMAPreprocessor.getInstance().preprocess(c1);
		if (c2.tokenList == null || c2.tokenList.size() == 0)
			NonUIMAPreprocessor.getInstance().preprocess(c2);

		List<Double> vals = new ArrayList<Double>();
		Concept[] concepts = { c1, c2 };

		// similarities
		for (int i = 0; i < measures.size(); i++) {
			ConceptSimilarityMeasure measure = measures.get(i);
			double sim = Muter.callMuted(measure::computeSimilarity, c1, c2);
			vals.add(sim);
		}

		// part of speech
		Set<String>[] tagsByConcept = new Set[2];
		for (int i = 0; i < 2; i++) {
			Set<String> tags = new HashSet<String>();
			// collect tags
			for (PToken t : concepts[i].tokenList) {
				tags.add(t.pos);
				if (Arrays.binarySearch(posList, t.pos) < 0)
					System.err.println("Unkown PoS: " + t.pos);
			}
			tagsByConcept[i] = tags;
		}
		// AND combos
		for (String pos : posList) {
			if (tagsByConcept[0].contains(pos) && tagsByConcept[1].contains(pos))
				vals.add(1.0);
			else
				vals.add(0.0);
		}

		// named entities
		for (int i = 0; i < 2; i++) {
			// collect tags
			Set<String> tags = new HashSet<String>();
			for (PToken t : concepts[i].tokenList) {
				String neTag = t.neTag != null ? t.neTag : "O";
				tags.add(neTag);
				if (Arrays.binarySearch(neList, neTag) < 0)
					System.err.println("Unkown NE: " + neTag);
			}
			tagsByConcept[i] = tags;
		}
		// AND combos
		for (String ne : neList) {
			if (tagsByConcept[0].contains(ne) && tagsByConcept[1].contains(ne))
				vals.add(1.0);
			else
				vals.add(0.0);
		}
		if ((tagsByConcept[0].contains("PERSON") || tagsByConcept[0].contains("ORGANIZATION")
				|| tagsByConcept[0].contains("LOCATION"))
				&& (tagsByConcept[1].contains("PERSON") || tagsByConcept[1].contains("ORGANIZATION")
						|| tagsByConcept[1].contains("LOCATION")))
			vals.add(1.0);
		else
			vals.add(0.0);

		// abbreviations
		if (this.hasAcronymMatch(c1, c2) || this.hasAcronymMatch(c2, c1))
			vals.add(1.0);
		else
			vals.add(0.0);

		// label
		if (label != null && label == true)
			vals.add(1.0);
		else
			vals.add(0.0);

		double[] valArray = new double[vals.size()];
		for (int i = 0; i < vals.size(); i++)
			valArray[i] = vals.get(i);

		instance = new DenseInstance(1.0, valArray);
		this.addToCache(c1, c2, instance);

		return instance;
	}

	private Instances createDataSet(List<ConceptSimilarityMeasure> measures) {

		ArrayList<Attribute> atts = new ArrayList<Attribute>();
		List<String> boolAtt = new ArrayList<String>();
		boolAtt.add("0");
		boolAtt.add("1");

		// similarities
		for (ConceptSimilarityMeasure measure : measures)
			atts.add(new Attribute(measure.getName()));

		// part of speech
		for (String pos : posList)
			atts.add(new Attribute("pos_both_" + pos, boolAtt));

		// named entities
		for (String ne : neList)
			atts.add(new Attribute("ne_both_" + ne, boolAtt));
		atts.add(new Attribute("ne_both_any", boolAtt));

		// abbreviations
		atts.add(new Attribute("acronym_match", boolAtt));

		// label
		atts.add(new Attribute("class", classes));

		Instances data = new Instances("data", atts, 0);
		data.setClassIndex(data.numAttributes() - 1);

		return data;
	}

	private boolean hasAcronymMatch(Concept c1, Concept c2) {
		// TODO: look for lists that can be used
		String[] t1 = c1.name.toLowerCase().split("[^A-Za-z]");
		String a = "";
		for (String w : t1)
			if (w.length() >= 1)
				a += w.substring(0, 1);
		if (a.length() < 2)
			return false;
		String l2 = c2.name.toLowerCase();
		return l2.contains(a);
	}

	private static String[] posList = { "!", "#", "$", "''", "(", ")", ",", "-LRB-", "-RRB-", ".", ":", "?", "CC", "CD",
			"DT", "EX", "FW", "IN", "JJ", "JJR", "JJRJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "NP", "PDT",
			"POS", "PRP", "PRP$", "PRT", "RB", "RBR", "RBS", "RN", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN",
			"VBP", "VBZ", "VP", "WDT", "WH", "WP", "WP$", "WRB", "``" };
	private static String[] neList = { "LOCATION", "O", "ORGANIZATION", "PERSON" };

	// CACHE
	private Instance getFromCache(Concept c1, Concept c2) {
		String cacheKey = c1.name.toLowerCase().trim() + " <-> " + c2.name.toLowerCase().trim();
		if (cache.containsKey(cacheKey))
			return cache.get(cacheKey);
		else {
			cacheKey = c2.name.toLowerCase().trim() + " <-> " + c1.name.toLowerCase().trim();
			if (cache.containsKey(cacheKey))
				return cache.get(cacheKey);
			else {
				return null;
			}
		}
	}

	private void addToCache(Concept c1, Concept c2, Instance instance) {
		String cacheKey = c1.name.toLowerCase().trim() + " <-> " + c2.name.toLowerCase().trim();
		cache.put(cacheKey, instance);
	}

	private Map<String, Instance> loadCache(String fileName) {
		System.out.println("Loading cache");
		Map<String, Instance> cache = new HashMap<String, Instance>();
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));
			cache = (Map<String, Instance>) in.readObject();
			in.close();
		} catch (FileNotFoundException e) {
			// ok, will return empty map then
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("--- done: " + cache.size());
		return cache;
	}

	private void saveCache(Map<String, Instance> cache, String fileName) {
		System.out.println("Saving cache");
		try {
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName));
			out.writeObject(cache);
			out.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("--- done: " + cache.size());
	}

}

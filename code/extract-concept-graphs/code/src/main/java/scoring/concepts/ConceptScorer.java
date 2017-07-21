package scoring.concepts;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import model.Concept;
import pipeline.CmmComponent;
import pipeline.Extractor;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class ConceptScorer extends CmmComponent {

	private static final String arffFileName = "data/selection/wiki_grouping-noun10conjarg2-lemma.arff";
	private static final String modelFile = "models/wiki_scoring_noun10conjarg2_lemma_Logistic_RepZ30Disc.model";

	@Override
	public void processCollection() {

		String topic = this.parent.getTargetLocation().substring(this.parent.getTargetLocation().lastIndexOf("/") + 1);

		// get concepts
		Extractor exComp = this.parent.getPrevExtractor(this);
		List<Concept> concepts = exComp.getConcepts();
		this.parent.log(this, "scoring components: " + concepts.size());

		// load arff with features
		Map<String, Instance> features = new HashMap<String, Instance>();
		try {
			ArffReader arff = new ArffReader(new BufferedReader(new FileReader(arffFileName)));
			Instances data = arff.getData();
			data.setClassIndex(data.numAttributes() - 2);
			String[] ids = new String[data.numInstances()];
			for (int i = 0; i < data.numInstances(); i++) {
				Instance ins = data.instance(i);
				ids[i] = ((int) ins.value(0)) + "_" + ((int) ins.value(1));
			}
			Remove rem = new Remove();
			rem.setAttributeIndices("1-2"); // last = likert
			rem.setInputFormat(data);
			data = Filter.useFilter(data, rem);
			for (int i = 0; i < data.numInstances(); i++)
				features.put(ids[i], data.instance(i));
		} catch (Exception e) {
			e.printStackTrace();
		}

		// load classifier
		AbstractClassifier clf = null;
		try {
			if (modelFile.startsWith("dummy"))
				clf = this.createDummyClassifier(modelFile);
			else
				clf = (AbstractClassifier) SerializationHelper.read(modelFile);
		} catch (Exception e) {
			e.printStackTrace();
		}

		// apply classifier
		for (Concept c : concepts) {
			Instance f = features.get(c.id + "_" + topic);
			try {
				double[] pred = clf.distributionForInstance(f);
				c.weight = pred[1];
				if (Double.isNaN(c.weight))
					System.err.println("NaN!");
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		Collections.sort(concepts);
	}

	// creates a dummy classifier that just returns one of the features
	private AbstractClassifier createDummyClassifier(String name) {

		String f = name.split("-")[1];
		int id = -1;

		if (f.equals("freq"))
			id = 14; // relative concept frequency
		if (f.equals("cfidf"))
			id = 16; // concept frequency * max term idf
		if (f.equals("pagerank"))
			id = 32; // page rank

		final int fId = id;
		return new AbstractClassifier() {
			@Override
			public void buildClassifier(Instances data) throws Exception {

			}

			@Override
			public double[] distributionForInstance(Instance features) {
				double[] pred = new double[2];
				pred[1] = features.value(fId);
				if (Double.isNaN(pred[1]))
					pred[1] = 0;
				return pred;
			}
		};
	}

}

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
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveRange;

public class ConceptScorerRanking extends CmmComponent {

	private static final String arffFileName = "data/selection/test_grouping-noun10conjarg2-sim5log-gopt.arff";
	private static final String modelFile = "models/scoring_noun10conjarg2_sim5log-gopt_SVMRankC30_RepDisc.model";

	@Override
	public void processCollection() {

		String topic = this.parent.getTargetLocation().substring(this.parent.getTargetLocation().lastIndexOf("/") + 1);

		// get concepts
		Extractor exComp = this.parent.getPrevExtractor(this);
		List<Concept> concepts = exComp.getConcepts();
		this.parent.log(this, "scoring components: " + concepts.size());

		// load arff with features
		Instances data = null;
		Map<Integer, Integer> conceptId2Index = new HashMap<Integer, Integer>();
		try {
			// load full dataset
			ArffReader arff = new ArffReader(new BufferedReader(new FileReader(arffFileName)));
			data = arff.getData();
			// remove likert label
			if (data.attribute(data.numAttributes() - 1).name().equals("~_label_likert")) {
				Remove r = new Remove();
				r.setAttributeIndices("last");
				r.setInputFormat(data);
				data = Filter.useFilter(data, r);
			}
			data.setClassIndex(data.numAttributes() - 1);
			// subset to current topic
			String range = ""; // -> 1-based!
			for (int i = 0; i < data.numInstances(); i++) {
				if (topic.equals(Integer.toString((int) data.instance(i).value(1)))) {
					range += (i + 1) + ",";
				}
			}
			RemoveRange remRange = new RemoveRange();
			remRange.setInstancesIndices(range.substring(0, range.length() - 1));
			remRange.setInvertSelection(true);
			remRange.setInputFormat(data);
			data = Filter.useFilter(data, remRange);
			// build id mapping
			for (int i = 0; i < data.numInstances(); i++) {
				conceptId2Index.put((int) data.instance(i).value(0), i);
			}
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
		double[][] preds = null;
		try {
			preds = clf.distributionsForInstances(data);
		} catch (Exception e) {
			e.printStackTrace();
		}
		for (Concept c : concepts)
			c.weight = preds[conceptId2Index.get(c.id)][1];

		Collections.sort(concepts);
	}

	// creates a dummy classifier that just returns one of the features
	private AbstractClassifier createDummyClassifier(String name) {

		String f = name.split("-")[1];
		int id = -1;

		if (f.equals("freq"))
			id = 16; // relative concept frequency
		if (f.equals("cfidf"))
			id = 18; // concept frequency * max term idf
		if (f.equals("pagerank"))
			id = 34; // page rank

		final int fId = id;
		return new AbstractClassifier() {
			@Override
			public void buildClassifier(Instances data) throws Exception {

			}

			@Override
			public double[][] distributionsForInstances(Instances data) {
				double[][] pred = new double[data.numInstances()][2];
				for (int i = 0; i < data.numInstances(); i++) {
					pred[i][1] = data.instance(i).value(fId);
					if (Double.isNaN(pred[i][1]))
						pred[i][1] = 0;
				}
				return pred;
			}
		};
	}

}

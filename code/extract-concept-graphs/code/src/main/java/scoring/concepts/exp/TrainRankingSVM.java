package scoring.concepts.exp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Map;

import com.google.common.io.Files;

import scoring.concepts.RankingSVM;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * Train and persist a ranking SVM classifier with filters
 * 
 * @author xxx
 *
 */
public class TrainRankingSVM {

	public static final String goldFolderName = "data/WikiCMaps/train";
	public static final String serFolderName = "data/WikiCMaps/train_system";
	public static final String name = "grouping-noun10conjarg2-sim5log-gopt";
	public static final String arffFileName = "data/selection/wiki_grouping-noun10conjarg2-sim5log-gopt.arff";
	public static final String modelName = "wiki_scoring_noun10conjarg2_sim5log-gopt_SVMRankC10_RepDisc.model";

	public static void main(String[] args) throws Exception {

		// 1) load arff
		ArffReader arff = new ArffReader(new BufferedReader(new FileReader(arffFileName)));
		Instances data = arff.getData();

		Map<String, Integer> dataMap = new HashMap<String, Integer>(70000);
		for (int i = 0; i < data.numInstances(); i++) {
			Instance ins = data.instance(i);
			String key = ((int) ins.value(1)) + "_" + ((int) ins.value(0));
			dataMap.put(key, i);
		}

		// remove likert label if present
		if (false) {
			Remove r = new Remove();
			r.setAttributeIndices("last");
			r.setInputFormat(data);
			data = Filter.useFilter(data, r);
		}
		data.setClassIndex(data.numAttributes() - 1);

		System.out.println("instances: " + data.numInstances());
		System.out.println("first att: " + data.attribute(0).name());
		System.out.println("last att: " + data.attribute(data.numAttributes() - 1).name());

		// setup filters
		Filter[] filters = new Filter[2];

		filters[0] = new ReplaceMissingValues();

		filters[1] = new Discretize();
		String[] opts2 = { "-D", "-Y", "-R", "3-last", "-precision", "6" };
		filters[1].setOptions(opts2);

		MultiFilter mFilter = new MultiFilter();
		mFilter.setFilters(filters);

		// 4) train model
		FilteredClassifier clf = new FilteredClassifier();
		RankingSVM svm = new RankingSVM();
		clf.setClassifier(svm);
		clf.setFilter(mFilter);

		System.out.println("training");
		clf.buildClassifier(data);
		System.out.println("done");

		// store
		Files.copy(new File("data/selection/tmp_ranksvm/training.pkl"),
				new File("models/" + modelName.replace(".model", "") + ".pkl"));
		svm.setModelPath("models/" + modelName.replace(".model", ""));

		SerializationHelper.write("models/" + modelName, clf);

		System.out.println("stored");
	}

}

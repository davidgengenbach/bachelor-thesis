package scoring.concepts;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.io.FileUtils;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class RankingSVM extends AbstractClassifier {

	private static final long serialVersionUID = 1L;

	private static final String pathLinux = "/media/sf_eclipseworkspace/cmapsum-props";
	private static final String tmpFolder = "data/selection/tmp_ranksvm";

	private String dlibModel;

	@Override
	public void buildClassifier(Instances data) throws Exception {

		// write data to file
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File(tmpFolder + "/training.arff"));
		saver.writeBatch();

		// start training
		String cmd = "\"C:\\Program Files\\Oracle\\VirtualBox\\VBoxManage.exe\" guestcontrol \"Ubuntu 14.04 64bit\" ";
		cmd += "--username windows --password \"N*[g\\>_Y]6{;,#mK\" ";
		cmd += "run " + pathLinux + "/src/main/scripts/scoring/run_ranking_svm.sh ";
		cmd += "train " + tmpFolder + "/training.arff";

		Process process = Runtime.getRuntime().exec(cmd);
		// BufferedReader stdErr = new BufferedReader(new
		// InputStreamReader(process.getErrorStream()));
		// String s;
		// while ((s = stdErr.readLine()) != null) {
		// System.out.println(s);
		// }
		//
		// BufferedReader stdOut = new BufferedReader(new
		// InputStreamReader(process.getInputStream()));
		// while ((s = stdOut.readLine()) != null) {
		// System.out.println(s);
		// }
		process.waitFor();

	}

	public void setModelPath(String path) {
		this.dlibModel = path;
	}

	@Override
	public double[][] distributionsForInstances(Instances batch) {

		// write data to file
		try {
			ArffSaver saver = new ArffSaver();
			saver.setInstances(batch);
			saver.setFile(new File(tmpFolder + "/batch.arff"));
			saver.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}

		// invoke rank svm
		try {
			String cmd = "\"C:\\Program Files\\Oracle\\VirtualBox\\VBoxManage.exe\" guestcontrol \"Ubuntu 14.04 64bit\" ";
			cmd += "--username windows --password \"N*[g\\>_Y]6{;,#mK\" ";
			cmd += "run " + pathLinux + "/src/main/scripts/scoring/run_ranking_svm.sh ";
			if (this.dlibModel != null)
				cmd += "rank " + tmpFolder + "/batch.arff " + dlibModel;
			else
				cmd += "rank " + tmpFolder + "/batch.arff " + tmpFolder + "/training";

			Process process = Runtime.getRuntime().exec(cmd);
			process.waitFor();

			// BufferedReader stdErr = new BufferedReader(new
			// InputStreamReader(process.getErrorStream()));
			// String s;
			// while ((s = stdErr.readLine()) != null) {
			// System.out.println(s);
			// }
			//
			// BufferedReader stdOut = new BufferedReader(new
			// InputStreamReader(process.getInputStream()));
			// while ((s = stdOut.readLine()) != null) {
			// System.out.println(s);
			// }
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		// process result
		Map<String, Double> results = new HashMap<String, Double>();
		try {
			for (String line : FileUtils.readLines(new File(tmpFolder + "/batch.ranked"))) {
				String[] cols = line.split("\t");
				results.put(cols[0] + "_" + cols[1], Double.parseDouble(cols[2]));
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		double[][] dists = new double[batch.numInstances()][2];
		for (int i = 0; i < batch.numInstances(); i++) {
			Instance ins = batch.instance(i);
			Double score = results.get(((int) ins.value(0)) + "_" + ((int) ins.value(1)));
			if (score == null)
				System.err.println("no score: " + ins);
			else {
				dists[i] = new double[2];
				dists[i][1] = score;
			}
		}

		return dists;
	}

}

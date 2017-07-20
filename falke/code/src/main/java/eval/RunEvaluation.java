package eval;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;

import eval.matcher.InclusiveMatch;
import eval.matcher.StemSWMatch;
import eval.metrics.ConceptMatchMetric;
import eval.metrics.ConceptYieldMetric;
import eval.metrics.Metric;
import eval.metrics.PropositionMatchMetric;
import eval.metrics.RelationMatchMetric;
import eval.metrics.RelationYieldMetric;
import model.ConceptMap;
import model.io.ConceptMapReader;
import model.io.Format;

/**
 * Run an evaluation that compares generated and reference concept maps on a
 * dataset with several topics
 */
public class RunEvaluation {

	public static String goldFolderName = "data/WikiCMaps/train";
	public static String mapFolderName = "data/WikiCMaps/train_system";
	public static String mapPattern = "full.*greedy.*cmap";

	public static void main(String[] args) throws IOException {

		// run evaluation per topic
		List<Evaluation> evaluations = runEval(goldFolderName, mapFolderName);

		// compute averages
		List<Evaluation> avgResults = Evaluation.getAvgResults(evaluations, false);
		for (Evaluation eval : avgResults) {
			System.out.println(eval.printResults());
		}

	}

	// run evaluations for all topics
	public static List<Evaluation> runEval(String goldFolderName, String mapFolderName) {

		// set up metrics
		InputStream sw = RunEvaluation.class.getResourceAsStream("/lists/stopwords_en_eval.txt");
		Metric conceptMatch = new ConceptMatchMetric(new StemSWMatch(sw));
		Metric conceptMatchInc = new ConceptMatchMetric(new InclusiveMatch());
		Metric conceptYield = new ConceptYieldMetric(null);
		sw = RunEvaluation.class.getResourceAsStream("/lists/stopwords_en_eval.txt");
		Metric propMatch = new PropositionMatchMetric(new StemSWMatch(sw), true);
		Metric propMatchInc = new PropositionMatchMetric(new InclusiveMatch(), true);
		sw = RunEvaluation.class.getResourceAsStream("/lists/stopwords_en_eval.txt");
		Metric relMatch = new RelationMatchMetric(new StemSWMatch(sw));
		Metric relMatchInc = new RelationMatchMetric(new InclusiveMatch());
		Metric propYield = new RelationYieldMetric(null);
		Metric[] metrics = { conceptMatch, conceptMatchInc, conceptYield };

		FilenameFilter filter = new FilenameFilter() {
			@Override
			public boolean accept(File dir, String name) {
				return name.endsWith(".cmap");
			}
		};

		System.out.println("Computing evaluation metrics");
		List<Evaluation> evaluations = new LinkedList<Evaluation>();

		// iterate over document cluster
		File goldFolder = new File(goldFolderName);
		for (File clusterFolder : goldFolder.listFiles()) {
			if (clusterFolder.isDirectory()) {

				// load maps
				File goldFile = new File(clusterFolder.listFiles(filter)[0].getPath());
				ConceptMap mapGold = ConceptMapReader.readFromFile(goldFile, Format.TSV);

				List<ConceptMap> maps = new LinkedList<ConceptMap>();
				File mFolder = new File(mapFolderName + "/" + clusterFolder.getName());
				for (File file : mFolder.listFiles(filter)) {
					if (file.getName().matches(mapPattern)) {
						ConceptMap map = ConceptMapReader.readFromFile(file, Format.TSV);
						maps.add(map);
					}
				}

				// evaluate
				Evaluation eval = new Evaluation(clusterFolder.getName(), mapGold);
				eval.addConceptMaps(maps);
				eval.addMetrics(metrics);
				eval.run();
				evaluations.add(eval);

				System.out.println(eval.printResults());
			}
		}
		return evaluations;
	}

	// save to file
	public static void saveToFile(String mapFolder, List<Evaluation> evaluations) {

		SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss");
		File resultFile = new File(mapFolder + "/eval-" + df.format(new Date()) + ".csv");
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(resultFile));
			for (Evaluation eval : evaluations) {
				writer.write(eval.getCsv());
			}
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}

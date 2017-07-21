package grouping;

import java.io.File;
import java.io.IOException;

import org.apache.uima.UIMAException;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.collection.CollectionReaderDescription;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.factory.CollectionReaderFactory;
import org.apache.uima.fit.pipeline.SimplePipeline;

import de.tudarmstadt.ukp.dkpro.core.io.bincas.BinaryCasReader;
import pipeline.ConceptMapMining;

/*
 * Concept recall after grouping
 */
public class PipelineGroupedConceptRecall {

	public static String dataFolder = "data/ng20/train";
	public static String mapFolder = "data/ng20/test_system";
	//public static final String mapName = "grouping-noun10conjarg2-sim5log-gopt";
	public static String mapName = "ng-20";

	public static String textPattern = "*.oie.bin6";

	public static String startTopic = "0";

	public static void main(String[] args) throws UIMAException, IOException {
		// iterate over topics
		if(args.length > 0) {
			dataFolder = args[0];
		}
		
		if(args.length > 1) {
			mapName = args[1];
		}
		
		if(args.length > 2) {
			mapFolder = args[2];
		}

		
		if(args.length > 3) {
			textPattern = args[3];
		}
		
		File folder = new File(dataFolder);
		for (File clusterFolder : folder.listFiles()) {
			if (clusterFolder.isDirectory()) {
				
				if (clusterFolder.getName().compareTo(startTopic) < 0)
					continue;

				System.out.println("------------------------------------------------------------");
				System.out.println(clusterFolder.getName());
				System.out.println("------------------------------------------------------------");

				// read preprocessed documents
				String docLocation = dataFolder + "/" + clusterFolder.getName();

				CollectionReaderDescription reader = CollectionReaderFactory.createReaderDescription(
						BinaryCasReader.class, BinaryCasReader.PARAM_SOURCE_LOCATION, docLocation,
						BinaryCasReader.PARAM_PATTERNS, textPattern, BinaryCasReader.PARAM_LANGUAGE, "en");

				// configure concept mapping pipeline
				String[] pipeline = { "extraction.PropositionExtractor", "grouping.ConceptGrouperSimLog",
						"grouping.ExtractionResultsSerializer"};
				String targetLocation = mapFolder + "/" + clusterFolder.getName();

				AnalysisEngineDescription cmm = AnalysisEngineFactory.createEngineDescription(ConceptMapMining.class,
						ConceptMapMining.PARAM_TARGET_LOCATION, targetLocation, ConceptMapMining.PARAM_COMPONENTS,
						pipeline, ConceptMapMining.PARAM_NAME, mapName);

				// run pipeline
				SimplePipeline.runPipeline(reader, cmm);
			}
		}

	}
}
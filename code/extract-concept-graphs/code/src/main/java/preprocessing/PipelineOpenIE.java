package preprocessing;

import java.io.IOException;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.uima.UIMAException;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.collection.CollectionReaderDescription;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.factory.CollectionReaderFactory;
import org.apache.uima.fit.pipeline.SimplePipeline;

import de.tudarmstadt.ukp.dkpro.core.io.bincas.BinaryCasReader;
import de.tudarmstadt.ukp.dkpro.core.io.bincas.BinaryCasWriter;

public class PipelineOpenIE {

	public static String textFolder = "data/CMapSummaries/test";
	public static String[] textPattern = { "/*/*.txt.bin6" };

	public static void main(String[] args) throws UIMAException, IOException {

		Logger.getRootLogger().setLevel(Level.INFO);

		if (args.length > 0)
			textFolder = args[0];
		if (args.length > 1)
			textPattern[0] = args[1];

		// read preprocessed documents
		CollectionReaderDescription reader = CollectionReaderFactory.createReaderDescription(BinaryCasReader.class,
				BinaryCasReader.PARAM_SOURCE_LOCATION, textFolder, BinaryCasReader.PARAM_PATTERNS, textPattern,
				BinaryCasReader.PARAM_LANGUAGE, "en");

		// find Open IE tuples
		AnalysisEngineDescription openIE = AnalysisEngineFactory.createEngineDescription(OpenIEAnnotator.class);

		// write annotated data to file
		AnalysisEngineDescription writer = AnalysisEngineFactory.createEngineDescription(BinaryCasWriter.class,
				BinaryCasWriter.PARAM_TARGET_LOCATION, textFolder, BinaryCasWriter.PARAM_STRIP_EXTENSION, true,
				BinaryCasWriter.PARAM_FILENAME_EXTENSION, ".oie.bin6", BinaryCasWriter.PARAM_OVERWRITE, true);

		// run pipeline
		SimplePipeline.runPipeline(reader, openIE, writer);
	}

}
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.uima.UIMAException;

import grouping.PipelineGroupedConceptRecall;
import model.ExtractionResult;
import preprocessing.PipelineOpenIE;
import preprocessing.PipelinePreprocessing;
import scoring.concepts.features.ExportGraphs;

public class Pipeline {
	public static String set = "train";
	public static String suffix = "";

	public static String folder = "data/ng20" + suffix + "/" + set;
	public static String mapName = "ng-20" + suffix;
	public static String graphFolder = "data/ng20" + suffix + "/" + set + "_graphs";

	public static void main(String[] args) throws UIMAException, IOException {
		print("Preprocessing");
		PipelinePreprocessing.main(new String[] { folder, "*/*.txt" });
		print("OpenIE");
		PipelineOpenIE.main(new String[] { folder, "/*/*.txt.bin6" });
		print("GroupedConceptRecall");
		PipelineGroupedConceptRecall.main(new String[] { folder, mapName, graphFolder, "*.oie.bin6" });
		print("ExportGraphs");
		ExportGraphs.main(new String[] { graphFolder, mapName });

		// Rename graph files so they have the right extension (*.gml)
		File folder = new File(graphFolder);
		for (File clusterFolder : folder.listFiles()) {
			if (clusterFolder.isDirectory()) {

				String graphFileName = graphFolder + "/" + clusterFolder.getName() + "/" + mapName + ".graph";
				String newFileName = graphFileName + ".gml";

				Files.copy(new File(graphFileName).toPath(), new File(newFileName).toPath(),
						StandardCopyOption.REPLACE_EXISTING);
			}
		}
		print("DONE!");
	}

	public static void print(String msg) {
		System.out.println("########################## " + msg + "\n\n\n");
	}
}

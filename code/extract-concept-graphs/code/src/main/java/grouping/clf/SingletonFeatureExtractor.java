package grouping.clf;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

public class SingletonFeatureExtractor {

	private static FeatureExtractor instance;

	public static FeatureExtractor getFeatureExtractor() {
		if (instance == null)
			SingletonFeatureExtractor.initializeFromFile("data/grouping/training.ser");
		return instance;
	}

	public static void initializeFromFile(String file) {
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(file));
			instance = (FeatureExtractor) in.readObject();
			in.close();
		} catch (IOException | ClassNotFoundException e) {
			e.printStackTrace();
		}
	}

	public static void setFeatureExtractor(FeatureExtractor ex) {
		if (instance == null)
			instance = ex;
	}

}

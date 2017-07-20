package util;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;

public class Stopwords implements Serializable {

	private static String STOPWORD_FILE = "lists/stopwords_en_semilar.txt";
	private Set<String> stopwords;

	public Stopwords(String fileName) {

		this.stopwords = new HashSet<String>();

		File f = new File(getClass().getClassLoader().getResource(fileName).getFile());
		try {
			LineIterator i = FileUtils.lineIterator(f);
			while (i.hasNext()) {
				this.stopwords.add(i.next());
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public Stopwords() {
		this(STOPWORD_FILE);
	}

	public boolean isSW(String word) {
		return this.stopwords.contains(word);
	}

}

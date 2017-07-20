package util;

import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.tcas.Annotation;

public class Dumper extends JCasAnnotator_ImplBase {

	public static final String PARAM_TYPE = "type";
	@ConfigurationParameter(name = PARAM_TYPE)
	private String type;

	@SuppressWarnings("unchecked")
	@Override
	public void process(JCas aJCas) throws AnalysisEngineProcessException {
		Class<? extends Annotation> typeClass = null;
		try {
			typeClass = (Class<? extends Annotation>) Class.forName(type);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		for (Annotation a : JCasUtil.select(aJCas, typeClass)) {
			System.out.println(a.getType().getShortName() + ": " + a.getCoveredText());
		}
	}

}

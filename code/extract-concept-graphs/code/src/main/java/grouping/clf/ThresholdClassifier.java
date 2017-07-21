package grouping.clf;

import java.text.DecimalFormat;
import java.text.NumberFormat;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.AttributeStats;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Binary classifier optimizing a threshold on a single numerical feature
 * 
 * predicts 1 if attribute is >= threshold
 * threshold tuned to best f1 on positive class
 * 
 * @author xxx
 *
 */
public class ThresholdClassifier implements Classifier {

	protected int att;
	protected String attName;
	protected double minValue;
	protected double maxValue;
	protected double threshold;

	@Override
	public void buildClassifier(Instances data) {

	}

	public void buildClassifier(Instances data, int att) throws Exception {
		this.att = att;
		this.attName = data.attribute(att).name();
		AttributeStats stats = data.attributeStats(this.att);
		this.minValue = stats.numericStats.min;
		this.maxValue = stats.numericStats.max;
		this.threshold = this.tuneThreshold(data, 0.01);
	}

	@Override
	public double[] distributionForInstance(Instance instance) {
		double[] probs = new double[2];

		double value = instance.value(att);
		double dist = Math.abs(this.threshold - value);

		if (value < this.threshold) {
			probs[1] = 0.5 - 0.5 * dist / (this.threshold - this.minValue);
		} else {
			probs[1] = 0.5001 + 0.5 * dist / (this.maxValue - this.threshold);
			probs[1] = Math.min(1, probs[1]);
		}
		probs[0] = 1 - probs[1];

		if (Double.isNaN(probs[1]))
			System.err.println("prediction is NaN");

		return probs;
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		double[] probs = this.distributionForInstance(instance);
		if (probs[1] >= 0.5)
			return 1;
		else
			return 0;
	}

	public double getThreshold() {
		return this.threshold;
	}

	@Override
	public String toString() {
		NumberFormat f = new DecimalFormat("#0.00");
		return this.attName + " (" + f.format(this.threshold) + ")";
	}

	private double tuneThreshold(Instances data, double step) throws Exception {

		ThresholdClassifier clf = new ThresholdClassifier();
		clf.att = this.att;
		clf.minValue = this.minValue;
		clf.maxValue = this.maxValue;

		double bestThreshold = 0;
		double bestMetric = -1;
		for (double t = this.minValue; t <= this.maxValue; t += step) {
			clf.threshold = t;
			Evaluation eval = new Evaluation(data);
			eval.evaluateModel(clf, data);
			if (eval.fMeasure(1) > bestMetric) {
				bestMetric = eval.fMeasure(1);
				bestThreshold = t;
			}
		}
		return bestThreshold;
	}

	@Override
	public Capabilities getCapabilities() {
		return null;
	}

}

package grouping;

import java.util.Collection;

import model.Concept;

public class LabelSelector {

	private Strategy[] stratsA;
	private Strategy[] stratsB;
	private int splitPoint;
	private double splitFrac;

	public LabelSelector(Strategy... strategy) {
		this.stratsA = strategy;
	}

	public LabelSelector(Strategy stratA, int splitSize, Strategy... stratB) {
		this.stratsA = new Strategy[1];
		this.stratsA[0] = stratA;
		this.stratsB = stratB;
		this.splitPoint = splitSize;
	}

	public LabelSelector(Strategy stratA, double splitFrac, Strategy... stratB) {
		this.stratsA = new Strategy[1];
		this.stratsA[0] = stratA;
		this.stratsB = stratB;
		this.splitFrac = splitFrac;
	}

	public Concept select(Collection<Concept> concepts) {
		if (concepts.size() == 1)
			return concepts.iterator().next();
		if (this.stratsB == null) {
			return this.select(concepts, this.stratsA);
		} else {
			if (splitFrac != 0) {
				Strategy[] stratFreq = new Strategy[1];
				stratFreq[0] = Strategy.FREQ;
				Concept freq = this.select(concepts, stratFreq);
				if ((freq.weight / concepts.size()) <= splitFrac)
					return this.select(concepts, this.stratsA);
				else
					return this.select(concepts, this.stratsB);
			} else {
				if (concepts.size() <= splitPoint)
					return this.select(concepts, this.stratsA);
				else
					return this.select(concepts, this.stratsB);
			}
		}
	}

	private Concept select(Collection<Concept> concepts, Strategy[] strats) {
		Concept best = concepts.iterator().next();
		for (Concept c : concepts) {
			if (this.isBetter(c, best, strats))
				best = c;
		}
		return best;
	}

	private boolean isBetter(Concept c, Concept best, Strategy[] strats) {
		int res = 0;
		for (Strategy s : strats) {
			res = this.compare(c, best, s);
			if (res > 0)
				return true;
			else if (res < 0)
				return false;
		}
		return false;
	}

	private int compare(Concept a, Concept b, Strategy strat) {
		int res = 0;
		switch (strat) {
		case FREQ:
			return Double.compare(a.weight, b.weight);
		case LONGEST:
			return Integer.compare(a.name.length(), b.name.length());
		case SHORTEST:
			return Integer.compare(b.name.length(), a.name.length());
		}
		return res;
	}

	public enum Strategy {
		SHORTEST, LONGEST, FREQ
	}

}

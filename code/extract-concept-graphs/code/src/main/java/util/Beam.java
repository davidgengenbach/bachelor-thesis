package util;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Fixed length priority queue
 * 
 * Methods:
 * - add: add element. if exceeds max size, least element will be removed
 * - iterate
 * 
 * @author xxx
 *
 * @param <E>
 */
public class Beam<E extends Comparable<E>> {

	private int beamSize;
	private PriorityQueue<E> queue;

	public Beam(int beamSize) {
		this.queue = new PriorityQueue<E>(11, Collections.reverseOrder());
		this.beamSize = beamSize;
	}

	public Beam(int beamSize, Collection<E> items) {
		this(beamSize);
		this.addAll(items);
	}

	public void add(E item) {
		this.queue.add(item);
		if (this.queue.size() > this.beamSize)
			this.queue.poll();
	}

	public void addAll(Collection<E> items) {
		for (E item : items)
			this.add(item);
	}

	public Collection<E> pollAll() {
		List<E> items = new ArrayList<E>();
		while (!this.queue.isEmpty())
			items.add(this.queue.poll());
		return items;
	}

	public boolean isEmpty() {
		return this.queue.isEmpty();
	}

	public int size() {
		return this.queue.size();
	}

}

package util;

import java.io.Serializable;

public class KeyValuePair<K, V> implements Serializable {

	private static final long serialVersionUID = 1L;

	public K key;
	public V value;

	public KeyValuePair(K key, V value) {
		this.key = key;
		this.value = value;
	}

}

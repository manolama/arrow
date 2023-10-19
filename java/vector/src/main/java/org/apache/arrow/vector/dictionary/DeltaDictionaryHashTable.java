package org.apache.arrow.vector.dictionary;

import org.apache.arrow.memory.util.hash.ArrowBufHasher;
import org.apache.arrow.memory.util.hash.SimpleHasher;

public class DeltaDictionaryHashTable {

  /**
   * Represents a null value in map.
   */
  static final int NULL_VALUE = -1;

  /**
   * The default initial capacity - MUST be a power of two.
   */
  static final int DEFAULT_INITIAL_CAPACITY = 1 << 4;

  /**
   * The maximum capacity, used if a higher value is implicitly specified
   * by either of the constructors with arguments.
   */
  static final int MAXIMUM_CAPACITY = 1 << 30;

  /**
   * The load factor used when none specified in constructor.
   */
  static final float DEFAULT_LOAD_FACTOR = 0.75f;

  static final DictionaryHashTable.Entry[] EMPTY_TABLE = {};

  /**
   * The table, initialized on first use, and resized as
   * necessary. When allocated, length is always a power of two.
   */
  transient DictionaryHashTable.Entry[] table = EMPTY_TABLE;

  /**
   * The number of key-value mappings contained in this map.
   */
  transient int size;

  /**
   * The next size value at which to resize (capacity * load factor).
   */
  int threshold;

  /**
   * The load factor for the hash table.
   */
  final float loadFactor;

  private final ArrowBufHasher hasher;

  public DeltaDictionaryHashTable(int initialCapacity, ArrowBufHasher hasher) {
    if (initialCapacity < 0) {
      throw new IllegalArgumentException("Illegal initial capacity: " +
          initialCapacity);
    }
    if (initialCapacity > MAXIMUM_CAPACITY) {
      initialCapacity = MAXIMUM_CAPACITY;
    }
    this.loadFactor = DEFAULT_LOAD_FACTOR;
    this.threshold = initialCapacity;
    this.hasher = hasher;
  }

  public DeltaDictionaryHashTable(ArrowBufHasher hasher) {
    this(DEFAULT_INITIAL_CAPACITY, hasher);
  }

  public DeltaDictionaryHashTable() {
    this(SimpleHasher.INSTANCE);
  }
}

package org.apache.arrow.vector.dictionary;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.util.hash.MurmurHasher;
import org.apache.arrow.vector.BaseIntVector;
import org.apache.arrow.vector.BaseVariableWidthVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.DictionaryEncoding;
import org.apache.arrow.vector.types.pojo.FieldType;

import java.io.Closeable;
import java.io.IOException;

public class DeltaDictionary implements Closeable, BaseDictionary {

  private final DictionaryEncoding encoding;

  private final BufferAllocator allocator;

  private final BaseVariableWidthVector dictionary;

  private final BaseIntVector indexVector;

  private final DictionaryHashTable hashTable;

  private int deltaIndex;
  private int dictionaryIndex;
  private int indexIndex;

  public DeltaDictionary(
      String name,
      DictionaryEncoding encoding,
      ArrowType dictionaryType,
      ArrowType indexType,
      BufferAllocator allocator
  ) {
    this.encoding = encoding;
    this.allocator = allocator;
    dictionary = (BaseVariableWidthVector) new FieldType(false, dictionaryType, null).createNewSingleVector(name + "-dictionary", allocator, null);
    indexVector = (BaseIntVector) new FieldType(false, indexType, encoding).createNewSingleVector(name, allocator, null);
    hashTable = new DictionaryHashTable();
  }

  public FieldVector getIndexVector() {
    return indexVector;
  }

  public FieldVector getVector() {
    return dictionary;
  }

  public ArrowType getVectorType() {
    return dictionary.getField().getType();
  }

  public DictionaryEncoding getEncoding() {
    return encoding;
  }

  public void add(byte[] value) {
    add(value, 0, value.length);
  }

  public void add(byte[] value, int offset, int len) {
    int index = getIndex(value, offset, len);
    System.out.println("   Wrote " + new String(value, offset, len) + " with dict [" + index + "] to " + indexIndex);
    indexVector.setWithPossibleTruncate(indexIndex++, index);
  }

  private int getIndex(byte[] value, int offset, int len) {
    int hash = MurmurHasher.hashCode(value, offset, len, 0);
    int i = hashTable.getIndex(hash);
    System.out.println("   Got idx " + i + " for " + new String(value));
    if (i >= 0) {
      return i;
    } else {
      hashTable.addEntry(hash, deltaIndex);
      dictionary.setSafe(dictionaryIndex++, value, offset, len);
      return deltaIndex++;
    }
  }

  @Override
  public void close() throws IOException {
    dictionary.close();
    indexVector.close();
  }

  public void reset() {
    dictionaryIndex = 0;
    indexIndex = 0;
    dictionary.reset();
    indexVector.reset();
  }

  public void mark() {
    dictionary.setValueCount(dictionaryIndex);
    indexVector.setValueCount(indexIndex);
    System.out.println("   [[[ Marked Dict: " + dictionaryIndex + "  i: " + indexIndex + " => " + dictionary + "  => " + indexVector);
  }
}

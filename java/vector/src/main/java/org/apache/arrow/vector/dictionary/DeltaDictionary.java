/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

/**
 * A dictionary implementation used for delta encoding across batches. This dictionary
 * is limited to a single column.
 */
public class DeltaDictionary implements Closeable, BaseDictionary {

  private final DictionaryEncoding encoding;

  private final BaseVariableWidthVector dictionary;

  private final BaseIntVector indexVector;

  private final DictionaryHashTable hashTable;

  private int deltaIndex;
  private int dictionaryIndex;

  public DeltaDictionary(
      String name,
      DictionaryEncoding encoding,
      ArrowType dictionaryType,
      ArrowType indexType,
      BufferAllocator allocator
  ) {
    this.encoding = encoding;
    dictionary = (BaseVariableWidthVector) new FieldType(false, dictionaryType, null).createNewSingleVector(name + "-dictionary", allocator, null);
    indexVector = (BaseIntVector) new FieldType(true, indexType, encoding).createNewSingleVector(name, allocator, null);
    hashTable = new DictionaryHashTable();
  }

  public DeltaDictionary(
      FieldVector dictionary,
      FieldVector indexVector
  ) {
    this.encoding = dictionary.getField().getDictionary();
    this.dictionary = (BaseVariableWidthVector) dictionary;
    this.indexVector = (BaseIntVector) indexVector;
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

  public void set(int index, byte[] value) {
    set(index, value, 0, value.length);
  }

  public void set(int index, byte[] value, int offset, int len) {
    int di = getIndex(value, offset, len);
    indexVector.setWithPossibleTruncate(index, di);
  }

  private int getIndex(byte[] value, int offset, int len) {
    int hash = MurmurHasher.hashCode(value, offset, len, 0);
    int i = hashTable.getIndex(hash);
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
    dictionary.reset();
    indexVector.reset();
  }

  public void mark() {
    dictionary.setValueCount(dictionaryIndex);
  }
}

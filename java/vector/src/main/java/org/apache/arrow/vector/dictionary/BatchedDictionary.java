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
 * A dictionary implementation that can be used when writing batches of data to
 * a stream or file. Supports delta or replacement encoding.
 *
 *
 *
 */
public class BatchedDictionary implements Closeable, BaseDictionary {

  private final DictionaryEncoding encoding;

  private final BaseVariableWidthVector dictionary;

  private final BaseIntVector indexVector;

  private final DictionaryHashTable hashTable;

  private final boolean forFileIPC;

  private int deltaIndex;

  private int dictionaryIndex;

  private boolean wasReset;

  public BatchedDictionary(
      String name,
      DictionaryEncoding encoding,
      ArrowType dictionaryType,
      ArrowType indexType,
      boolean forFileIPC,
      BufferAllocator allocator
  ) {
    this(name, encoding, dictionaryType, indexType, forFileIPC, allocator, "-dictionary");
  }

  public BatchedDictionary(
      String name,
      DictionaryEncoding encoding,
      ArrowType dictionaryType,
      ArrowType indexType,
      boolean forFileIPC,
      BufferAllocator allocator,
      String suffix
  ) {
    this.encoding = encoding;
    this.forFileIPC = forFileIPC;
    FieldVector vector = new FieldType(false, dictionaryType, null)
        .createNewSingleVector(name + suffix, allocator, null);
    if (!(BaseVariableWidthVector.class.isAssignableFrom(vector.getClass()))) {
      throw new IllegalArgumentException("Dictionary must be a superclass of 'BaseVariableWidthVector' " +
          "such as 'VarCharVector'.");
    }
    dictionary = (BaseVariableWidthVector) vector;
    vector = new FieldType(true, indexType, encoding)
        .createNewSingleVector(name, allocator, null);
    if (!(BaseIntVector.class.isAssignableFrom(vector.getClass()))) {
      throw new IllegalArgumentException("Index vector must be a superclass type of 'BaseIntVector' " +
          "such as 'IntVector' or 'Uint4Vector'.");
    }
    indexVector = (BaseIntVector) vector;
    hashTable = new DictionaryHashTable();
  }

  public BatchedDictionary(
      FieldVector dictionary,
      FieldVector indexVector,
      boolean forFileIPC
  ) {
    this.encoding = dictionary.getField().getDictionary();
    this.forFileIPC = forFileIPC;
    if (!(BaseVariableWidthVector.class.isAssignableFrom(dictionary.getClass()))) {
      throw new IllegalArgumentException("Dictionary must be a superclass of 'BaseVariableWidthVector' " +
          "such as 'VarCharVector'.");
    }
    if (dictionary.getField().isNullable()) {
      throw new IllegalArgumentException("Dictionary must be non-nullable.");
    }
    this.dictionary = (BaseVariableWidthVector) dictionary;
    if (!(BaseIntVector.class.isAssignableFrom(indexVector.getClass()))) {
      throw new IllegalArgumentException("Index vector must be a superclass type of 'BaseIntVector' " +
          "such as 'IntVector' or 'Uint4Vector'.");
    }
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
    if (value == null) {
      setNull(index);
      return;
    }
    set(index, value, 0, value.length);
  }

  public void set(int index, byte[] value, int offset, int len) {
    if (value == null || len == 0) {
      setNull(index);
      return;
    }
    int di = getIndex(value, offset, len);
    indexVector.setWithPossibleTruncate(index, di);
  }

  public void setNull(int index) {
    indexVector.setNull(index);
  }

  private int getIndex(byte[] value, int offset, int len) {
    int hash = MurmurHasher.hashCode(value, offset, len, 0);
    int i = hashTable.getIndex(hash);
    if (i >= 0) {
      return i;
    } else {
      if (wasReset && forFileIPC && !encoding.isDelta()) {
        throw new IllegalStateException("Dictionary was reset and is not in delta mode. " +
            "This is not supported for file IPC.");
      }
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

  public void mark() {
    dictionary.setValueCount(dictionaryIndex);
    // not setting the index vector value count. The root can do that.
  }

  public void reset() {
    wasReset = true;
    dictionaryIndex = 0;
    dictionary.reset();
    indexVector.reset();
    if (!forFileIPC && !encoding.isDelta()) {
      // replacement mode.
      deltaIndex = 0;
      hashTable.clear();
    }
  }

  DictionaryHashTable getHashTable() {
    return hashTable;
  }
}

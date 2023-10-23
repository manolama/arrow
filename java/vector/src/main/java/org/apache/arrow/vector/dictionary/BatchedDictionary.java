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
import org.apache.arrow.util.VisibleForTesting;
import org.apache.arrow.vector.BaseIntVector;
import org.apache.arrow.vector.BaseVariableWidthVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.ipc.ArrowWriter;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.DictionaryEncoding;
import org.apache.arrow.vector.types.pojo.FieldType;

import java.io.Closeable;
import java.io.IOException;

/**
 * A dictionary implementation that can be used when writing batches of data to
 * a stream or file. Supports delta or replacement encoding.
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

  /**
   * Creates a dictionary with two vectors of the given types. The dictionary vector
   * will be named "{name}-dictionary".
   * <p>
   * To use this dictionary, provide the dictionary vector to a {@link DictionaryProvider},
   * add the {@link #getIndexVector()} to the {@link org.apache.arrow.vector.VectorSchemaRoot}
   * and call the {@link #setSafe(int, byte[], int, int)} or other set methods.
   *
   * @param name A name for the vector and dictionary.
   * @param encoding The dictionary encoding to use.
   * @param dictionaryType The type of the dictionary data.
   * @param indexType The type of the encoded dictionary index.
   * @param forFileIPC Whether the data will be written to a file or stream IPC. Throws an
   *                   exception if a replacement dictionary is provided to a file IPC.
   * @param allocator The allocator to use.
   */
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

  /**
   * Creates a dictionary with two vectors of the given types.
   *
   * @param name A name for the vector and dictionary.
   * @param encoding The dictionary encoding to use.
   * @param dictionaryType The type of the dictionary data.
   * @param indexType The type of the encoded dictionary index.
   * @param forFileIPC Whether the data will be written to a file or stream IPC. Throws an
   *                   exception if a replacement dictionary is provided to a file IPC.
   * @param allocator The allocator to use.
   * @param suffix A non-null suffix to append to the name of the dictionary.
   */
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

  /**
   * Creates a dictionary that will populate the provided vectors with data. Useful if
   * dictionaries need to be children of a parent vector.
   * @param dictionary The dictionary to hold the original data.
   * @param indexVector The index to store the encoded offsets.
   * @param forFileIPC Whether the data will be written to a file or stream IPC. Throws an
   *                   exception if a replacement dictionary is provided to a file IPC.
   */
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

  /**
   * @return The index vector.
   */
  public FieldVector getIndexVector() {
    return indexVector;
  }

  @Override
  public FieldVector getVector() {
    return dictionary;
  }

  @Override
  public ArrowType getVectorType() {
    return dictionary.getField().getType();
  }

  @Override
  public DictionaryEncoding getEncoding() {
    return encoding;
  }

  /**
   * Considers the entire byte array as the dictionary value. If the value is null,
   * a null will be written to the index.
   *
   * @param index the value to change
   * @param value the value to write.
   */
  public void setSafe(int index, byte[] value) {
    if (value == null) {
      setNull(index);
      return;
    }
    setSafe(index, value, 0, value.length);
  }

  /**
   * Encodes the given range in the dictionary. If the value is null, a null will be
   * written to the index.
   *
   * @param index the value to change
   * @param value the value to write.
   * @param offset An offset into the value array.
   * @param len The length of the value to write.
   */
  public void setSafe(int index, byte[] value, int offset, int len) {
    if (value == null || len == 0) {
      setNull(index);
      return;
    }
    int di = getIndex(value, offset, len);
    indexVector.setWithPossibleTruncate(index, di);
  }

  /**
   * Set the element at the given index to null.
   *
   * @param index the value to change
   */
  public void setNull(int index) {
    indexVector.setNull(index);
  }

  @Override
  public void close() throws IOException {
    dictionary.close();
    indexVector.close();
  }

  /**
   * Mark the dictionary as complete for the batch. Called by the {@link ArrowWriter}
   * on {@link ArrowWriter#writeBatch()}.
   */
  public void mark() {
    dictionary.setValueCount(dictionaryIndex);
    // not setting the index vector value count. That will happen when the user calls
    // VectorSchemaRoot#setRowCount().
  }

  /**
   * Resets the dictionary to be used for a new batch. Called by the {@link ArrowWriter} on
   * {@link ArrowWriter#writeBatch()}.
   */
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

  @VisibleForTesting
  DictionaryHashTable getHashTable() {
    return hashTable;
  }
}

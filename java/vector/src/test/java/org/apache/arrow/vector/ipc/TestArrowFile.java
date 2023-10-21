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

package org.apache.arrow.vector.ipc;

import static java.nio.channels.Channels.newChannel;
import static org.apache.arrow.vector.TestUtils.newVarCharVector;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.util.Collections2;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.ValueVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.dictionary.Dictionary;
import org.apache.arrow.vector.dictionary.DictionaryEncoder;
import org.apache.arrow.vector.types.pojo.DictionaryEncoding;
import org.apache.arrow.vector.types.pojo.Field;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestArrowFile extends BaseFileTest {
  private static final Logger LOGGER = LoggerFactory.getLogger(TestArrowFile.class);

  // overriding here since a number of other UTs sharing the BaseFileTest class use
  // legacy JUnit.
  @BeforeEach
  public void init() {
    allocator = new RootAllocator(Integer.MAX_VALUE);
  }

  @AfterEach
  public void tearDown() {
    allocator.close();
  }

  @Test
  public void testWrite() throws IOException {
    File file = new File("target/mytest_write.arrow");
    int count = COUNT;
    try (
        BufferAllocator vectorAllocator = allocator.newChildAllocator("original vectors", 0, Integer.MAX_VALUE);
        StructVector parent = StructVector.empty("parent", vectorAllocator)) {
      writeData(count, parent);
      write(parent.getChild("root"), file, new ByteArrayOutputStream());
    }
  }

  @Test
  public void testWriteComplex() throws IOException {
    File file = new File("target/mytest_write_complex.arrow");
    int count = COUNT;
    try (
        BufferAllocator vectorAllocator = allocator.newChildAllocator("original vectors", 0, Integer.MAX_VALUE);
        StructVector parent = StructVector.empty("parent", vectorAllocator)) {
      writeComplexData(count, parent);
      FieldVector root = parent.getChild("root");
      validateComplexContent(count, new VectorSchemaRoot(root));
      write(root, file, new ByteArrayOutputStream());
    }
  }

  /**
   * Writes the contents of parents to file. If outStream is non-null, also writes it
   * to outStream in the streaming serialized format.
   */
  private void write(FieldVector parent, File file, OutputStream outStream) throws IOException {
    VectorSchemaRoot root = new VectorSchemaRoot(parent);

    try (FileOutputStream fileOutputStream = new FileOutputStream(file);
         ArrowFileWriter arrowWriter = new ArrowFileWriter(root, null, fileOutputStream.getChannel());) {
      LOGGER.debug("writing schema: " + root.getSchema());
      arrowWriter.start();
      arrowWriter.writeBatch();
      arrowWriter.end();
    }

    // Also try serializing to the stream writer.
    if (outStream != null) {
      try (ArrowStreamWriter arrowWriter = new ArrowStreamWriter(root, null, outStream)) {
        arrowWriter.start();
        arrowWriter.writeBatch();
        arrowWriter.end();
      }
    }
  }

  @Test
  public void testFileStreamHasEos() throws IOException {

    try (VarCharVector vector1 = newVarCharVector("varchar1", allocator)) {
      vector1.allocateNewSafe();
      vector1.set(0, "foo".getBytes(StandardCharsets.UTF_8));
      vector1.set(1, "bar".getBytes(StandardCharsets.UTF_8));
      vector1.set(3, "baz".getBytes(StandardCharsets.UTF_8));
      vector1.set(4, "bar".getBytes(StandardCharsets.UTF_8));
      vector1.set(5, "baz".getBytes(StandardCharsets.UTF_8));
      vector1.setValueCount(6);

      List<Field> fields = Arrays.asList(vector1.getField());
      List<FieldVector> vectors = Collections2.asImmutableList(vector1);
      VectorSchemaRoot root = new VectorSchemaRoot(fields, vectors, vector1.getValueCount());

      // write data
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      ArrowFileWriter writer = new ArrowFileWriter(root, null, newChannel(out));
      writer.start();
      writer.writeBatch();
      writer.end();

      byte[] bytes = out.toByteArray();
      byte[] bytesWithoutMagic = new byte[bytes.length - 8];
      System.arraycopy(bytes, 8, bytesWithoutMagic, 0, bytesWithoutMagic.length);

      try (ArrowStreamReader reader = new ArrowStreamReader(new ByteArrayInputStream(bytesWithoutMagic), allocator)) {
        assertTrue(reader.loadNextBatch());
        // here will throw exception if read footer instead of eos.
        assertFalse(reader.loadNextBatch());
      }
    }
  }

  @ParameterizedTest
  @MethodSource("dictionaryParams")
  public void testMultiBatchDictionaries(int state) throws Exception {
    File file = new File("target/mytest_multi_batch_dictionaries_" + state + ".arrow");
    writeDataMultiBatchWithDictionaries(file, state);

    try (FileInputStream fileInputStream = new FileInputStream(file);
         ArrowFileReader reader = new ArrowFileReader(fileInputStream.getChannel(), allocator);) {
      for (int i = 0; i < 4; i++) {
        reader.loadNextBatch();
        assertBlock(file, reader, i, state);
      }
    }
  }

  @ParameterizedTest
  @MethodSource("dictionaryParams")
  public void testMultiBatchDictionariesOutOfOrder(int state) throws Exception {
    File file = new File("target/mytest_multi_batch_dictionaries_ooo_" + state + ".arrow");
    writeDataMultiBatchWithDictionaries(file, state);

    try (FileInputStream fileInputStream = new FileInputStream(file);
         ArrowFileReader reader = new ArrowFileReader(fileInputStream.getChannel(), allocator);) {
      int[] order = new int[] {2, 1, 3, 0};
      for (int i = 0; i < 4; i++) {
        int block = order[i];
        reader.loadRecordBatch(reader.getRecordBlocks().get(block));
        assertBlock(file, reader, i, state);
      }
    }
  }

  @ParameterizedTest
  @MethodSource("dictionaryParams")
  public void testMultiBatchDictionariesSeek(int state) throws Exception {
    File file = new File("target/mytest_multi_batch_dictionaries_seek_" + state + ".arrow");
    writeDataMultiBatchWithDictionaries(file, state);
    for (int i = 0; i < 4; i++) {
      assertBlock(file, i, state);
    }
  }

  private void assertDictionary(FieldVector encoded, ArrowFileReader reader, String... expected) throws Exception {
    DictionaryEncoding dictionaryEncoding = encoded.getField().getDictionary();
    Dictionary dictionary = reader.getDictionaryVectors().get(dictionaryEncoding.getId());
    try (ValueVector decoded = DictionaryEncoder.decode(encoded, dictionary)) {
      Assertions.assertEquals(expected.length, encoded.getValueCount());
      for (int i = 0; i < expected.length; i++) {
        if (expected[i] == null) {
          Assertions.assertNull(decoded.getObject(i));
        } else {
          assertNotNull(decoded.getObject(i));
          Assertions.assertEquals(expected[i], decoded.getObject(i).toString());
        }
      }
    }
  }

  private void assertBlock(File file, int block, int state) throws Exception {
    try (FileInputStream fileInputStream = new FileInputStream(file);
         ArrowFileReader reader = new ArrowFileReader(fileInputStream.getChannel(), allocator);) {
      assertBlock(file, reader, block, state);
    }
  }

  private void assertBlock(File file, ArrowFileReader reader, int block, int state) throws Exception {
    VectorSchemaRoot r = reader.getVectorSchemaRoot();
    FieldVector dictA = r.getVector("vectorA");
    FieldVector dictB = r.getVector("vectorB");
    FieldVector dictC = r.getVector("vectorC");

    reader.loadRecordBatch(reader.getRecordBlocks().get(block));
    switch (state) {
      case 1:
        assertDictionary(dictA, reader, valuesPerBlock.get(block)[0]);
        assertNull(dictB);
        assertNull(dictC);
        break;
      case 2:
        assertNull(dictA);
        assertDictionary(dictB, reader, valuesPerBlock.get(block)[1]);
        assertNull(dictC);
        break;
      case 3:
        assertDictionary(dictA, reader, valuesPerBlock.get(block)[0]);
        assertDictionary(dictB, reader, valuesPerBlock.get(block)[1]);
        assertNull(dictC);
        break;
      case 4:
        assertNull(dictA);
        assertNull(dictB);
        assertDictionary(dictC, reader, valuesPerBlock.get(block)[2]);
        break;
      case 5:
        assertDictionary(dictA, reader, valuesPerBlock.get(block)[0]);
        assertNull(dictB);
        assertDictionary(dictC, reader, valuesPerBlock.get(block)[2]);
        break;
      case 6:
        assertDictionary(dictA, reader, valuesPerBlock.get(block)[0]);
        assertDictionary(dictB, reader, valuesPerBlock.get(block)[1]);
        assertDictionary(dictC, reader, valuesPerBlock.get(block)[2]);
        break;
    }
  }

  public static Collection<Arguments> dictionaryParams() {
    List<Arguments> params = new ArrayList<>();
    for (int i = 1; i < 7; i++) {
      params.add(Arguments.of(i));
    }
    return params;
  }
}

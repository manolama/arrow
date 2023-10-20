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
import java.util.Arrays;
import java.util.List;

import org.apache.arrow.memory.BufferAllocator;
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
import org.junit.Test;
import org.junit.jupiter.api.Assertions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestArrowFile extends BaseFileTest {
  private static final Logger LOGGER = LoggerFactory.getLogger(TestArrowFile.class);

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

  @Test
  public void testMultiBatchDeltaDictionary() throws Exception {
    File file = new File("target/mytest_multi_delta_dictionary.arrow");
    writeDataMultiBatchWithDictionaries(file, 1);

    try (FileInputStream fileInputStream = new FileInputStream(file);
         ArrowFileReader reader = new ArrowFileReader(fileInputStream.getChannel(), allocator);) {
      VectorSchemaRoot r = reader.getVectorSchemaRoot();
      FieldVector dictionary = r.getVector("vectorA");

      reader.loadNextBatch();
      assertDictionary(dictionary, reader, "foo", "bar");

      reader.loadNextBatch();
      assertDictionary(dictionary, reader, "meep", "bar");

      reader.loadNextBatch();
      assertDictionary(dictionary, reader, null, "bazz");

      reader.loadNextBatch();
      assertDictionary(dictionary, reader, "bar", "zap");
    }
  }

  @Test
  public void testMultiBatchDeltaDictionaryOutOfOrder() throws Exception {
    File file = new File("target/mytest_multi_delta_dictionary_ooo.arrow");
    writeDataMultiBatchWithDictionaries(file, 1);

    try (FileInputStream fileInputStream = new FileInputStream(file);
         ArrowFileReader reader = new ArrowFileReader(fileInputStream.getChannel(), allocator);) {
      VectorSchemaRoot r = reader.getVectorSchemaRoot();
      FieldVector dictionary = r.getVector("vectorA");

      reader.loadRecordBatch(reader.getRecordBlocks().get(2));
      assertDictionary(dictionary, reader, null, "bazz");

      reader.loadRecordBatch(reader.getRecordBlocks().get(1));
      assertDictionary(dictionary, reader, "meep", "bar");

      reader.loadRecordBatch(reader.getRecordBlocks().get(0));
      assertDictionary(dictionary, reader, "foo", "bar");

      reader.loadRecordBatch(reader.getRecordBlocks().get(3));
      assertDictionary(dictionary, reader, "bar", "zap");
    }
  }

  @Test
  public void testMultiBatchDeltaDictionarySeek() throws Exception {
    File file = new File("target/mytest_multi_delta_dictionary_seek.arrow");
    writeDataMultiBatchWithDictionaries(file, 1);

    assertBlock(file, 0, new String[]{"foo", "bar"}, null);
    assertBlock(file, 1, new String[]{"meep", "bar"}, null);
    assertBlock(file, 2, new String[]{null, "bazz"}, null);
    assertBlock(file, 3, new String[]{"bar", "zap"}, null);
  }

  @Test
  public void testMultiBatchReplacementDictionary() throws Exception {
    File file = new File("target/mytest_multi_delta_replacement.arrow");
    writeDataMultiBatchWithDictionaries(file, 2);

    try (FileInputStream fileInputStream = new FileInputStream(file);
         ArrowFileReader reader = new ArrowFileReader(fileInputStream.getChannel(), allocator);) {
      VectorSchemaRoot r = reader.getVectorSchemaRoot();
      FieldVector dictionary = r.getVector("vectorB");

      reader.loadNextBatch();
      assertDictionary(dictionary, reader, "lorem", "ipsum");

      reader.loadNextBatch();
      assertDictionary(dictionary, reader, "ipsum", "lorem");

      reader.loadNextBatch();
      assertDictionary(dictionary, reader, "ipsum", null);

      reader.loadNextBatch();
      assertDictionary(dictionary, reader, null, "lorem");
    }
  }

  @Test
  public void testMultiBatchReplacementDictionaryOutOfOrder() throws Exception {
    File file = new File("target/mytest_multi_delta_replacement_ooo.arrow");
    writeDataMultiBatchWithDictionaries(file, 2);

    try (FileInputStream fileInputStream = new FileInputStream(file);
         ArrowFileReader reader = new ArrowFileReader(fileInputStream.getChannel(), allocator);) {
      VectorSchemaRoot r = reader.getVectorSchemaRoot();
      FieldVector dictionary = r.getVector("vectorB");

      reader.loadRecordBatch(reader.getRecordBlocks().get(2));
      assertDictionary(dictionary, reader, "ipsum", null);

      reader.loadRecordBatch(reader.getRecordBlocks().get(1));
      assertDictionary(dictionary, reader, "ipsum", "lorem");

      reader.loadRecordBatch(reader.getRecordBlocks().get(3));
      assertDictionary(dictionary, reader, null, "lorem");

      reader.loadRecordBatch(reader.getRecordBlocks().get(0));
      assertDictionary(dictionary, reader, "lorem", "ipsum");
    }
  }

  @Test
  public void testMultiBatchReplacementDictionarySeek() throws Exception {
    File file = new File("target/mytest_multi_delta_replacement_seek.arrow");
    writeDataMultiBatchWithDictionaries(file, 2);

    assertBlock(file, 0, null, new String[]{"lorem", "ipsum"});
    assertBlock(file, 1, null, new String[]{"ipsum", "lorem"});
    assertBlock(file, 2, null, new String[]{"ipsum", null});
    assertBlock(file, 3, null, new String[]{null, "lorem"});
  }

  @Test
  public void testMultiBatchMixedDictionaries() throws Exception {
    File file = new File("target/mytest_multi_mixed_dictionaries.arrow");
    writeDataMultiBatchWithDictionaries(file, 3);

    try (FileInputStream fileInputStream = new FileInputStream(file);
         ArrowFileReader reader = new ArrowFileReader(fileInputStream.getChannel(), allocator);) {
      VectorSchemaRoot r = reader.getVectorSchemaRoot();
      FieldVector dictA = r.getVector("vectorA");
      FieldVector dictB = r.getVector("vectorB");

      reader.loadNextBatch();
      assertDictionary(dictA, reader, "foo", "bar");
      assertDictionary(dictB, reader, "lorem", "ipsum");

      reader.loadNextBatch();
      assertDictionary(dictA, reader, "meep", "bar");
      assertDictionary(dictB, reader, "ipsum", "lorem");

      reader.loadNextBatch();
      assertDictionary(dictA, reader, null, "bazz");
      assertDictionary(dictB, reader, "ipsum", null);

      reader.loadNextBatch();
      assertDictionary(dictA, reader, "bar", "zap");
      assertDictionary(dictB, reader, null, "lorem");
    }
  }

  @Test
  public void testMultiBatchMixedDictionariesOutOfOrder() throws Exception {
    File file = new File("target/mytest_multi_mixed_dictionaries_ooo.arrow");
    writeDataMultiBatchWithDictionaries(file, 3);

    try (FileInputStream fileInputStream = new FileInputStream(file);
         ArrowFileReader reader = new ArrowFileReader(fileInputStream.getChannel(), allocator);) {
      VectorSchemaRoot r = reader.getVectorSchemaRoot();
      FieldVector dictA = r.getVector("vectorA");
      FieldVector dictB = r.getVector("vectorB");

      reader.loadRecordBatch(reader.getRecordBlocks().get(2));
      assertDictionary(dictA, reader, null, "bazz");
      assertDictionary(dictB, reader, "ipsum", null);

      reader.loadRecordBatch(reader.getRecordBlocks().get(1));
      assertDictionary(dictA, reader, "meep", "bar");
      assertDictionary(dictB, reader, "ipsum", "lorem");

      reader.loadRecordBatch(reader.getRecordBlocks().get(3));
      assertDictionary(dictA, reader, "bar", "zap");
      assertDictionary(dictB, reader, null, "lorem");

      reader.loadRecordBatch(reader.getRecordBlocks().get(0));
      assertDictionary(dictA, reader, "foo", "bar");
      assertDictionary(dictB, reader, "lorem", "ipsum");
    }
  }

  @Test
  public void testMultiBatchMixedDictionariesSeek() throws Exception {
    File file = new File("target/mytest_multi_mixed_seek.arrow");
    writeDataMultiBatchWithDictionaries(file, 3);

    assertBlock(file, 0, new String[]{"foo", "bar"}, new String[]{"lorem", "ipsum"});
    assertBlock(file, 1, new String[]{"meep", "bar"}, new String[]{"ipsum", "lorem"});
    assertBlock(file, 2, new String[]{null, "bazz"}, new String[]{"ipsum", null});
    assertBlock(file, 3, new String[]{"bar", "zap"}, new String[]{null, "lorem"});
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

  private void assertBlock(File file, int block, String[] delta, String[] replace) throws Exception {
    try (FileInputStream fileInputStream = new FileInputStream(file);
         ArrowFileReader reader = new ArrowFileReader(fileInputStream.getChannel(), allocator);) {
      VectorSchemaRoot r = reader.getVectorSchemaRoot();
      FieldVector dictA = r.getVector("vectorA");
      FieldVector dictB = r.getVector("vectorB");

      reader.loadRecordBatch(reader.getRecordBlocks().get(block));
      if (delta != null) {
        assertDictionary(dictA, reader, delta);
      } else {
        assertNull(dictA);
      }
      if (replace != null) {
        assertDictionary(dictB, reader, replace);
      } else{
        assertNull(dictB);
      }
    }
  }
}

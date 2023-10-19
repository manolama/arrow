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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Collections2;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.UInt2Vector;
import org.apache.arrow.vector.ValueVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.dictionary.DeltaDictionary;
import org.apache.arrow.vector.dictionary.Dictionary;
import org.apache.arrow.vector.dictionary.DictionaryEncoder;
import org.apache.arrow.vector.dictionary.DictionaryProvider;
import org.apache.arrow.vector.ipc.message.ArrowBlock;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.DictionaryEncoding;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.junit.Test;
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
  public void newdict() throws Exception {
    File file = new File("target/mytest_multi_newdict.arrow");
    DictionaryProvider.MapDictionaryProvider provider = new DictionaryProvider.MapDictionaryProvider();
    DictionaryEncoding de = new DictionaryEncoding(42, false, new ArrowType.Int(16, false), true);
    DeltaDictionary dict = new DeltaDictionary(
        "vectorA",
        de,
        new ArrowType.Utf8(),
        new ArrowType.Int(16, false),
        allocator
    );
    provider.put(dict);

    dict.add("foo".getBytes(StandardCharsets.UTF_8));
    dict.add("bar".getBytes(StandardCharsets.UTF_8));

    VectorSchemaRoot root = VectorSchemaRoot.of(dict.getIndexVector());
    root.setRowCount(2);
    try (FileOutputStream fileOutputStream = new FileOutputStream(file);
         ArrowFileWriter arrowWriter = new ArrowFileWriter(root, provider, fileOutputStream.getChannel());) {

      // batch 1
      arrowWriter.start();
      arrowWriter.writeBatch();
      dict.reset();

      // batch 2
      dict.add("meep".getBytes(StandardCharsets.UTF_8));
      dict.add("bar".getBytes(StandardCharsets.UTF_8));

      root.setRowCount(2);
      arrowWriter.writeBatch();
      dict.reset();

      dict.add("bazz".getBytes(StandardCharsets.UTF_8));
      root.setRowCount(1);
      arrowWriter.writeBatch();dict.reset();

      dict.add("bar".getBytes(StandardCharsets.UTF_8));
      dict.add("foo".getBytes(StandardCharsets.UTF_8));
      root.setRowCount(2);
      arrowWriter.writeBatch();

      arrowWriter.end();
    }
    dict.close();

    System.out.println("-------------READ ");
    try (FileInputStream fileInputStream = new FileInputStream(file);
         ArrowFileReader reader = new ArrowFileReader(fileInputStream.getChannel(), allocator);) {
      for (ArrowBlock arrowBlock : reader.getRecordBlocks()) {
        reader.loadRecordBatch(arrowBlock);
        VectorSchemaRoot r = reader.getVectorSchemaRoot();
        FieldVector dv = r.getVector("vectorA");
        DictionaryEncoding dictionaryEncoding = dv.getField().getDictionary();
        Dictionary d = reader.getDictionaryVectors().get(dictionaryEncoding.getId());
        try (ValueVector readVector = DictionaryEncoder.decode(dv, d)) {
          System.out.println("Decoded data: " + readVector);
        }
      }
    }
  }

  @Test
  public void testMultiBatchWithOneDictionary() throws Exception {
    File file = new File("target/mytest_multi_delta_dictionary.arrow");
    writeSingleDeltaDictionaryB(file);

    try (FileInputStream fileInputStream = new FileInputStream(file);
         ArrowFileReader reader = new ArrowFileReader(fileInputStream.getChannel(), allocator);) {
      VectorSchemaRoot root = reader.getVectorSchemaRoot();
      assertEquals(reader.getRecordBlocks().size(), 4);
      assertTrue(reader.loadNextBatch());
      assertVectorA(reader, root, 0);

      assertTrue(reader.loadNextBatch());
      assertVectorA(reader, root, 1);

      assertTrue(reader.loadNextBatch());
      assertVectorA(reader, root, 2);

      assertTrue(reader.loadNextBatch());
      assertVectorA(reader, root, 3);

      assertFalse(reader.loadNextBatch());
    }

    // load just the second block
    try (FileInputStream fileInputStream = new FileInputStream(file);
         ArrowFileReader reader = new ArrowFileReader(fileInputStream.getChannel(), allocator);) {
      VectorSchemaRoot root = reader.getVectorSchemaRoot();
      assertEquals(4,reader.getRecordBlocks().size());
      assertTrue(reader.loadRecordBatch(reader.getRecordBlocks().get(1)));
      assertVectorA(reader, root, 1);
    }
  }

  private void assertVectorA(ArrowFileReader reader, VectorSchemaRoot root, int block) throws Exception {
    System.out.println("Block: " + block);
    FieldVector encoded = root.getVector("vectorA");
    System.out.println(encoded);
    DictionaryEncoding dictionaryEncoding = encoded.getField().getDictionary();
    Dictionary dictionary = reader.getDictionaryVectors().get(dictionaryEncoding.getId());
    try (ValueVector decoded = DictionaryEncoder.decode(encoded, dictionary)) {
      System.out.println(decoded);
      if (block == 0) {
        assertEquals("foo", decoded.getObject(0).toString());
        assertEquals("bar", decoded.getObject(1).toString());
      } else if (block == 1) {
        assertEquals("meep", decoded.getObject(0).toString());
        assertEquals("bar", decoded.getObject(1).toString());
      } else if (block == 2) {
        assertEquals("baz", decoded.getObject(0).toString());
      } else if (block == 3) {
        assertEquals("bar", decoded.getObject(0).toString());
        assertEquals("foo", decoded.getObject(1).toString());
      }
    }
  }

  private void assertVectorB(ArrowFileReader reader, VectorSchemaRoot root, int block) throws Exception {
    FieldVector encoded = root.getVector("vectorB");
    DictionaryEncoding dictionaryEncoding = encoded.getField().getDictionary();
    Dictionary dictionary = reader.getDictionaryVectors().get(dictionaryEncoding.getId());
    try (ValueVector decoded = DictionaryEncoder.decode(encoded, dictionary)) {
      if (block == 0) {
        assertEquals("a", decoded.getObject(0).toString());
        assertEquals("b", decoded.getObject(1).toString());
      } else if (block == 1) {
        assertEquals("b", decoded.getObject(0).toString());
        assertEquals("a", decoded.getObject(1).toString());
      } else if (block == 2) {
        assertNull(decoded.getObject(0));
        assertNull(decoded.getObject(1));
      } else if (block == 3) {
        assertNull(decoded.getObject(0));
        assertEquals("a", decoded.getObject(1).toString());
      }
    }
  }

  private void writeSingleDeltaDictionary(File file) throws Exception {
    Map<String, Integer> stringToIndex = new HashMap<>();

    try (VarCharVector dictionaryVector = new VarCharVector("dictionary", allocator)) {
      DictionaryEncoding dictionaryEncoding = new DictionaryEncoding(42, false, new ArrowType.Int(16, false), true);

      Dictionary dictionary = new Dictionary(dictionaryVector, dictionaryEncoding);
      DictionaryProvider.MapDictionaryProvider provider = new DictionaryProvider.MapDictionaryProvider();
      provider.put(dictionary);

      try (UInt2Vector vector = new UInt2Vector(
          "vectorA",
          new FieldType(false, new ArrowType.Int(16, false), dictionaryEncoding),
          allocator)) {
        vector.allocateNew();
        dictionaryVector.allocateNew(2);

        dictionaryVector.set(0, "foo".getBytes(StandardCharsets.UTF_8));
        stringToIndex.put("foo", 0);
        dictionaryVector.set(1, "bar".getBytes(StandardCharsets.UTF_8));
        stringToIndex.put("bar", 1);

        vector.set(0, stringToIndex.get("foo"));
        vector.set(1, stringToIndex.get("bar"));

        VectorSchemaRoot root = VectorSchemaRoot.of(dictionaryVector, vector);
        root.setRowCount(2);
        try (FileOutputStream fileOutputStream = new FileOutputStream(file);
             ArrowFileWriter arrowWriter = new ArrowFileWriter(root, provider, fileOutputStream.getChannel());) {

          // batch 1
          arrowWriter.start();
          arrowWriter.writeBatch();
          dictionaryVector.reset();
          vector.reset();

          // batch 2
          dictionaryVector.set(0, "meep".getBytes(StandardCharsets.UTF_8));
          stringToIndex.put("meep", 2);

          vector.set(0, stringToIndex.get("meep"));
          vector.set(1, stringToIndex.get("bar"));

          root.setRowCount(2);

          arrowWriter.writeBatch();
          arrowWriter.end();
        }
      }
    }
  }

  private void writeSingleDeltaDictionaryB(File file) throws Exception {
    DictionaryProvider.MapDictionaryProvider provider = new DictionaryProvider.MapDictionaryProvider();
    ManualDictionaryVectorA vectorA = new ManualDictionaryVectorA(allocator, provider);

    vectorA.next();
    VectorSchemaRoot root = VectorSchemaRoot.of(vectorA.vector);
    root.setRowCount(2);
    try (FileOutputStream fileOutputStream = new FileOutputStream(file);
         ArrowFileWriter arrowWriter = new ArrowFileWriter(root, provider, fileOutputStream.getChannel());) {

      arrowWriter.start();
      for (int i = 0; i < 4; i++) {
        root.setRowCount(i == 2 ? 1 : 2);
        arrowWriter.writeBatch();
        vectorA.next();
      }

      arrowWriter.end();
    }

    vectorA.close();
  }

  class ManualDictionaryVectorA {

    private final BufferAllocator allocator;
    private Map<String, Integer> stringToIndex = new HashMap<>();
    private int mapIndex;
    private UInt2Vector vector;
    private VarCharVector dictionaryVector;
    private DictionaryEncoding dictionaryEncoding;
    private int index;
    private int batch;

    ManualDictionaryVectorA(BufferAllocator allocator, DictionaryProvider.MapDictionaryProvider provider) {
      this.allocator = allocator;
      dictionaryVector = new VarCharVector("vectorA-dictionary", allocator);
      dictionaryVector.allocateNew();
      dictionaryEncoding = new DictionaryEncoding(42, false, new ArrowType.Int(16, false), true);
      Dictionary dictionary = new Dictionary(dictionaryVector, dictionaryEncoding);
      provider.put(dictionary);
      vector = new UInt2Vector(
          "vectorA",
          new FieldType(false, new ArrowType.Int(16, false), dictionaryEncoding),
          allocator);
      vector.allocateNew();
    }

    void next() {
      if (batch == 0) {
        stringToIndex.put("foo", mapIndex++);
        dictionaryVector.set(index, "foo".getBytes(StandardCharsets.UTF_8));
        vector.set(index++, stringToIndex.get("foo"));

        stringToIndex.put("bar", mapIndex++);
        dictionaryVector.set(index, "bar".getBytes(StandardCharsets.UTF_8));
        vector.set(index++, stringToIndex.get("bar"));
        dictionaryVector.setValueCount(2);
        vector.setValueCount(2);
      } else if (batch == 1) {
        dictionaryVector.reset();
        vector.reset();
        index = 0;

        stringToIndex.put("meep", mapIndex++);
        dictionaryVector.set(index, "meep".getBytes(StandardCharsets.UTF_8));
        dictionaryVector.setValueCount(1);
        vector.set(index++, stringToIndex.get("meep"));
        vector.set(index++, stringToIndex.get("bar"));
        dictionaryVector.setValueCount(1);
        vector.setValueCount(2);
      } else if (batch == 2) {
        dictionaryVector.reset();
        vector.reset();
        index = 0;

        stringToIndex.put("baz", mapIndex++);
        dictionaryVector.set(index, "baz".getBytes(StandardCharsets.UTF_8));
        vector.set(index++, stringToIndex.get("baz"));
        dictionaryVector.setValueCount(1);
        vector.setValueCount(1);
      } else if (batch == 3) {
        dictionaryVector.reset();
        vector.reset();
        index = 0;

        vector.set(index++, stringToIndex.get("bar"));
        vector.set(index++, stringToIndex.get("foo"));
        dictionaryVector.setValueCount(0);
        vector.setValueCount(2);
      }
      batch++;
    }

    void close() {
      dictionaryVector.close();
      vector.close();
    }

  }
}

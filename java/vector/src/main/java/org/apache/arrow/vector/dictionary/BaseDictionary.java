package org.apache.arrow.vector.dictionary;

import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.DictionaryEncoding;

/**
 * Interface for all dictionary types.
 */
public interface BaseDictionary {

  /**
   * The dictionary vector containing unique entries.
   */
  FieldVector getVector();

  /**
   * The encoding used for the dictionary vector.
   */
  DictionaryEncoding getEncoding();

  /**
   * The type of the dictionary vector.
   */
  ArrowType getVectorType();

}

package org.apache.arrow.vector.dictionary;

import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.DictionaryEncoding;

public interface BaseDictionary {

  public FieldVector getVector();

  public DictionaryEncoding getEncoding();

  public ArrowType getVectorType();

}

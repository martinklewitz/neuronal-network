package info.klewitz.kaggle.forest

class NormalizerTest extends GroovyTestCase {

  Normalizer normalizer

  @Override
  void setUp() {
    normalizer = new Normalizer()
  }

  void testRead() {
    def listOfLists = normalizer.read("test.csv").getData()
    assertEquals(10, listOfLists.size())
    assertEquals(56, listOfLists[0].size())
  }

  void testReadAndRemove() {
    def listOfLists = normalizer.read("test.csv").removeRow(0).getData()
    assertEquals(10, listOfLists.size())
    listOfLists.each {
      assertEquals(55, it.size())
    }
  }

  void testReadAndToArray() {
    def listOfLists = normalizer.read("test.csv").getDataAsArray()
    assertEquals(10, listOfLists.length)
    listOfLists.each {
      assertEquals(56, it.size())
    }
  }

  void testSpreadIntegerAsRows() {
    def listOfLists = normalizer.read("test.csv").spreadIntegerValueAsRows(55, 7).getDataAsArray()
    assertEquals(10, listOfLists.length)
    listOfLists.each {
      assertEquals(62, it.size())
    }
    assertEquals(0.0, listOfLists[0][55])
    assertEquals(0.0, listOfLists[0][56])
    assertEquals(0.0, listOfLists[0][57])
    assertEquals(0.0, listOfLists[0][58])
    assertEquals(1.0, listOfLists[0][59])
    assertEquals(0.0, listOfLists[0][60])
    assertEquals(0.0, listOfLists[0][61])

    assertEquals(0.0, listOfLists[2][55])
    assertEquals(1.0, listOfLists[2][56])
    assertEquals(0.0, listOfLists[2][57])
    assertEquals(0.0, listOfLists[2][58])
    assertEquals(0.0, listOfLists[2][59])
    assertEquals(0.0, listOfLists[2][60])
    assertEquals(0.0, listOfLists[2][61])
  }

  void testNormalize() {
    def listOfLists = normalizer.read("test.csv").normalize().getDataAsArray()
    assertEquals(10, listOfLists.length)
    listOfLists.each {
      assertEquals(56, it.size())
    }
    assertEquals(0.0, listOfLists[0][1])
    assertEquals(0.05, listOfLists[1][1])
    assertEquals(0.1, listOfLists[2][1])
    assertEquals(0.5, listOfLists[3][1])
    assertEquals(0.45, listOfLists[4][1])
    assertEquals(0.55, listOfLists[5][1])
    assertEquals(0.5, listOfLists[6][1])
    assertEquals(0.9, listOfLists[7][1])
    assertEquals(0.95, listOfLists[8][1])
    assertEquals(1.0, listOfLists[9][1])
  }
}

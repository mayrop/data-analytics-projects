package uk.ac.gla.dcs.dsms;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;

import org.junit.Test;
import org.terrier.indexing.IndexTestUtils;
import org.terrier.structures.Index;
import org.terrier.structures.postings.BlockPosting;
import org.terrier.structures.postings.IterablePosting;
import org.terrier.tests.ApplicationSetupBasedTest;
import org.terrier.utility.ApplicationSetup;

/** 
 * In this class we test the average distance
 * for all position combinations in document.
 *
 * Feature implemented/tested is: avg_dist(a,b,D)
 *
 * This is proximity feature 4 mentioned in this paper:
 * http://delivery.acm.org/10.1145/1580000/1571986/p251-cummins.pdf
 * 
 * @author 2419105v@student.gla.ac.uk
 */
public class TestAvgDistanceDependencyScoreModifier extends ApplicationSetupBasedTest
{
	@Test public void testWithTwoTerms00() throws Exception {
		// boolean[] okToUse = new boolean[] {true, true};
		// // Get posting iterators for two terms 'a' and 'b'
		// IterablePosting[] ips = getTestPostings(
		// 	new String[]{"a b c d"}, 
		// 	"a,b",
		// 	okToUse
		// );

		// assertEquals(1.0d, calculateScore(ips, okToUse), 0.01d);


		//make an index with a single sample document
		ApplicationSetup.setProperty("termpipelines", "");
		Index index = IndexTestUtils.makeIndexBlocks(
				new String[]{"docno", "docno2", "docno3"}, 
				new String[]{"The quick fox fox my my brown fox jumps over the lazy dog", "The brown whatever", "The brown brown, brown, whatever"});

		//get posting iterators for two terms 'fox' and 'jumps'
		IterablePosting[] ips = new IterablePosting[3];
		ips[0] = index.getInvertedIndex().getPostings(index.getLexicon().getLexiconEntry("fox"));
		ips[1] = index.getInvertedIndex().getPostings(index.getLexicon().getLexiconEntry("jumps"));
		ips[2] = index.getInvertedIndex().getPostings(index.getLexicon().getLexiconEntry("brown"));
		ips[0].next();
		ips[1].next();
		ips[2].next();
		assertEquals(0, ips[0].getId());
		assertEquals(0, ips[1].getId());
		System.out.println("Positions of term 'fox'="+ Arrays.toString( ((BlockPosting)ips[0]).getPositions()));
		System.out.println("Positions of term 'jumps'="+ Arrays.toString( ((BlockPosting)ips[1]).getPositions()));
		System.out.println("Positions of term 'brown'="+ Arrays.toString( ((BlockPosting)ips[2]).getPositions()));
		System.out.println(ips[2]);

		// SampleProxFeatureDSM sample = new SampleProxFeatureDSM();
		// double score = sample.calculateDependence(
  //           ips, //posting lists
  //           new boolean[]{true,true},  //is this posting list on the correct document?
  //           new double[]{1d,1d}, false//doesnt matter
		// );
		// System.out.println(score);
		//TODO: make your assertion about what the score should be
		//assertEquals(XXX, score, 0.0d);
	}

	// @Test public void testWithTwoTerms01() throws Exception {
	// 	boolean[] okToUse = new boolean[] {true, true};
	// 	// Get posting iterators for two terms 'a' and 'b'
	// 	IterablePosting[] ips = getTestPostings(
	// 		new String[]{"a b c d a b d e f g h a i j"}, 
	// 		"a,b",
	// 		okToUse
	// 	);
		
	// 	assertEquals(4.33d, calculateScore(ips, okToUse), 0.01d);
	// }

	// @Test public void testWithTwoTerms02() throws Exception {
	// 	boolean[] okToUse = new boolean[] {true, true};
	// 	// Get posting iterators for two terms 'a' and 'b'
	// 	IterablePosting[] ips = getTestPostings(
	// 		new String[]{"a c d b a x b d e f g h a i j"}, 
	// 		"a,b",
	// 		okToUse
	// 	);
		
	// 	assertEquals(4.5d, calculateScore(ips, okToUse), 0.0d);
	// }


	// @Test public void testWithTwoTerms03() throws Exception {
	// 	boolean[] okToUse = new boolean[] {true, true};
	// 	// Get posting iterators for two terms 'a' and 'b'
	// 	IterablePosting[] ips = getTestPostings(
	// 		new String[]{"a 1 2 3 b 1 a 1 2 b d e f g b a j"}, 
	// 		"a,b",
	// 		okToUse
	// 	);
		
	// 	assertEquals(6.44d, calculateScore(ips, okToUse), 0.01d);
	// }


	// @Test public void testWithTwoTerms04() throws Exception {
	// 	boolean[] okToUse = new boolean[] {true, true};
	// 	// Get posting iterators for two terms 'a' and 'b'
	// 	IterablePosting[] ips = getTestPostings(
	// 		new String[]{"a 1 2 3 b 1 x a 1 2 b d e f g h b x a i j"}, 
	// 		"a,b",
	// 		okToUse
	// 	);
		
	// 	assertEquals(7.66d, calculateScore(ips, okToUse), 0.01d);
	// }


	// @Test public void testWithTwoTerms05() throws Exception {
	// 	boolean[] okToUse = new boolean[] {true, true};
	// 	// Get posting iterators for two terms 'a' and 'b'
	// 	IterablePosting[] ips = getTestPostings(
	// 		new String[]{"a 1 2 3 b 1 2 3 4 5 a 1 b 1 2 b d e f g h a i j x b a"}, 
	// 		"a,b",
	// 		okToUse
	// 	);
		
	// 	assertEquals(10.5d, calculateScore(ips, okToUse), 0.0d);
	// }


	// @Test public void testWithThreeTerms01() throws Exception {
	// 	boolean[] okToUse = new boolean[] {true, true, true};
	// 	IterablePosting[] ips = getTestPostings(
	// 		new String[]{"t1 t2 t1 t3 t5 t4 t2 t3 t4"}, 
	// 		"t1,t2,t3",
	// 		okToUse
	// 	);
		
	// 	assertEquals(10.0d, calculateScore(ips, okToUse), 0.0d);
	// }


	// @Test public void testWithThreeTerms02() throws Exception {
	// 	boolean[] okToUse = new boolean[] {true, true, true};
	// 	IterablePosting[] ips = getTestPostings(
	// 		new String[]{"t1 t5 t2 t6 t1 t3 t5 t4 t2 t5 t6 t7 t3 t4"}, 
	// 		"t1,t2,t3",
	// 		okToUse
	// 	);
		
	// 	assertEquals(15.5d, calculateScore(ips, okToUse), 0.0d);
	// }


	// @Test public void testWithThreeTerms03() throws Exception {
	// 	boolean[] okToUse = new boolean[] {true, true, true};
	// 	IterablePosting[] ips = getTestPostings(
	// 		new String[]{"t1 t5 t2 t6 t3 t1 t5 t4 t2 t5 t6 t7 t3 t4"}, 
	// 		"t1,t2,t3",
	// 		okToUse
	// 	);
		
	// 	assertEquals(15.0d, calculateScore(ips, okToUse), 0.0d);
	// }


	// @Test public void testWithThreeTerms04() throws Exception {
	// 	boolean[] okToUse = new boolean[] {true, true, true};
	// 	IterablePosting[] ips = getTestPostings(
	// 		new String[]{"t1 t5 t2 t6 t2 t1 t5 t4 t2 t5 t6 t7 t3 t4"}, 
	// 		"t1,t2,t3",
	// 		okToUse
	// 	);
		
	// 	assertEquals(20.33d, calculateScore(ips, okToUse), 0.01d);
	// }


	// @Test public void testWithThreeTerms05() throws Exception {
	// 	boolean[] okToUse = new boolean[] {true, true, true};
	// 	IterablePosting[] ips = getTestPostings(
	// 		new String[]{"t1 t5 t2 t3 t2 t7 t1 t5 t4 t2 t5 t6 t7 t3 t4"}, 
	// 		"t1,t2,t3",
	// 		okToUse
	// 	);
		
	// 	assertEquals(15.83d, calculateScore(ips, okToUse), 0.01d);
	// }


	// @Test public void testWithThreeTerms07() throws Exception {
	// 	boolean[] okToUse = new boolean[] {true, true, true};
	// 	IterablePosting[] ips = getTestPostings(
	// 		new String[]{"t1 t5 t2 t9 t3 t6 t8 t2 t7 t1 t5 t4 t2 t5 t6 t7 t3 t4"}, 
	// 		"t1,t2,t3",
	// 		okToUse
	// 	);
		
	// 	assertEquals(20.16d, calculateScore(ips, okToUse), 0.01d);
	// }


	// private IterablePosting[] getTestPostings(String[] documents, String string, boolean[] okToUse) throws Exception {
	// 	ApplicationSetup.setProperty("termpipelines", "");

	// 	// Make an index with a single sample document		
	// 	Index index = IndexTestUtils.makeIndexBlocks(
	// 		new String[]{"doc1"}, 
	// 		documents
	// 	);

	// 	String[] terms = string.split(",");
		
	// 	IterablePosting[] postings = new IterablePosting[terms.length];

	// 	for (int i = 0, length = terms.length; i < length; i++) {
	// 		postings[i] = okToUse[i]
	// 			? index.getInvertedIndex().getPostings(index.getLexicon().getLexiconEntry(terms[i]))
	// 			: null;
	// 		if (okToUse[i]) {
	// 			postings[i].next();
	// 		}
	// 	}

	// 	return postings;
	// } 


	// private Double calculateScore(IterablePosting[] ips, boolean[] okToUse) {
	// 	AvgDistanceDependencyScoreModifier sample = new AvgDistanceDependencyScoreModifier();

	// 	return sample.calculateDependence(
	// 		ips, // posting lists
	// 		okToUse,  // is this posting list on the correct document?
	// 		new double[] {1d, 1d, 1d, 1d, 1d, 1d, 1d, 1d, 1d}, 
	// 		true // does not matter as it does always full dependency as requested
	// 	);
	// }	
}

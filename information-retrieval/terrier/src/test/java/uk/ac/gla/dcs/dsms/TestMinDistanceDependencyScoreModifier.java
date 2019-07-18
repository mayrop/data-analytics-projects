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
 * In this class we test the minimum distance
 * between any occurrences of a and b in a document.
 *
 * Feature implemented/tested is: min_dist(a,b,D)
 *
 * This is proximity feature 1 mentioned in this paper:
 * http://delivery.acm.org/10.1145/1580000/1571986/p251-cummins.pdf
 * 
 * @author 2419105v@student.gla.ac.uk
 */
public class TestMinDistanceDependencyScoreModifier extends ApplicationSetupBasedTest
{
	@Test public void testWithTwoTerms00() throws Exception {
		boolean[] okToUse = new boolean[] {true, true};
		// Get posting iterators for two terms 'a' and 'b'
		IterablePosting[] ips = getTestPostings(
			new String[]{"a b c d"}, 
			"a,b",
			okToUse
		);

		assertEquals(1.0d, calculateScore(ips, okToUse), 0.0d);
	}

	@Test public void testWithTwoTerms01() throws Exception {
		boolean[] okToUse = new boolean[] {true, true};
		// Get posting iterators for two terms 'a' and 'b'
		IterablePosting[] ips = getTestPostings(
			new String[]{"a b c d a b d e f g h a i j"}, 
			"a,b",
			okToUse
		);
		
		assertEquals(1.0d, calculateScore(ips, okToUse), 0.0d);
	}

	@Test public void testWithTwoTerms02() throws Exception {
		boolean[] okToUse = new boolean[] {true, true};
		// Get posting iterators for two terms 'a' and 'b'
		IterablePosting[] ips = getTestPostings(
			new String[]{"a c d b a x b d e f g h a i j"}, 
			"a,b",
			okToUse
		);
		
		assertEquals(1.0d, calculateScore(ips, okToUse), 0.0d);
	}


	@Test public void testWithTwoTerms03() throws Exception {
		boolean[] okToUse = new boolean[] {true, true};
		// Get posting iterators for two terms 'a' and 'b'
		IterablePosting[] ips = getTestPostings(
			new String[]{"a 1 2 3 b 1 a 1 2 b d e f g b a j"}, 
			"a,b",
			okToUse
		);
		
		assertEquals(1.0d, calculateScore(ips, okToUse), 0.0d);
	}


	@Test public void testWithTwoTerms04() throws Exception {
		boolean[] okToUse = new boolean[] {true, true};
		// Get posting iterators for two terms 'a' and 'b'
		IterablePosting[] ips = getTestPostings(
			new String[]{"a 1 2 3 b 1 x a 1 2 b d e f g h b x a i j"}, 
			"a,b",
			okToUse
		);
		
		assertEquals(2.0d, calculateScore(ips, okToUse), 0.0d);
	}


	@Test public void testWithTwoTerms05() throws Exception {
		boolean[] okToUse = new boolean[] {true, true};
		// Get posting iterators for two terms 'a' and 'b'
		IterablePosting[] ips = getTestPostings(
			new String[]{"a 1 2 3 b 1 2 3 4 5 a 1 b 1 2 b d e f g h a i j x b a"}, 
			"a,b",
			okToUse
		);
		
		assertEquals(1.0d, calculateScore(ips, okToUse), 0.0d);
	}


	@Test public void testWithThreeTerms01() throws Exception {
		boolean[] okToUse = new boolean[] {true, true, true};
		IterablePosting[] ips = getTestPostings(
			new String[]{"t1 t2 t1 t3 t5 t4 t2 t3 t4"}, 
			"t1,t2,t3",
			okToUse
		);
		
		assertEquals(1.0d, calculateScore(ips, okToUse), 0.0d);
	}


	@Test public void testWithThreeTerms02() throws Exception {
		boolean[] okToUse = new boolean[] {true, true, true};
		IterablePosting[] ips = getTestPostings(
			new String[]{"t1 t5 t2 t6 t1 t3 t5 t4 t2 t5 t6 t7 t3 t4"}, 
			"t1,t2,t3",
			okToUse
		);
		
		assertEquals(1.0d, calculateScore(ips, okToUse), 0.0d);
	}


	@Test public void testWithThreeTerms03() throws Exception {
		boolean[] okToUse = new boolean[] {true, true, true};
		IterablePosting[] ips = getTestPostings(
			new String[]{"t1 t5 t2 t6 t3 t1 t5 t4 t2 t5 t6 t7 t3 t4"}, 
			"t1,t2,t3",
			okToUse
		);
		
		assertEquals(1.0d, calculateScore(ips, okToUse), 0.0d);
	}


	@Test public void testWithThreeTerms04() throws Exception {
		boolean[] okToUse = new boolean[] {true, true, true};
		IterablePosting[] ips = getTestPostings(
			new String[]{"t1 t5 t2 t6 t2 t1 t5 t4 t2 t5 t6 t7 t3 t4"}, 
			"t1,t2,t3",
			okToUse
		);
		
		assertEquals(1.0d, calculateScore(ips, okToUse), 0.0d);
	}


	@Test public void testWithThreeTerms05() throws Exception {
		boolean[] okToUse = new boolean[] {true, true, true};
		IterablePosting[] ips = getTestPostings(
			new String[]{"t1 t5 t2 t3 t2 t7 t1 t5 t4 t2 t5 t6 t7 t3 t4"}, 
			"t1,t2,t3",
			okToUse
		);
		
		assertEquals(1.0d, calculateScore(ips, okToUse), 0.0d);
	}


	@Test public void testWithThreeTerms07() throws Exception {
		boolean[] okToUse = new boolean[] {true, true, true};
		IterablePosting[] ips = getTestPostings(
			new String[]{"t1 t5 t2 t9 t3 t6 t8 t2 t7 t1 t5 t4 t2 t5 t6 t7 t3 t4"}, 
			"t1,t2,t3",
			okToUse
		);
		
		assertEquals(2.0d, calculateScore(ips, okToUse), 0.0d);
	}


	private IterablePosting[] getTestPostings(String[] documents, String string, boolean[] okToUse) throws Exception {
		ApplicationSetup.setProperty("termpipelines", "");

		// Make an index with a single sample document		
		Index index = IndexTestUtils.makeIndexBlocks(
			new String[]{"doc1"}, 
			documents
		);

		String[] terms = string.split(",");
		
		IterablePosting[] postings = new IterablePosting[terms.length];

		for (int i = 0, length = terms.length; i < length; i++) {
			postings[i] = okToUse[i]
				? index.getInvertedIndex().getPostings(index.getLexicon().getLexiconEntry(terms[i]))
				: null;
			if (okToUse[i]) {
				postings[i].next();
			}
		}

		return postings;
	} 


	private Double calculateScore(IterablePosting[] ips, boolean[] okToUse) {
		MinDistanceDependencyScoreModifier sample = new MinDistanceDependencyScoreModifier();

		return sample.calculateDependence(
			ips, // posting lists
			okToUse,  // is this posting list on the correct document?
			new double[] {1d, 1d, 1d, 1d, 1d, 1d, 1d, 1d, 1d}, 
			true // does not matter as it does always full dependency as requested
		);
	}
}

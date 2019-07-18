package uk.ac.gla.dcs.dsms;

import java.util.Arrays;
import java.lang.String;
import java.lang.Integer;
import java.lang.Double;
import org.terrier.structures.postings.BlockPosting;
import org.terrier.structures.postings.Posting;
import org.terrier.matching.dsms.DependenceScoreModifier;
import java.lang.RuntimeException;

/** 
 * In this class we calculate the minimum distance
 * between any occurrences of a and b in a document.
 *
 * Feature implemented is: min_dist(a,b,D)
 *
 * This is proximity feature 1 mentioned in this paper:
 * http://delivery.acm.org/10.1145/1580000/1571986/p251-cummins.pdf
 * 
 * @author 2419105v@student.gla.ac.uk
 */
public class MinDistanceDependencyScoreModifier extends DependenceScoreModifier {
    
    /** 
     * This class is passed the postings of the current document,
     * and should return a score to represent that document.
     */
    @Override
    protected double calculateDependence(
        Posting[] ips, // postings for this document (these are actually IterablePosting[])
        boolean[] okToUse,  // is each posting on the correct document?
        double[] phraseTermWeights, // not used
        boolean SD // sequential dependence vs full dependence
    ) {
        final int numberOfQueryTerms = okToUse.length;

        Double score = null;

        for (int i = 0; i < numberOfQueryTerms; i++) {
            // no need to check proximity if this term does not exist
            if (!okToUse[i]) {
                continue;
            }

            INNERTERMS: for (int j = 0; j < numberOfQueryTerms; j++) {
                // We make sure we don't compare same terms
                // We also make sure that we don't compare term[j] if it does not exist
                if (j == i || !okToUse[j]) {
                    continue INNERTERMS;
                }

                Double temp = calculateProximity(ips, i, j);
                
                if (score == null || temp < score) {
                    score = temp;
                }
            }
        }

        return score;
    }

    private Double calculateProximity(Posting[] ips, int i, int j) {
        int positions[][] = getPositionsByPostings(((BlockPosting) ips[i]), ((BlockPosting) ips[j]));

        // we set the minimum distance to be the same as doc length (so it's the max)
        // we assume ips[i].getDocumentLength == ips[j].getDocumentLength
        return calculateProximityFromPositions(positions, ((BlockPosting) ips[i]).getDocumentLength());    
    }

    private int[][] getPositionsByPostings(BlockPosting posting1, BlockPosting posting2) {
        int[][] positions = new int[2][];
    
        positions[0] = posting1.getPositions();
        positions[1] = posting2.getPositions();

        // i would assume these are sorted, but just in case
        Arrays.sort(positions[0]);
        Arrays.sort(positions[1]);

        // We want to make sure that positions length
        // is less than positions2 so we iterate through the one
        // with the least amount of terms
        if (posting2.getFrequency() < posting1.getFrequency()) {
            // swaping
            int[] temp = positions[0];
            positions[0] = positions[1];
            positions[1] = temp;
        }

        return positions;
    }

    private Double calculateProximityFromPositions(int[][] positions, int currentDocLength) {
        int min = currentDocLength;
        int lengthJ = positions[1].length;
        int diff;
        int j = 1;

        for (int i = 0, length = positions[0].length; i < length; i++) {
            int localMin = currentDocLength;

            // the first loop will start at j = 0
            // further loops don't need to start from 0 as positiongs are sorted
            // we only need to check the previous index
            INNER: for (j = j - 1; j < lengthJ; j++) {
                diff = Math.abs(positions[0][i] - positions[1][j]);

                // Global min
                if (diff < min) {
                    min = diff;
                    localMin = diff;
                // Local min
                } else if (diff < localMin) {
                    localMin = diff;
                } else {
                    // since positions are sorted
                    // no need to check further terms since they're further away
                    break INNER;
                }
            }
        }

        return Double.valueOf(min);
    }

    /** You do NOT need to implement this method */
    @Override
    protected double scoreFDSD(int matchingNGrams, int docLength) {
        throw new UnsupportedOperationException();
    }


    @Override
    public String getName() {
        return "Min Distance Dependency Score Modifier";
    }
}
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
 * In this class we calculate the average distance
 * for all position combinations in document.
 *
 * Feature implemented is: avg_dist(a,b,D)
 *
 * This is proximity feature 4 mentioned in this paper:
 * http://delivery.acm.org/10.1145/1580000/1571986/p251-cummins.pdf
 * 
 * @author 2419105v@student.gla.ac.uk
 */
public class AvgDistanceDependencyScoreModifier extends DependenceScoreModifier {
    
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
    ) 
    {
        final int numberOfQueryTerms = okToUse.length;
        Double score = 0.0d;

        // Outer loop goes from 0 to the previous to last term
        for (int i = 0; i < numberOfQueryTerms - 1; i++) {
            if (!okToUse[i]) {
                continue;
            }

            // Inner loop goes from i + 1 to the last term
            INNERTERMS: for (int j = i + 1; j < numberOfQueryTerms; j++) {
                // we don't calculate dependence if term i or term j is not present
                if (!okToUse[j]) {
                    continue INNERTERMS;
                }
                
                score += calculateProximity(ips, i, j);
            }
        } 

        return score;
    }

    private Double calculateProximity(Posting[] ips, int i, int j)
    {
        int positions[][] = getPositionsByPostings(((BlockPosting) ips[i]), ((BlockPosting) ips[j]));

        // we set the minimum distance to be the same as doc length (so it's the max)
        // we assume ips[i].getDocumentLength == ips[j].getDocumentLength
        return calculateProximityFromPositions(positions, ((BlockPosting) ips[i]).getDocumentLength());    
    }

    private int[][] getPositionsByPostings(BlockPosting ips1, BlockPosting ips2)
    {
        int[][] positions = new int[2][];
    
        positions[0] = ips1.getPositions();
        positions[1] = ips2.getPositions();

        // i would assume these are sorted, but just in case
        Arrays.sort(positions[0]);
        Arrays.sort(positions[1]);

        // We want to make sure that positions length
        // is less than positions2 so we iterate through the one
        // with the least amount of terms
        if (ips2.getFrequency() < ips1.getFrequency()) {
            // swaping
            int[] temp = positions[0];
            positions[0] = positions[1];
            positions[1] = temp;
        }

        return positions;
    }

    protected Double calculateProximityFromPositions(int[][] positions, int min)
    {
        Double proximity = 0.0d;

        for (int i = 0, length = positions[0].length; i < length; i++) {
            for (int j = 0, lengthJ = positions[1].length; j < lengthJ; j++) {
                proximity += Math.abs(positions[0][i] - positions[1][j]);
            }
        }

        return proximity / (Double.valueOf(positions[0].length) * Double.valueOf(positions[1].length));
    }

    /** You do NOT need to implement this method */
    @Override
    protected double scoreFDSD(int matchingNGrams, int docLength) {
        throw new UnsupportedOperationException();
    }


    @Override
    public String getName() {
        return "Average Distance Dependency Score Modifier";
    }
}

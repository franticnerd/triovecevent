package vmf;

import java.io.FileNotFoundException;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;
import java.util.Scanner;

// import org.apache.commons.math3.distribution.LogNormalDistribution;

// import data.DocumentDataModel;
// import data.DocumentDataSet;
// import data.TermDataEmbModel;
// import data.TextDataModelAsWordEmbVecs;
// import jdistlib.math.Bessel;
import util.MapSorter;
import util.VecOp;


public class preCalBessel {
	

	public double[] besselTabel;
    public int tableSize;
    public double maxBessel;
    public int s;
    public double threshold;
	
	public double t1ini;

	public preCalBessel(int tableSize, double maxBessel, int s, double threshold) {
        this.s = s;
        this.maxBessel = maxBessel;
        this.tableSize = tableSize;
        this.besselTabel = new double[tableSize];
        this.threshold = threshold;
        this.t1ini = Math.sqrt(s / (2 * Math.PI)) / (1 + 1 / (12 * s) + 1 / (288 * s * s) - 139 / (51840 * s * s * s));
    }

    public void calculate(){
    	for (int i = 0; i < tableSize; ++i) {
    		System.out.println(i);
    		besselTabel[i] = calculateBessel(i * maxBessel / tableSize);
    	}
    }

    public double calculateBessel(double x){
    	double r = 1;
    	double t1 = Math.pow(x * Math.E/ (2 * s), s) * this.t1ini;
    	double m = 1 / (double) s;
    	double k = 1;
    	while (true) {
    		r = r * 0.25 * x * x / (k * ( s + k));
    		m = m + r;
    		// System.out.println(r+"\t"+m+"\t"+r/m);
    		if ((r/m) < threshold) {
    			break;
    		}
    		k = k + 1;
    	}
    	return t1 * m;
    }

    public void saveTable(String fileName) throws Exception {
    	System.out.println(fileName);

        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(fileName)));
        bw.write(maxBessel + "\t" + tableSize + "\t"+ s+ "\n");
        for (int i = 0; i < tableSize; ++i) {
            bw.write(besselTabel[i] + "\n");
        }
        bw.close();
    }

	static public void main(String[] args) throws Exception {
		preCalBessel pre99 = new preCalBessel(500000, 900, 99, 0.0001);
		pre99.calculate();
		pre99.saveTable("/shared/data/ll2/tweets/besselTable/bessel99.txt");
		preCalBessel pre100 = new preCalBessel(500000, 900, 100, 0.0001);
		pre99.calculate();
		pre99.saveTable("/shared/data/ll2/tweets/besselTable/bessel100.txt");
	}
}

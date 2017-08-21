package vmf;

import java.io.FileNotFoundException;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;
import java.util.Scanner;

public class oneWindowDataFull{
    public oneWindowDataFull(String fileName, int setSize, int embLen) throws Exception {
        this.x = new double[setSize][embLen];
        this.d = new double[setSize][2];
        this.z = new int[setSize];
        this.setSize = setSize;
        File file = new File(fileName);
        Scanner scan = new Scanner(file);
        for (int i = 0; i < setSize; ++i) {
            for (int j = 0; j < embLen; ++j) {
                x[i][j] = scan.nextDouble();
            }
            d[i][0] = scan.nextDouble();
            d[i][1] = scan.nextDouble();
        }
    }
    
    double[][] d;
    double[][] x;
    int[] z;
    int setSize;
}

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

import jdistlib.math.Bessel;
import util.MapSorter;
import util.VecOp;

import vmf.oneWindowData;

public class onlineVmfMixture {
    
    public onlineVmfMixture(double alpha, double r0, double c0, int dim, int clusNum, boolean exitOnFault, boolean kappaUpdateMode, double kappaStep, double virtualVmfNum, boolean fastProportion, boolean debugMode, boolean uniformIni) {//, double threshold) {
        this.uniformIni = uniformIni;
        this.alpha = alpha;
        this.R0 = r0;
        this.C0 = c0;
        this.embLen = dim;
        this.numVmf = clusNum;
        this.calculator = new besselCal(dim*0.5 - 1, exitOnFault);//, threshold);
        this.kappaUpdateMode = kappaUpdateMode;
        this.kappaStep = kappaStep;
        this.virtualVmfNum = virtualVmfNum;
        this.fastProportion = fastProportion;
        this.debugMode = debugMode;
    }
    
    public boolean uniformIni;
    public boolean debugMode;
    public boolean fastProportion;
    public boolean noneMultiple;
    public int numVmf = 10;
    public int embLen;
    public int noneEmptyVmf = 0;
    public final static double emptyThreshold = 0.01;
    public int sumSumZs = 0;
    public boolean kappaUpdateMode;
    public double virtualVmfNum;
    public double emptyProportion;

    public double[] distVmf;
    public double[][] musVmf;
    public double[] kappasVmf;
    
    public besselCal calculator;

    double[][] sumXs;
    double[] sumZs;
    // double totSumZs = 0.0;

    public double alpha;
    public double R0;
    public double C0;
    public double[] mu0;

    public double kappaIniScale = Math.log(100), kappaIniShape = 0.01; 
    // public double logNormalScale = 100, logNormalShape = 0.01; 
    // public LogNormalDistribution logNormal = new LogNormalDistribution(Math.log(logNormalScale), logNormalShape);
    public double kappaStep;
    // 1 - true
    // 0.01 - false
    
    private double proposeKappa(double initKappa) {
        double ret;
        if (kappaUpdateMode) {
            double oneStep = kappaStep * rand.nextGaussian();
            ret = initKappa + oneStep;
            if (ret <= besselCal.minValue || ret >= besselCal.maxValue)
                ret = initKappa - oneStep;
        } else {
            double oneStep = kappaStep * rand.nextGaussian();
            ret = Math.exp(Math.log(initKappa) + oneStep);
            if (ret <= besselCal.minValue || ret >= besselCal.maxValue)
                ret = Math.exp(Math.log(initKappa) - oneStep);
        }
//      System.out.println("init kappa:" + initKappa + ", proposed kappa:" + ret);
        return ret;
    }
    
    private static Random rand = new Random(1234567);

    public void initVmfMixture() {
        initVmfMixture(null);
    }
    
    public void initVmfMixture(double[][] musVmfInit) {
        distVmf   = new double[numVmf];
        musVmf    = new double[numVmf][embLen];
        kappasVmf = new double[numVmf];
        sumXs = new double[numVmf][embLen];
        sumZs = new double[numVmf];
        
        for (int h = 0; h < numVmf; ++h) {
            distVmf[h] = uniformIni ? 1 : rand.nextDouble();
            kappasVmf[h] = Math.exp(kappaIniShape * rand.nextGaussian() + kappaIniScale);
            // System.out.println("kappaIni: "+kappasVmf[h]+"nextGaussian: "+rand.nextGaussian()+"kappaIniShape: "+kappaIniShape+"kappaIniScale: "+kappaIniScale);
            if (musVmfInit == null) {
                for (int j = 0; j < embLen; ++j) musVmf[h][j] = rand.nextGaussian();
            } else {
                for (int j = 0; j < embLen; ++j) musVmf[h][j] = musVmfInit[h][j];
            }
            musVmf[h] = VecOp.normalize(musVmf[h]);
        }
        distVmf = VecOp.vec2Dist(distVmf);

        mu0 = new double[embLen];
        for (int j = 0; j < embLen; ++j) 
            mu0[j] = 1.0;
        mu0 = VecOp.normalize(mu0);
    }
    
    public double vmfLogLikelihood(double[] mu, double kappa, double[] x) {
        double l = 0.0;
        l += kappa * VecOp.innerProd(mu, x);
        l += (embLen * 0.5 - 1) * Math.log(kappa);
        double besselI = calculator.cal(kappa);//Bessel.i(kappa, embLen * 0.5 - 1, false);//calculator.cal(kappa);
//      if (besselI < 10 * Double.MIN_VALUE) return Double.MAX_VALUE;  
        l -= (embLen * 0.5) * Math.log(2 * Math.PI) + Math.log(besselI);
        return l;
    }

    public double vmfLogConstant(double kappa) {
        double c = 0.0;
        c += (embLen * 0.5 - 1) * Math.log(kappa);
        // System.out.println("\n0");
        double besselI = calculator.cal(kappa);//Bessel.i(kappa, embLen * 0.5 - 1, false);//calculator.cal(kappa);
        // System.out.println("1");
        // System.out.println("kappa: "+kappa+" lib: "+besselI+" cal: "+calculator.cal(kappa));
        c -= (embLen * 0.5) * Math.log(2 * Math.PI) + Math.log(besselI);
        return c;
    }
    
    private double calcLogBesselIQuotientInt(double a, double b, double nu) {
        if (fastProportion && a > besselCal.minValue && a < besselCal.maxValue && b > besselCal.minValue && b < besselCal.maxValue)
            return Math.log(calculator.cal(a)) - Math.log(calculator.cal(b));
        double ret = nu * (Math.log(a) - Math.log(b));
        int N = 50;
        double delta = Math.PI / N;
        double[] s1 = new double[N], s2 = new double[N];
        double maxS1 = -Double.MAX_VALUE, maxS2 = -Double.MAX_VALUE;
        int i = 0;
        for (double t = delta * 0.5; t < Math.PI; t += delta) {
            s1[i] = -a * Math.cos(t) + nu * Math.log(Math.sin(t));
            s2[i] = -b * Math.cos(t) + nu * Math.log(Math.sin(t));
            maxS1 = maxS1 > s1[i] ? maxS1 : s1[i];
            maxS2 = maxS2 > s2[i] ? maxS2 : s2[i];
            ++i;
        }
        double r1 = 0.0, r2 = 0.0;
        i = 0;
        for (double t = delta * 0.5; t < Math.PI; t += delta) {
            r1 += delta * Math.exp(s1[i] - maxS1);
            r2 += delta * Math.exp(s2[i] - maxS2);
            ++i;
        }
        ret += Math.log(r1) - Math.log(r2) + maxS1 - maxS2;
        return ret;
    }
    
    private double calcLogVmfConstantQuotient(double k1, double k2) {
        double c = 0.0;
        c += (embLen * 0.5 - 1) * (Math.log(k1) - Math.log(k2));
        c -= calcLogBesselIQuotientInt(k1, k2, embLen * 0.5 - 1);
        return c;   
    }
    
    
    private double calcVmfConstantQuotient4Kappa(double k1, double k2, double sumZ, double[] sumX) {
        double c = 0.0;
        double[] tempVec = new double[embLen];
        for (int j = 0; j < embLen; ++j) 
            tempVec[j] = sumX[j] + R0 * mu0[j];
        double vecl2 = VecOp.getL2(tempVec);
        c += (sumZ + C0) * calcLogVmfConstantQuotient(k1, k2);
        c += calcLogVmfConstantQuotient(k2 * vecl2, k1 * vecl2);
        return Math.exp(c); 
    }

    public void calProportion() {
        emptyProportion = Math.log(((double)(virtualVmfNum - noneEmptyVmf + 0.01))/(numVmf - noneEmptyVmf + 0.01));
    }

    public void inferVmfMixtureByOnlineGibbsSamplingZFirstBatch(oneWindowData curData) {
        // Initialization 
        double[][] x = curData.x;
        int[] z = curData.z;
        
        for (int i = 0; i < x.length; ++i) {
            // for (int j = 0; j < embLen; ++j) sumXs[z[i]][j] -= x[i][j];
            // sumZs[z[i]] -= 1.0;
            calProportion();

            sumSumZs += 1;
            double[] prob = new double[numVmf];
            double maxLogProb = - Double.MAX_VALUE;
            for (int h = 0; h < numVmf; ++h) {
                double[] vecSum = new double[embLen];
                for (int j = 0; j < embLen; ++j) 
                    vecSum[j] = kappasVmf[h] * (sumXs[h][j] + R0 * mu0[j]);
                double lengthExc = VecOp.getL2(vecSum);
//                  System.out.println("length_exc_" + h + "=" + lengthExc);
                for (int j = 0; j < embLen; ++j) 
                    vecSum[j] += kappasVmf[h] * x[i][j];
                double lengthInc = VecOp.getL2(vecSum);
//                  System.out.println("length_inc_" + h + "=" + lengthInc);
                if (kappasVmf[h] >= besselCal.maxValue || kappasVmf[h] <= besselCal.minValue) {
                    System.err.println("Idx: "+ h+" sumZ: " + sumZs[h] + " kappa: " + kappasVmf[h]);
                }
                prob[h] = Math.log(alpha + sumZs[h])
                        + (sumZs[h] < emptyThreshold ? emptyProportion : 0)
                        + vmfLogConstant(kappasVmf[h])
                        + calcLogVmfConstantQuotient(lengthExc, lengthInc);
                if (prob[h] > maxLogProb) maxLogProb = prob[h];
            }
            
            for (int h = 0; h < numVmf; ++h) prob[h] = Math.exp(prob[h] - maxLogProb);
            prob = VecOp.vec2Dist(prob);
            int newZi = VecOp.drawFromCatDist(prob);
            
            z[i] = newZi;
            for (int j = 0; j < embLen; ++j) sumXs[z[i]][j] += x[i][j];
            noneEmptyVmf += (sumZs[z[i]] < emptyThreshold) ? 1 : 0;
            sumZs[z[i]] += 1.0;
        }
    }

    public void inferVmfMixtureByOnlineGibbsSamplingKappa(int kappaIters) {
        // Initialization 

        // Sample kappa_h's
        for (int h = 0; h < numVmf; ++h) {
            double kappaCur = kappasVmf[h];
            
            // Metropolis
            int acc = 0, tot = 0;
            // System.out.println("kappa: "+ kappaCur);
            for (int kappaIter = 0; kappaIter < kappaIters; ++kappaIter) {
                double kappaNext = proposeKappa(kappaCur);
                // double logPiCur = calcLogKappaPosterior(kappaCur, sumZs[h], sumXs[h]);
                // double logPiNext = calcLogKappaPosterior(kappaNext, sumZs[h], sumXs[h]);
                // double r = Math.exp(logPiNext - logPiCur);
                double r = calcVmfConstantQuotient4Kappa(kappaNext, kappaCur, sumZs[h], sumXs[h]);
                if (rand.nextDouble() <= r) {
                    kappaCur = kappaNext;
                    ++acc;
                }
                ++tot;
            }
            
            if (debugMode) 
                System.out.println("[Vmf Mixture] Kappa:" + kappaCur + " acc rate: " + (double)acc / tot + " Z num: " + sumZs[h]);
            kappasVmf[h] = kappaCur;
        }
        double tmpcountZ = 0;
        for (int h = 0; h < numVmf; ++h)
            tmpcountZ += sumZs[h];
        System.out.println(" Vmf num: " + numVmf + " None Empty: " + noneEmptyVmf + " sum: "+ tmpcountZ + " should be: "+sumSumZs);
    }

    public void inferVmfMixtureParams() {        
            // Push this sample
        double[] distTemp = new double[numVmf];
        for (int h = 0; h < numVmf; ++h) distTemp[h] = alpha + sumZs[h];
        this.distVmf = VecOp.vec2Dist(distTemp);

        for (int h = 0; h < numVmf; ++h) {
            double[] vecSum = new double[embLen];
            for (int j = 0; j < embLen; ++j) 
                vecSum[j] = sumXs[h][j] + R0 * mu0[j];
            this.musVmf[h] = VecOp.normalize(vecSum);
        }
    }    

    public void inferVmfMixtureByGibbsSamplingSecondBatch(oneWindowData curData) {
        // Initialization 
        double[][] x = curData.x;
        int[] z = curData.z;
        
        for (int i = 0; i < x.length; ++i) {

            // System.out.printf("\r[Vmf Mixture] "+ i+" 1");
            for (int j = 0; j < embLen; ++j) sumXs[z[i]][j] -= x[i][j];
            sumZs[z[i]] -= 1.0;
            noneEmptyVmf -= (sumZs[z[i]] < emptyThreshold) ? 1 : 0;

            calProportion();

            // System.out.printf("\r[Vmf Mixture] "+ i+" 2");
            double[] prob = new double[numVmf];
            double maxLogProb = - Double.MAX_VALUE;
            for (int h = 0; h < numVmf; ++h) {
                double[] vecSum = new double[embLen];
                for (int j = 0; j < embLen; ++j) 
                    vecSum[j] = kappasVmf[h] * (sumXs[h][j] + R0 * mu0[j]);
                double lengthExc = VecOp.getL2(vecSum);
//                  System.out.println("length_exc_" + h + "=" + lengthExc);
                for (int j = 0; j < embLen; ++j) 
                    vecSum[j] += kappasVmf[h] * x[i][j];
                double lengthInc = VecOp.getL2(vecSum);
//                  System.out.println("length_inc_" + h + "=" + lengthInc);

                // System.out.printf("\r[Vmf Mixture] "+ i+" 2.1");
                prob[h] = Math.log(alpha + sumZs[h])
                        + (sumZs[h] < emptyThreshold ? emptyProportion : 0)
                        + vmfLogConstant(kappasVmf[h])
                        + calcLogVmfConstantQuotient(lengthExc, lengthInc);

                // System.out.printf("\r[Vmf Mixture] "+ i+" "+h+" 2.2");
                if (prob[h] > maxLogProb) maxLogProb = prob[h];
            }
            // System.out.printf("\r[Vmf Mixture] "+ i+" 3 |");
            for (int h = 0; h < numVmf; ++h) prob[h] = Math.exp(prob[h] - maxLogProb);
            prob = VecOp.vec2Dist(prob);
            int newZi = VecOp.drawFromCatDist(prob);
            
            // System.out.printf("\r[Vmf Mixture] "+ i+" 4 |");
            z[i] = newZi;
            for (int j = 0; j < embLen; ++j) sumXs[z[i]][j] += x[i][j];

            // System.out.printf("\r[Vmf Mixture] "+ i+" 5 |");
            noneEmptyVmf += (sumZs[z[i]] < emptyThreshold) ? 1 : 0;
            sumZs[z[i]] += 1.0;
        }
    }
    

    public void removeBatch(oneWindowData curData) {
        // Initialization 
        double[][] x = curData.x;
        int[] z = curData.z;
        
        for (int i = 0; i < x.length; ++i) {
            sumSumZs -= 1;
            for (int j = 0; j < embLen; ++j) 
                sumXs[z[i]][j] -= x[i][j];
            sumZs[z[i]] -= 1.0;
            noneEmptyVmf -= (sumZs[z[i]] < emptyThreshold) ? 1 : 0;
        }
        System.out.println("Removed: " + x.length);
    }

    public void addBatch(oneWindowData curData, int gibbsIters, int kappaIters) {
        // Initialization 
        System.out.printf("\r[Vmf Mixture] gibbsIters: 0");
        inferVmfMixtureByOnlineGibbsSamplingZFirstBatch(curData);
        inferVmfMixtureByOnlineGibbsSamplingKappa(kappaIters);
        for (int i = 1; i < gibbsIters; ++i){
            System.out.printf("\r[Vmf Mixture] gibbsIters: " + i);
            inferVmfMixtureByGibbsSamplingSecondBatch(curData);
            inferVmfMixtureByOnlineGibbsSamplingKappa(kappaIters);
        }
        System.out.printf("\n");
    }

    public void addBatchListIni(LinkedList<oneWindowData> dataSet, int gibbsIters, int kappaIters) {
        System.out.println("[Vmf Mixture] Ini Z");
        for (int i = 0; i < dataSet.size(); ++i){
            oneWindowData curData = dataSet.get(i);
            double[][] x = curData.x;
            int[] z = curData.z;
            for (int j = 0; j < x.length; ++j) {
                sumSumZs += 1;
                z[j] = VecOp.drawFromCatDist(distVmf);
                noneEmptyVmf += (sumZs[z[j]] < emptyThreshold) ? 1 : 0;
                sumZs[z[j]] += 1.0;
                // totSumZs += 1.0;
                for (int k = 0; k < embLen; ++k) sumXs[z[j]][k] += x[j][k];
            }
        }
        System.out.println("[Vmf Mixture] Kappa First Sample");
        inferVmfMixtureByOnlineGibbsSamplingKappa(kappaIters);
        for (int i = 0; i < gibbsIters; ++i){
            for (int j = 0; j < dataSet.size(); ++j) {
                System.out.printf("\r[Vmf Mixture] gibbsIters: " + i + "\tdataset: " + j);
                inferVmfMixtureByGibbsSamplingSecondBatch(dataSet.get(j));
            }
            inferVmfMixtureByOnlineGibbsSamplingKappa(kappaIters);
        }
        System.out.printf("\n");
    }

    public double calcLogLikelihood(double[][] x) {
        // Calculate Likelihood
        double L = 0.0;
        for (int i = 0; i < x.length; ++i) {
            double pXi = 0.0;
            double[] tempDist = new double[numVmf];
            double maxL = -Double.MAX_VALUE;
            for (int h = 0; h < numVmf; ++h) {
                double l = 0.0;
                l += Math.log(distVmf[h]);
                l += vmfLogLikelihood(musVmf[h], kappasVmf[h], x[i]);
//              pXi += condProb[i][h] * l;
                tempDist[h] = l;
                maxL = maxL > tempDist[h] ? maxL : tempDist[h];
            }
            for (int h = 0; h < numVmf; ++h) pXi += Math.exp(tempDist[h] - maxL);
            pXi = Math.log(pXi) + maxL;
            L += 1 * pXi;
        }
        return L;
    }
    
    public void saveModel(String fileName) throws Exception {
        System.out.println(fileName);
        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(fileName)));
        bw.write(numVmf + "\t" + embLen + "\n");
        for (int h = 0; h < numVmf; ++h) {
            bw.write(kappasVmf[h] + "\t" + distVmf[h] + "\n");
            for (int j = 0; j < embLen; ++j) {
                bw.write(musVmf[h][j] + "\t");
            }
            bw.write("\n");
        }
        bw.close();
    }
    
    public void saveResult(String fileName, oneWindowData curData) throws Exception{
        System.out.println(fileName);
        // cluster_result
        BufferedWriter nbw = new BufferedWriter(new FileWriter(new File(fileName)));
        double[][] x = curData.x;
        int setSize = curData.setSize;
        // System.out.println(setSize);
        for (int i = 0; i < setSize; ++i){
        //System.out.println(i);
            int maxIdx = -1;
        // double[] tempDist = new double[numVmf];
            double maxL = -Double.MAX_VALUE;
            for (int h = 0; h < numVmf; ++h) {
                double l = 0.0;
                l += Math.log(distVmf[h]);
                l += vmfLogLikelihood(musVmf[h], kappasVmf[h], x[i]);
                //System.out.println(l);
    //              pXi += condProb[i][h] * l;
                // tempDist[h] = l;
                nbw.write(l + ",");
                if (l >= maxL){
                    maxL = l;
                    maxIdx = h;
                }
            }
            nbw.write(maxIdx + "\n");
        }
        nbw.close();
    }

    public class besselCal {

        public boolean exitOnFault;
        public double s;
        public double [] table;
        public static final int tableSize = 1000000;
        public static final double minValue = 0.5, maxValue = 708.989; 
        public static final double stepSize = (708.99 - 0.5) / (1000000 + 1);
        // public double threshold;
        
        // public double t1in
        public besselCal(double s, boolean exitOnFault) { //, double threshold) {
            this.s = s;
            this.exitOnFault = exitOnFault;
            table = new double[tableSize];
            preCal();
            // this.threshold = threshold;
            // this.t1ini = Math.sqrt(s / (2 * Math.PI)) / (1 + 1 / (12 * s) + 1 / (288 * s * s) - 139 / (51840 * s * s * s));
        }

        public void preCal(){
            for (int i = 0; i < tableSize; ++i)
                table[i] = Bessel.i(minValue + stepSize * i, s, false);
        }

        public double cal(double x){
            // double r = 1;
            // double t1 = Math.pow( (x * Math.E)/ (2 * s), s) * this.t1ini;
            // double m = 1 / s;
            // double k = 1;
            // System.out.println("x: " + x);
            // while (true) {
            //     // System.out.printf("\rk: " + k);
            //     r = r * 0.25 * x * x / (k * ( s + k));
            //     m = m + r;
            //     // System.out.println(r+"\t"+m+"\t"+r/m);
            //     if ((r/m) < threshold) {
            //         break;
            //     }
            //     k = k + 1;
            // }
            // return t1 * m;
            double retnum = 0;
            if (x > minValue && x < maxValue) {
                try{
                    retnum = table[(int)((x - minValue)/stepSize)];
                } catch (Exception e) {
                    System.err.println("x: " + x + " minValue: "+minValue+" stepSize: "+stepSize+" maxValue: "+maxValue+" tableSize: "+tableSize+" calIndex: "+(int)((x - minValue)/stepSize));
                }
            } else {
                System.err.println("kappa out of range! kappa: "+x);
                if (exitOnFault)
                    System.exit(-1);
                else
                    retnum = Bessel.i(x, s, false);
            }
            return retnum;
        }
    }

    static public void main(String[] args) throws Exception {
        if (args.length < 17) {
            System.out.println("usage: input_folder output_folder setting_file window_size numVmf gibbsIters kappaIters exitOnFault alpha R0 C0 kappaUpdateMode stepSize virtualVmfNum fastProportion debugMode uniformIni");
            System.out.println("example: java -cp bin:lib/* vmf/onlineVmfMixture /shared/data/ll2/tweets/online/embed/ /shared/data/ll2/tweets/online/model/ /shared/data/ll2/tweets/online/nsetting.csv 5 500 10 100 1 1 0.1 0.01 1 3 2000 1 0 1");
            return;
        }
        String input_folder = args[0];
        String output_folder = args[1];
        String setting_file = args[2];
        int windowSize = Integer.parseInt(args[3]);
        int numVmf = Integer.parseInt(args[4]);
        // double threshold = Double.parseDouble(args[5]);
        int gibbsIters = Integer.parseInt(args[5]);
        int kappaIters = Integer.parseInt(args[6]);
        boolean exitOnFault = Integer.parseInt(args[7]) > 0;
        double alpha = Double.parseDouble(args[8]);
        double r0 = Double.parseDouble(args[9]);
        double c0 = Double.parseDouble(args[10]);        
        boolean kappaUpdateMode = Integer.parseInt(args[11]) > 0;
        double stepSize = Double.parseDouble(args[12]);
        double virtualVmfNum = Double.parseDouble(args[13]);
        boolean fastProportion = Integer.parseInt(args[14]) > 0;
        boolean debugMode = Integer.parseInt(args[15]) > 0;
        boolean uniformIni = Integer.parseInt(args[16]) > 0;

        File setting = new File(setting_file);
        Scanner scan = new Scanner(setting);
        int insNum = scan.nextInt();
        int embLen = scan.nextInt();

        HashMap<Integer, Integer> lengthMap = new HashMap<Integer, Integer>();
        HashMap<Integer, Integer> lengthSave = new HashMap<Integer, Integer>();
        for (int i = 0 ;i < insNum; ++i){
            int k = scan.nextInt();
            int v = scan.nextInt();
            int s = scan.nextInt();
            lengthMap.put(k, v);
            lengthSave.put(k, s);
        }

        System.out.println("input_folder: "+input_folder);
        System.out.println("output_folder: "+output_folder);
        System.out.println("setting_file: "+setting_file);
        System.out.println("window_size: "+windowSize);
        System.out.println("numVmf: "+numVmf);
        // System.out.println("threshold: "+threshold);
        System.out.println("gibbsIters: "+gibbsIters);
        System.out.println("kappaIters: "+kappaIters);
        System.out.println("insNum: "+insNum);
        System.out.println("embLen: "+embLen);
        System.out.println("exitOnFault: "+exitOnFault);
        System.out.println("alpha: "+alpha);
        System.out.println("R0: "+r0);
        System.out.println("C0: "+c0);
        System.out.println("kappa Update Mode: "+kappaUpdateMode);
        System.out.println("stepSize: "+stepSize);
        System.out.println("virtual Vmf num: "+virtualVmfNum);
        System.out.println("fastProportion: "+fastProportion);
        System.out.println("fastProportion: "+uniformIni);

        LinkedList<oneWindowData> dataSet = new LinkedList<oneWindowData>();
        int i; 
        System.out.println("Reading first window");
        for (i = 0; i < windowSize; ++i){
            // System.out.println("file: "+input_folder+Integer.toString(i)+".embed; len:"+ lengthMap.get(i)+"; elen:" +embLen);
            oneWindowData curData = new oneWindowData(input_folder+Integer.toString(i)+".embed", lengthMap.get(i), embLen);
            dataSet.addLast(curData);
        }

        onlineVmfMixture vmfMixture = new onlineVmfMixture(alpha, r0, c0, embLen, numVmf, exitOnFault, kappaUpdateMode, stepSize, virtualVmfNum, fastProportion, debugMode, uniformIni);//, threshold);
        System.out.println("[Vmf Mixture] Initialization");
        vmfMixture.initVmfMixture();
        System.out.println("[Vmf Mixture] Sampling for first window");
        vmfMixture.addBatchListIni(dataSet, gibbsIters, kappaIters);
        System.out.println("[Vmf Mixture] Infer Params");
        vmfMixture.inferVmfMixtureParams();
        System.out.println("[Vmf Mixture] Saving Model");
        vmfMixture.saveModel("/shared/data/ll2/tweets/online/model/ini.model");
        for (int j = 0; j < i; ++j){
            if (lengthSave.get(j) == 1){
                System.out.println("[Vmf Mixture] Saving results for: " + j);
                vmfMixture.saveResult(output_folder+Integer.toString(j)+".csv", dataSet.get(j));
            }
        }

        for (; i < insNum; ++i){
            System.out.println("[Vmf Mixture] removing the oldest file");
            vmfMixture.removeBatch(dataSet.getFirst());
            dataSet.remove();
            System.out.println("[Vmf Mixture] Loading file: " + i + "Total: " + insNum);
            oneWindowData curData = new oneWindowData(input_folder+Integer.toString(i)+".embed", lengthMap.get(i), embLen);
            dataSet.addLast(curData);
            System.out.println("[Vmf Mixture] Sampling for: " + i);
            vmfMixture.addBatch(dataSet.getLast(), gibbsIters, kappaIters);
            if (lengthSave.get(i) == 1){
                System.out.println("[Vmf Mixture] Saving Model for: " + i);
                vmfMixture.saveModel(output_folder+Integer.toString(i)+".model");
                System.out.println("[Vmf Mixture] Saving results for: " + i);
                vmfMixture.saveResult(output_folder+Integer.toString(i)+".csv", dataSet.getLast());
            }
        }
    }
}

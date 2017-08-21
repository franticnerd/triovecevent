package vmf;

import java.io.*;

import java.util.*;
import java.util.Map.Entry;

import jdistlib.math.Bessel;
import util.MapSorter;
import util.VecOp;

import vmf.oneWindowDataFull;

public class onlineVmfGaussianMixture {
    
    public onlineVmfGaussianMixture(double alpha, double r0, double c0, int dim, int clusNum, boolean exitOnFault, boolean kappaUpdateMode, double kappaStep, double virtualVmfNum, boolean fastProportion, boolean debugMode, boolean uniformIni, double lambda0, double nu0, double delta0) {//, double threshold) {
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
        this.lambda0 = lambda0;
        this.nu0 = nu0;
        this.delta0 = delta0;
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
    public double[][] icovGaussian;
    public double[] detGaussian;
    public double[][] etaGaussian;

    public besselCal calculator;

    double[][] sumXs;
    double[] sumZs;
    // double[] detRec;
    double[][] sRec;
    // double totSumZs = 0.0;

    public double alpha;
    public double R0;
    public double C0;
    public double[] mu0;
    public double[] eta0;
    public double[] s0;
    public double lambda0;
    public double nu0;
    public double delta0;

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
        icovGaussian = new double[numVmf][4];
        detGaussian = new double[numVmf];
        etaGaussian = new double[numVmf][2];

        sumXs = new double[numVmf][embLen];
        sumZs = new double[numVmf];
        // detRec = new double[numVmf];
        sRec = new double[numVmf][4];
        
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

        s0 = new double[4];
        s0[0] = delta0; s0[3] = delta0; s0[1] = 0; s0[2] = 0;
        eta0= new double[2];
        eta0[0] = 0; eta0[1] = 0;

        for (int i = 0; i < numVmf; ++i) {
            // detRec[i] = 1;
            detGaussian[i] = 1;
            for (int j = 0; j < 4; ++j) {
                sRec[i][j] = s0[j];
                icovGaussian[i][j] = s0[j];
            }
            etaGaussian[i][0] = eta0[0];
            etaGaussian[i][1] = eta0[1];
        }
    }
    
    public double gaussianLogLikelihood(double sdet, double[] eta, double[] icov, double[] d) {
        double l = 0.0;
        double etaSub0 = d[0] - eta[0], etaSub1 = d[1] - eta[1];
        l -= 0.5 * Math.log(sdet);
        l -= 0.5 * (etaSub0 * icov[0] * etaSub0 - etaSub0 * (icov[1] +icov[2]) * etaSub1 + etaSub1 * icov[3] * etaSub1);
        l -= Math.log(2 * Math.PI);
        return l;
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

    public void inferVmfMixtureByOnlineGibbsSamplingZFirstBatch(oneWindowDataFull curData) {
        // Initialization 
        double[][] x = curData.x;
        double[][] d = curData.d;
        int[] z = curData.z;
        
        for (int i = 0; i < x.length; ++i) {
            // for (int j = 0; j < embLen; ++j) sumXs[z[i]][j] -= x[i][j];
            // sumZs[z[i]] -= 1.0;
            calProportion();

            sumSumZs += 1;
            double[] prob = new double[numVmf];
            double maxLogProb = - Double.MAX_VALUE;
            // double etaSub0, etaSub1, sRatio;
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
                // if (kappasVmf[h] >= besselCal.maxValue || kappasVmf[h] <= besselCal.minValue) {
                //     System.err.println("Idx: "+ h+" sumZ: " + sumZs[h] + " kappa: " + kappasVmf[h]);
                // }
                prob[h] = Math.log(alpha + sumZs[h])
                        + Math.log(sumZs[h] + lambda0) - Math.log(sumZs[h] + 1 + lambda0)
                        + gaussianLogRatio(h, d[i])
                        + (sumZs[h] < emptyThreshold ? emptyProportion : 0)
                        + vmfLogConstant(kappasVmf[h])
                        + calcLogVmfConstantQuotient(lengthExc, lengthInc);
                if (prob[h] > maxLogProb) maxLogProb = prob[h];
            }
            
            for (int h = 0; h < numVmf; ++h) prob[h] = Math.exp(prob[h] - maxLogProb);
            prob = VecOp.vec2Dist(prob);
            int newZi = VecOp.drawFromCatDist(prob);
            
            z[i] = newZi;
            for (int j = 0; j < embLen; ++j) sumXs[newZi][j] += x[i][j];
            noneEmptyVmf += (sumZs[newZi] < emptyThreshold) ? 1 : 0;
            addGaussianPoint(newZi, d[i]);
            sumZs[newZi] += 1.0;
        }
    }

    public double gaussianLogRatio(int zInd, double[] d){
        double[] sInc = new double[4];
        double sRatio = (lambda0 + sumZs[zInd]) / (lambda0 + 1 + sumZs[zInd]);
        double etaSub0 = (etaGaussian[zInd][0] - d[0]), etaSub1 = (etaGaussian[zInd][1] - d[1]);
        sInc[0] = sRec[zInd][0] + sRatio * etaSub0 * etaSub0;
        sInc[1] = sRec[zInd][1] + sRatio * etaSub0 * etaSub1;
        sInc[2] = sRec[zInd][2] + sRatio * etaSub1 * etaSub0;
        sInc[3] = sRec[zInd][3] + sRatio * etaSub1 * etaSub1;
        if (debugMode) System.out.printf(detGaussian[zInd]+",");
        double rres = sumZs[zInd] * 0.5 * Math.log(detGaussian[zInd]);
        if (debugMode) System.out.printf(rres+",");
        rres -= (sumZs[zInd] + 1) * 0.5 * Math.log(sInc[0] * sInc[3] - sInc[1] * sInc[2]);
        if (debugMode) System.out.printf(rres+",");
        rres += Math.log((nu0 + sumZs[zInd] - 1) * 0.5);
        if (debugMode) System.out.printf(rres+",");
        return rres;
    }

    public void addGaussianPoint(int zInd, double[] d){
        //sumZs[ind] = eta_n
        etaGaussian[zInd][0] = ((sumZs[zInd] + lambda0) * etaGaussian[zInd][0] + d[0]) / (lambda0 + sumZs[zInd] + 1); //eta_n+1
        etaGaussian[zInd][1] = ((sumZs[zInd] + lambda0) * etaGaussian[zInd][1] + d[1]) / (lambda0 + sumZs[zInd] + 1);
        double sRatio = (lambda0 + 1 + sumZs[zInd]) / (lambda0 + sumZs[zInd]);
        double etaSub0 = etaGaussian[zInd][0] - d[0]; 
        double etaSub1 = etaGaussian[zInd][1] - d[1];
        sRec[zInd][0] += sRatio * etaSub0 * etaSub0;
        sRec[zInd][1] += sRatio * etaSub0 * etaSub1;
        sRec[zInd][2] += sRatio * etaSub1 * etaSub0;
        sRec[zInd][3] += sRatio * etaSub1 * etaSub1;
        detGaussian[zInd] = sRec[zInd][0] * sRec[zInd][3] - sRec[zInd][1] * sRec[zInd][2];
        if (2 == zInd && debugMode) System.out.println("newdet:"+detGaussian[zInd]+"zInd:"+zInd+"size:"+sumZs[zInd]+"sRec:"+sRec[zInd][0]+","+sRec[zInd][1]+","+sRec[zInd][2]+","+sRec[zInd][3]+" d: "+d[0]+" "+d[1]+" eta: "+etaGaussian[zInd][0]+" "+etaGaussian[zInd][1]);
    }

    public void removeGaussianPoint(int zInd, double[] d){
        // if(sumZs[zInd] == 1){

        // }
        double sRatio = (lambda0 + sumZs[zInd]) / (lambda0 + sumZs[zInd] - 1);
        double etaSub0 = etaGaussian[zInd][0] - d[0]; 
        double etaSub1 = etaGaussian[zInd][1] - d[1];
        sRec[zInd][0] -= sRatio * etaSub0 * etaSub0;
        sRec[zInd][1] -= sRatio * etaSub0 * etaSub1;
        sRec[zInd][2] -= sRatio * etaSub1 * etaSub0;
        sRec[zInd][3] -= sRatio * etaSub1 * etaSub1;
        detGaussian[zInd] = sRec[zInd][0] * sRec[zInd][3] - sRec[zInd][1] * sRec[zInd][2];
        if (2 == zInd && debugMode) System.out.println("remove!newdet:"+detGaussian[zInd]+"zind:"+zInd+"size:"+sumZs[zInd]+"sRec:"+sRec[zInd][0]+","+sRec[zInd][1]+","+sRec[zInd][2]+","+sRec[zInd][3]+" d: "+d[0]+" "+d[1]+" eta: "+etaGaussian[zInd][0]+" "+etaGaussian[zInd][1]);
        etaGaussian[zInd][0] = ((sumZs[zInd] + lambda0) * etaGaussian[zInd][0] - d[0]) / (lambda0 + sumZs[zInd] - 1);
        etaGaussian[zInd][1] = ((sumZs[zInd] + lambda0) * etaGaussian[zInd][1] - d[1]) / (lambda0 + sumZs[zInd] - 1);
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
            icovGaussian[h][0] = sRec[h][3] / (nu0 + 4);
            icovGaussian[h][1] = -sRec[h][1] / (nu0 + 4);
            icovGaussian[h][2] = -sRec[h][2] / (nu0 + 4);
            icovGaussian[h][3] = sRec[h][0] / (nu0 + 4);
            detGaussian[h] = icovGaussian[h][0] * icovGaussian[h][3] - icovGaussian[h][1] * icovGaussian[h][2];
            icovGaussian[h][0] = icovGaussian[h][0] / detGaussian[h];
            icovGaussian[h][1] = icovGaussian[h][1] / detGaussian[h];
            icovGaussian[h][2] = icovGaussian[h][2] / detGaussian[h];
            icovGaussian[h][3] = icovGaussian[h][3] / detGaussian[h];
        }
    }    

    public void inferVmfMixtureByGibbsSamplingSecondBatch(oneWindowDataFull curData) {
        // Initialization 
        double[][] x = curData.x;
        int[] z = curData.z;
        double[][] d = curData.d;
        for (int i = 0; i < x.length; ++i) {

            // System.out.printf("\r[Vmf Mixture] "+ i+" 1");
            removeGaussianPoint(z[i], d[i]);
            for (int j = 0; j < embLen; ++j) sumXs[z[i]][j] -= x[i][j];
            sumZs[z[i]] -= 1.0;
            noneEmptyVmf -= (sumZs[z[i]] < emptyThreshold) ? 1 : 0;

            calProportion();

            if (debugMode) System.out.printf(i+":");
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
                // prob[h] = Math.log(sumZs[h] + lambda0) - Math.log(sumZs[h] + 1 + lambda0);
                // if (debugMode) System.out.printf("("+h+":");
                // prob[h] += gaussianLogRatio(h, d[i]);
                // System.out.printf(prob[h]+",");
                prob[h] = Math.log(alpha + sumZs[h])
                        + Math.log(sumZs[h] + lambda0) - Math.log(sumZs[h] + 1 + lambda0)
                        + gaussianLogRatio(h, d[i])
                        + (sumZs[h] < emptyThreshold ? emptyProportion : 0)
                        + vmfLogConstant(kappasVmf[h])
                        + calcLogVmfConstantQuotient(lengthExc, lengthInc);
                if (debugMode) System.out.printf("),");
                if (prob[h] > maxLogProb) maxLogProb = prob[h];
            }
            // System.out.printf("\r[Vmf Mixture] "+ i+" 3 |");
            for (int h = 0; h < numVmf; ++h) prob[h] = Math.exp(prob[h] - maxLogProb);
            prob = VecOp.vec2Dist(prob);
            int newZi = VecOp.drawFromCatDist(prob);
            
            // System.out.printf("\r[Vmf Mixture] "+ i+" 4 |");
            z[i] = newZi;
            for (int j = 0; j < embLen; ++j) sumXs[newZi][j] += x[i][j];

            // System.out.printf("\r[Vmf Mixture] "+ i+" 5 |");
            addGaussianPoint(newZi, d[i]);
            if (debugMode) System.out.printf("\n");
            noneEmptyVmf += (sumZs[newZi] < emptyThreshold) ? 1 : 0;
            sumZs[z[i]] += 1.0;
        }
    }
    

    public void removeBatch(oneWindowDataFull curData) {
        // Initialization 
        double[][] x = curData.x;
        int[] z = curData.z;
        double[][] d = curData.d;

        for (int i = 0; i < x.length; ++i) {
            removeGaussianPoint(z[i], d[i]);
            for (int j = 0; j < embLen; ++j) 
                sumXs[z[i]][j] -= x[i][j];
            sumZs[z[i]] -= 1.0;
            sumSumZs -= 1;
            noneEmptyVmf -= (sumZs[z[i]] < emptyThreshold) ? 1 : 0;
        }
        System.out.println("Removed: " + x.length);
    }

    public void addBatch(oneWindowDataFull curData, int gibbsIters, int kappaIters) {
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

    public void addBatchListIni(LinkedList<oneWindowDataFull> dataSet, int gibbsIters, int kappaIters) {
        System.out.println("[Vmf Mixture] Ini Z");
        // if(debugMode) {
        //     for (int h = 0; h < numVmf; ++h){
        //         System.out.println("detini: "+ h+" "+ detGaussian[h]+" sumZ: "+sumZs[h]+"sRec: "+sRec[h][0]+","+sRec[h][1]+","+sRec[h][2]+","+sRec[h][3]);
        //     }
        // }
        for (int i = 0; i < dataSet.size(); ++i){
            oneWindowDataFull curData = dataSet.get(i);
            double[][] x = curData.x;
            int[] z = curData.z;
            double[][] d = curData.d;
            for (int j = 0; j < x.length; ++j) {
                z[j] = VecOp.drawFromCatDist(distVmf);
                addGaussianPoint(z[j], d[j]);
                // totSumZs += 1.0;
                for (int k = 0; k < embLen; ++k) sumXs[z[j]][k] += x[j][k];
                noneEmptyVmf += (sumZs[z[j]] < emptyThreshold) ? 1 : 0;
                sumZs[z[j]] += 1.0;
                sumSumZs += 1;
            }
        }
        // if(debugMode) {
        //     for (int h = 0; h < numVmf; ++h){
        //         System.out.println("detini: "+ h+" "+ detGaussian[h]+" sumZ: "+sumZs[h]+"sRec:"+sRec[h][0]+","+sRec[h][1]+","+sRec[h][2]+","+sRec[h][3]);
        //     }
        //     System.out.println("startRemoving: ");
        // }
        // for (int i = 0; i < dataSet.size(); ++i){
        //     oneWindowDataFull curData = dataSet.get(i);
        //     double[][] x = curData.x;
        //     int[] z = curData.z;
        //     double[][] d = curData.d;
        //     for (int j = 0; j < x.length; ++j) {
        //         removeGaussianPoint(z[j], d[j]);
        //         for (int k = 0; k < embLen; ++k) sumXs[z[j]][k] -= x[j][k];
        //         sumZs[z[j]] -= 1.0;
        //         noneEmptyVmf -= (sumZs[z[j]] < emptyThreshold) ? 1 : 0;
        //     }
        // }
        // if(debugMode) {
        //     for (int h = 0; h < numVmf; ++h){
        //         System.out.println("detini: "+ h+" "+ detGaussian[h]+" sumZ: "+sumZs[h]+"sRec:"+sRec[h][0]+","+sRec[h][1]+","+sRec[h][2]+","+sRec[h][3]);
        //     }
        // }
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

//     public double calcLogLikelihood(double[][] x, double[][] d) {
//         // Calculate Likelihood
//         double L = 0.0;
//         for (int i = 0; i < x.length; ++i) {
//             double pXi = 0.0;
//             double[] tempDist = new double[numVmf];
//             double maxL = -Double.MAX_VALUE;
//             for (int h = 0; h < numVmf; ++h) {
//                 double l = 0.0;
                // l += Math.log(distVmf[h]);
                // l += vmfLogLikelihood(musVmf[h], kappasVmf[h], x[i]);
                // l += gaussianLogLikelihood(detGaussian[h], etaGaussian[h], icov[h], d[i]);
// //              pXi += condProb[i][h] * l;
//                 tempDist[h] = l;
//                 maxL = maxL > tempDist[h] ? maxL : tempDist[h];
//             }
//             for (int h = 0; h < numVmf; ++h) pXi += Math.exp(tempDist[h] - maxL);
//             pXi = Math.log(pXi) + maxL;
//             L += 1 * pXi;
//         }
//         return L;
//     }
    
    public void saveModel(String fileName) throws Exception {
        System.out.println(fileName);
        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(fileName)));
        bw.write(numVmf + "\t" + embLen + "\n");
        for (int h = 0; h < numVmf; ++h) {
            bw.write(kappasVmf[h] + "\t" + distVmf[h] + "\t"+ detGaussian[h]+"\t"+etaGaussian[h][0]+"\t"+etaGaussian[h][1]+"\t"+icovGaussian[h][0]+"\t"+icovGaussian[h][1]+"\t"+icovGaussian[h][2]+"\t"+icovGaussian[h][3]+"\n");
            for (int j = 0; j < embLen; ++j) {
                bw.write(musVmf[h][j] + "\t");
            }
            bw.write("\n");
        }
        bw.close();
    }
    
    public void saveResult(String fileName, oneWindowDataFull curData) throws Exception{
        System.out.println(fileName);
        // cluster_result
        BufferedWriter nbw = new BufferedWriter(new FileWriter(new File(fileName)));
        double[][] x = curData.x;
        double[][] d = curData.d;
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
                l += gaussianLogLikelihood(detGaussian[h], etaGaussian[h], icovGaussian[h], d[i]);
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
        public static final double minValue = 2, maxValue = 708.989; 
        public static final double stepSize = (708.99 - 2) / (1000000 + 1);
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



    public static List<List<Integer>> loadQueries(String query_file) throws IOException {
        List<List<Integer>> queries = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(query_file));
        while(true)  {
            String line = br.readLine();
            if(line == null)
                break;
            line = line.replace("\n", "").replace("\r", "");
            String[] items = line.split(" ");
            List<Integer> query = new ArrayList();
            for(String e : items) {
                query.add(new Integer(e));
            }
            queries.add(query);
        }
        br.close();
        System.out.println(queries);
        return queries;
    }


    public static LinkedList<oneWindowDataFull> loadWindowData(String input_folder, List<Integer> query,
                                                           HashMap<Integer, Integer> lengthMap, int embLen) throws Exception {
        LinkedList<oneWindowDataFull> dataSet = new LinkedList();
        System.out.println("Reading window data...");
        for (int i = 0; i < query.size(); ++i){
            int binId = query.get(i);
            String dataFileName = input_folder + Integer.toString(binId) + ".embed";
            oneWindowDataFull curData = new oneWindowDataFull(dataFileName, lengthMap.get(binId), embLen);
            dataSet.addLast(curData);
        }
        return dataSet;
    }


    static public void main(String[] args) throws Exception {
        if (args.length < 18) {
            System.out.println("usage: data_folder window_size numVmf gibbsIters kappaIters exitOnFault alpha R0 C0 kappaUpdateMode stepSize virtualVmfNum fastProportion debugMode uniformIni lambda0 nu0 delta0");
            System.out.println("example: java -cp bin:lib/* vmf/onlineVmfGaussianMixture /shared/data/ll2/tweets/ 5 500 10 100 1 1 0.1 0.01 1 3 2000 1 0 0 1 2 1");
            return;
        }
        String data_folder = args[0];
        int windowSize = Integer.parseInt(args[1]);
        int numVmf = Integer.parseInt(args[2]);
        int gibbsIters = Integer.parseInt(args[3]);
        int kappaIters = Integer.parseInt(args[4]);
        boolean exitOnFault = Integer.parseInt(args[5]) > 0;
        double alpha = Double.parseDouble(args[6]);
        double r0 = Double.parseDouble(args[7]);
        double c0 = Double.parseDouble(args[8]);
        boolean kappaUpdateMode = Integer.parseInt(args[9]) > 0;
        double stepSize = Double.parseDouble(args[10]);
        double virtualVmfNum = Double.parseDouble(args[11]);
        boolean fastProportion = Integer.parseInt(args[12]) > 0;
        boolean debugMode = Integer.parseInt(args[13]) > 0;
        boolean uniformIni = Integer.parseInt(args[14]) > 0;
        double lambda0 = Double.parseDouble(args[15]);
        double nu0 = Double.parseDouble(args[16]);
        double delta0 = Double.parseDouble(args[17]);

        String input_folder = data_folder + "embeddings/";
        String query_file = data_folder + "input/queries.txt";
        String output_folder = data_folder + "cluster/";
        String setting_file = data_folder + "input/dataset_info.txt";

        File setting = new File(setting_file);
        Scanner scan = new Scanner(setting);
        int insNum = scan.nextInt();
        int embLen = scan.nextInt();

        HashMap<Integer, Integer> lengthMap = new HashMap<Integer, Integer>();
        // HashMap<Integer, Integer> lengthSave = new HashMap<Integer, Integer>();
        for (int i = 0 ;i < insNum; ++i){
            int k = scan.nextInt();
            int value = scan.nextInt();
            // int s = scan.nextInt();
            lengthMap.put(k, value);
            // lengthSave.put(k, s);
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
        System.out.println("debugMode: "+debugMode);
        System.out.println("uniformIni: "+uniformIni);
        System.out.println("lambda0: "+lambda0);
        System.out.println("nu0: "+nu0);
        System.out.println("delta0: "+delta0);


        List<List<Integer>> queries = loadQueries(query_file);
        for (int qid = 0; qid < queries.size(); qid++) {
            List<Integer> query = queries.get(qid);
            LinkedList<oneWindowDataFull> dataSet = loadWindowData(input_folder, query, lengthMap, embLen);
            onlineVmfGaussianMixture vmfMixture = new onlineVmfGaussianMixture(alpha, r0, c0, embLen, numVmf, exitOnFault, kappaUpdateMode, stepSize, virtualVmfNum, fastProportion, debugMode, uniformIni, lambda0, nu0, delta0);//, threshold);
            System.out.println("[Vmf Mixture] Initialization");
            vmfMixture.initVmfMixture();
            long tStart = System.currentTimeMillis();
            System.out.println("[Vmf Mixture] Sampling for query " + qid);
            vmfMixture.addBatchListIni(dataSet, gibbsIters, kappaIters);
            long tEnd = System.currentTimeMillis();
            long tDelta = tEnd - tStart;
            double elapsedSeconds = tDelta / 1000.0;
            System.out.println("Elpased time for batch mode inference" + elapsedSeconds);
            System.out.println("[Vmf Mixture] Infer Params");
            vmfMixture.inferVmfMixtureParams();

            // saving results
            for (int j = 0; j < query.size(); ++j){
                int binId = query.get(j);
                String outputFileName = output_folder + Integer.toString(qid) + "-" + Integer.toString(binId)+".csv";
                System.out.println("[Vmf Mixture] Saving results for: " + j);
                vmfMixture.saveResult(outputFileName, dataSet.get(j));
            }

        }



//        LinkedList<oneWindowDataFull> dataSet = new LinkedList<oneWindowDataFull>();
//        int i;
//        System.out.println("Reading first window");
//        for (i = 0; i < windowSize; ++i){
//            // System.out.println("file: "+input_folder+Integer.toString(i)+".embed; len:"+ lengthMap.get(i)+"; elen:" +embLen);
//            oneWindowDataFull curData = new oneWindowDataFull(input_folder+Integer.toString(i)+".embed", lengthMap.get(i), embLen);
//            dataSet.addLast(curData);
//        }

//        onlineVmfGaussianMixture vmfMixture = new onlineVmfGaussianMixture(alpha, r0, c0, embLen, numVmf, exitOnFault, kappaUpdateMode, stepSize, virtualVmfNum, fastProportion, debugMode, uniformIni, lambda0, nu0, delta0);//, threshold);
//        System.out.println("[Vmf Mixture] Initialization");
//        vmfMixture.initVmfMixture();
//        System.out.println("[Vmf Mixture] Sampling for first window");
//        vmfMixture.addBatchListIni(dataSet, gibbsIters, kappaIters);
//        System.out.println("[Vmf Mixture] Infer Params");
//        vmfMixture.inferVmfMixtureParams();
//        System.out.println("[Vmf Mixture] Saving Model");
//        // vmfMixture.saveModel("/shared/data/ll2/tweets/online/model/ini.model");
//        vmfMixture.saveModel(output_folder + "ini.model");
//        for (int j = 0; j < i; ++j){
//            if (lengthSave.get(j) == 1){
//                System.out.println("[Vmf Mixture] Saving results for: " + j);
//                vmfMixture.saveResult(output_folder+Integer.toString(j)+".csv", dataSet.get(j));
//            }
//        }

        // for (; i < insNum; ++i){
        //     System.out.println("[Vmf Mixture] removing the oldest file");
        //     vmfMixture.removeBatch(dataSet.getFirst());
        //     dataSet.remove();
        //     System.out.println("[Vmf Mixture] Loading file: " + i + "Total: " + insNum);
        //     oneWindowDataFull curData = new oneWindowDataFull(input_folder+Integer.toString(i)+".embed", lengthMap.get(i), embLen);
        //     dataSet.addLast(curData);
        //     System.out.println("[Vmf Mixture] Sampling for: " + i);
        //     vmfMixture.addBatch(dataSet.getLast(), gibbsIters, kappaIters);
        //     if (lengthSave.get(i) == 1){
        //         System.out.println("[Vmf Mixture] Saving Model for: " + i);
        //         vmfMixture.saveModel(output_folder+Integer.toString(i)+".model");
        //         System.out.println("[Vmf Mixture] Saving results for: " + i);
        //         vmfMixture.saveResult(output_folder+Integer.toString(i)+".csv", dataSet.getLast());
        //     }
        // }

    }
}

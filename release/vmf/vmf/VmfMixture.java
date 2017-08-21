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
import jdistlib.math.Bessel;
import util.MapSorter;
import util.VecOp;


public class VmfMixture {
	
	public VmfMixture(int dim, int clusNum) {
		this.embLen = dim;
		this.numVmf = clusNum;
	}
	
	public int numVmf = 10;
	public int embLen;
	
	public double[] distVmf;
	public double[][] musVmf;
	public double[] kappasVmf;
	
	public int emMaxIter = 1000;
	public double eps  = 1e-6;
	private double kappaMinValue = 0.08;
	
	public double alpha = 0.5;
	public double R0 = 0.1;
	public double C0 = 0.5;
	public double[] mu0;

//Important!!!
//too small kappa may result in fail!!

	public double kappaIniScale = 100, kappaIniShape = 0.01; 
	// public double logNormalScale = 100, logNormalShape = 0.01; 
	// public LogNormalDistribution logNormal = new LogNormalDistribution(Math.log(logNormalScale), logNormalShape);
	public int gibbsIter = 30;
//	public int gibbsBurnIn = 500;
	public int kappaGibbsIter = 2000;
	public double kappaStep = 0.01;
	
	private double proposeKappa(double initKappa) {
		double ret = Math.exp(Math.log(initKappa) + kappaStep * rand.nextGaussian());
//		System.out.println("init kappa:" + initKappa + ", proposed kappa:" + ret);
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
		for (int h = 0; h < numVmf; ++h) {
			distVmf[h] = rand.nextDouble();
			// kappasVmf[h] = Math.exp(logNormalShape * rand.nextGaussian() + Math.log(logNormalScale));
			kappasVmf[h] = Math.exp(kappaIniShape * rand.nextGaussian() + Math.log(kappaIniScale));
			if (musVmfInit == null) {
				for (int j = 0; j < embLen; ++j) musVmf[h][j] = rand.nextGaussian();
			} else {
				for (int j = 0; j < embLen; ++j) musVmf[h][j] = musVmfInit[h][j];
			}
			musVmf[h] = VecOp.normalize(musVmf[h]);
		}
		distVmf = VecOp.vec2Dist(distVmf);

	}
	
	public double vmfLogLikelihood(double[] mu, double kappa, double[] x) {
		double l = 0.0;
		l += kappa * VecOp.innerProd(mu, x);
		l += (embLen * 0.5 - 1) * Math.log(kappa);
		double besselI = Bessel.i(kappa, embLen * 0.5 - 1, false);
//		if (besselI < 10 * Double.MIN_VALUE) return Double.MAX_VALUE;  
		l -= (embLen * 0.5) * Math.log(2 * Math.PI) + Math.log(besselI);
		return l;
	}


	
	public double vmfLogConstant(double kappa) {
		double c = 0.0;
		c += (embLen * 0.5 - 1) * Math.log(kappa);
		double besselI = Bessel.i(kappa, embLen * 0.5 - 1, false);
		c -= (embLen * 0.5) * Math.log(2 * Math.PI) + Math.log(besselI);
		return c;
	}
	
	static private double calcLogBesselIQuotientInt(double a, double b, double nu) {
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
	
	// static private double calcLogBesselIQuotientIntWithNominatorPow(double a, double pow, double b, double nu) {
	// 	double ret = nu * (pow * Math.log(a) - Math.log(b));
	// 	int N = 50;
	// 	double delta = Math.PI / N;
	// 	double[] s1 = new double[N], s2 = new double[N];
	// 	double maxS1 = -Double.MAX_VALUE, maxS2 = -Double.MAX_VALUE;
	// 	int i = 0;
	// 	for (double t = delta * 0.5; t < Math.PI; t += delta) {
	// 		s1[i] = -a * Math.cos(t) + nu * Math.log(Math.sin(t));
	// 		s2[i] = -b * Math.cos(t) + nu * Math.log(Math.sin(t));
	// 		maxS1 = maxS1 > s1[i] ? maxS1 : s1[i];
	// 		maxS2 = maxS2 > s2[i] ? maxS2 : s2[i];
	// 		++i;
	// 	}
	// 	double r1 = 0.0, r2 = 0.0;
	// 	i = 0;
	// 	for (double t = delta * 0.5; t < Math.PI; t += delta) {
	// 		r1 += delta * Math.exp(s1[i] - maxS1);
	// 		r2 += delta * Math.exp(s2[i] - maxS2);
	// 		++i;
	// 	}
	// 	ret += pow * Math.log(r1) - Math.log(r2) + pow * maxS1 - maxS2;
	// 	return ret;
	// }

//	static private double calcLogBesselIQuotient(double a, double b, double nu) {
//		double ret = 0.0;
//		return ret;
//	}
	
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

	// public double calcLogVmfConstantQuotientWithNominatorPow(double k1, double n, double k2) {
	// 	double ret = 0.0;
	// 	ret += (embLen * 0.5 - 1) * (n * Math.log(k1) - Math.log(k2));
	// 	ret -= (n - 1) * (embLen * 0.5) * Math.log(2 * Math.PI) + calcLogBesselIQuotientIntWithNominatorPow(k1, n, k2, embLen * 0.5 - 1);
	// 	return ret;
	// }
	
	public void inferVmfMixtureByGibbsSampling(double[][] x) {
		// Initialization 
		int[] z = new int[x.length];
		double[][] sumXs = new double[numVmf][embLen];
		double[] sumZs = new double[numVmf];
		double totSumZs = 0.0;
		
		//non-inf??
		mu0 = new double[embLen];
		for (int j = 0; j < embLen; ++j) 
			mu0[j] = 1.0;
		mu0 = VecOp.normalize(mu0);

		
		for (int i = 0; i < x.length; ++i) {
			z[i] = VecOp.drawFromCatDist(distVmf);
			sumZs[z[i]] += 1.0 * 1;
			totSumZs += 1.0 * 1;
			for (int j = 0; j < embLen; ++j) sumXs[z[i]][j] += x[i][j] * 1;
		}
		
		for (int h = 0; h < numVmf; ++h) {
			double kappaCur = kappasVmf[h];
			// Metropolis
			int acc = 0, tot = 0;
			for (int kappaIter = 0; kappaIter < this.kappaGibbsIter; ++kappaIter) {
				double kappaNext = proposeKappa(kappaCur);
				// double logPiCur = calcLogKappaPosterior(kappaCur, sumZs[h], sumXs[h]);
				// double logPiNext = calcLogKappaPosterior(kappaNext, sumZs[h], sumXs[h]);
				// double r = Math.exp(logPiNext - logPiCur);
				double r = calcVmfConstantQuotient4Kappa(kappaNext, kappaCur, sumZs[h], sumXs[h]);
				//System.out.println("kappaNext="+kappaNext+",logPiCur="+logPiCur+",logPiNext="+logPiNext);
				if (rand.nextDouble() <= r) {
					kappaCur = kappaNext;
					++acc;
				}
				++tot;
			}

			System.out.println("[Vmf Mixture]kappa_" + h + "=" + kappasVmf[h] + ", alpha_" + h + "=" + distVmf[h] + ", sumZ_" + h + "=" + sumZs[h]+"，Kappa_" + h + " acc rate: " + (double)acc / tot);
			kappasVmf[h] = kappaCur;
		}
		// Gibbs sampling
		for (int iter = 0; iter < gibbsIter; ++iter) {
			System.out.println("[Vmf mixture] ===== Gibbs Sampling Iter " + iter + " ======" );
			// Sample z_i's
			for (int i = 0; i < x.length; ++i) {
				for (int j = 0; j < embLen; ++j) sumXs[z[i]][j] -= x[i][j] * 1;
				sumZs[z[i]] -= 1.0 * 1;
				
				double[] prob = new double[numVmf];
				double maxLogProb = - Double.MAX_VALUE;
				for (int h = 0; h < numVmf; ++h) {
					double[] vecSum = new double[embLen];
					for (int j = 0; j < embLen; ++j) 
						vecSum[j] = kappasVmf[h] * (sumXs[h][j] + R0 * mu0[j]);
					double lengthExc = VecOp.getL2(vecSum);
//					System.out.println("length_exc_" + h + "=" + lengthExc);
					for (int j = 0; j < embLen; ++j) 
						vecSum[j] += kappasVmf[h] * x[i][j] * 1;
					double lengthInc = VecOp.getL2(vecSum);
//					System.out.println("length_inc_" + h + "=" + lengthInc);
					prob[h] = Math.log(alpha + sumZs[h])
							+ vmfLogConstant(kappasVmf[h])
							+ calcLogVmfConstantQuotient(lengthExc, lengthInc);
					if (prob[h] > maxLogProb) maxLogProb = prob[h];
				}
//				for (int h = 0; h < numVmf; ++h)
//					System.out.println("z_cond_prob_" + h + "=" + prob[h]);
				
				for (int h = 0; h < numVmf; ++h) prob[h] = Math.exp(prob[h] - maxLogProb);
				prob = VecOp.vec2Dist(prob);
				int newZi = VecOp.drawFromCatDist(prob);
				
				z[i] = newZi;
				for (int j = 0; j < embLen; ++j) sumXs[z[i]][j] += x[i][j] * 1;
				sumZs[z[i]] += 1.0 * 1;
			}
			
			// Sample kappa_h's
			for (int h = 0; h < numVmf; ++h) {
				double kappaCur = kappasVmf[h];
				
				// Metropolis
				int acc = 0, tot = 0;
				for (int kappaIter = 0; kappaIter < this.kappaGibbsIter; ++kappaIter) {
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
				System.out.println("[Vmf Mixture] Kappa_" + h + " acc rate: " + (double)acc / tot);
				kappasVmf[h] = kappaCur;
			}
			
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

			for (int h = 0; h < numVmf; ++h)
				System.out.println("[Vmf Mixture]kappa_" + h + "=" + kappasVmf[h] + ", alpha_" + h + "=" + distVmf[h]);
			
			double L = calcLogLikelihood(x);
			System.out.println("[Vmf Mixture] Gibbs L = " + L);
		}

		// Use the last sample from posterior as the estimate
	}
	
	private double calcLogKappaPosterior(double kappa, double sumZ, double[] sumX) {
		double[] tempVec = new double[embLen];
		for (int j = 0; j < embLen; ++j) 
			tempVec[j] = kappa * (sumX[j] + R0 * mu0[j]);
		double logPi = (C0 + sumZ) * vmfLogConstant(kappa); //
		System.out.println("sumz:"+sumZ+",C0:"+C0+",R0:"+R0+",l2:"+VecOp.getL2(tempVec)+",logPi:"+logPi);
		logPi-= vmfLogConstant(VecOp.getL2(tempVec));
				// + logNormal.logDensity(kappa)
				     // + calcLogVmfConstantQuotientWithNominatorPow(kappa, sumZ, VecOp.getL2(tempVec));
		return logPi;
	}
	
	
// 	public void inferVmfMixtureByEM(double[][] d) {

// 		// Initialization
// 		double[][] condProb = new double[d.length][numVmf];
		
// 		// Inference
// 		double oldL = -Double.MAX_VALUE;
// 		int convCnt = 0;
// 		for (int iter = 0; iter < emMaxIter; ++iter) {
// 			System.out.println("[Vmf Mixture]=====Iter " + iter + "======");
			
// 			// E-step
// 			for (int i = 0; i < d.length; ++i) {
// 				double max = - Double.MAX_VALUE;
// //				System.out.println("dictlength_" + i + ":" + VecOp.getL2(d[i]));
// 				for (int h = 0; h < numVmf; ++h) {
// 					condProb[i][h] = Math.log(distVmf[h]) + vmfLogLikelihood(musVmf[h], kappasVmf[h], d[i]);
// 					max = max > condProb[i][h] ? max : condProb[i][h];
// 				}
// 				for (int h = 0; h < numVmf; ++h) condProb[i][h] = Math.exp(condProb[i][h] - max);
// 				condProb[i] = VecOp.vec2Dist(condProb[i]);
// 			}
			
// 			// M-step
// 			double alphaSum = 0.0;
// 			for (int h = 0; h < numVmf; ++h) {
// 				distVmf[h] = 0.0;
// 				for (int j = 0; j < embLen; ++j) musVmf[h][j] = 0.0;
// 				double tot = 0.0;
// 				for (int i = 0; i < d.length; ++i) {
// 					distVmf[h] += condProb[i][h] * 1;
// 					for (int j = 0; j < embLen; ++j) musVmf[h][j] += condProb[i][h] * 1 * d[i][j];
// 					tot += 1;
// 				}
// 				double rLen = VecOp.getL2(musVmf[h]);
// 				double rAvg = rLen / tot;
// 				distVmf[h] /= tot;
// 				kappasVmf[h] = (rAvg * embLen - Math.pow(rAvg, 3)) / (1 - rAvg * rAvg);
// 				if (kappasVmf[h] < kappaMinValue) kappasVmf[h] = kappaMinValue;  // Truncated to prevent numerical difficulty
// 				musVmf[h] = VecOp.normalize(musVmf[h]);
// 				alphaSum += distVmf[h];
// 				// DEBUG:
// 				System.out.println("[Vmf Mixture]kappa_" + h + "=" + kappasVmf[h] + ", alpha_" + h + "=" + distVmf[h]);
// 			}
			

// 			double L = calcLogLikelihood(d);
// 			if ((L - oldL) / oldL < eps) ++convCnt;
// 			else convCnt = 0;
// 			System.out.println("[Vmf Mixture]L = " + L);
// 			oldL = L;
// 			if (convCnt > 5) {
// 				System.out.println("[Vmf Mixture]Converged.");
// 				break;
// 			}
// 		}
// 	}
	
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
//				pXi += condProb[i][h] * l;
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
	
	public void saveResult(String fileName, double [][] x, long setSize) throws Exception{
		System.out.println(fileName);
		// cluster_result
		BufferedWriter nbw = new BufferedWriter(new FileWriter(new File(fileName)));
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
//				pXi += condProb[i][h] * l;
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
	
	static public void main(String[] args) throws Exception {
//		System.out.println(VmfMixture.calcLogBesselIQuotientInt(30000, 31000, 200));
		
//		String fileNameList =  "/Users/hzhuang/lab/odt/exp/1130_odttest_nyt/intrusion.1.data";
		String bgFileName = "/shared/data/ll2/tweets/embedding_p2v2_1.txt"; 
		int embLen = 100;
		int setSize = 5768;
		double[][] x;
		File file = new File(bgFileName);
		Scanner scan = new Scanner(file);
	        x = new double[setSize][embLen];
	        for (int i = 0;i < setSize; ++i){
	        	for (int j = 0;j < embLen; ++j){
	        		x[i][j] = scan.nextDouble();
	        	}
	        }
// 	
		int numVmf = 200;
		VmfMixture vmfMixture = new VmfMixture(embLen, numVmf);
		vmfMixture.initVmfMixture();
		vmfMixture.inferVmfMixtureByGibbsSampling(x);
		vmfMixture.saveModel("/shared/data/ll2/tweets/model/p2vn20_2.txt");
		vmfMixture.saveResult("/shared/data/ll2/tweets/results/p2vn20_y_2.csv", x, setSize);
	
	}
}

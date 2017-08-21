package util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

public class VecOp {

	static Random rand = new Random(299792458);
	
	static public double[] vec2Dist(double[] v) {
		double z = getL1(v);
		if (z == 0.0) return null;
		double[] ret = new double[v.length];
		for (int i = 0; i < v.length; ++i) ret[i] = v[i] / z;
		return ret;
	}
	
	static public <V extends Number> List<Double> vec2Dist(List<V> v) {
		double z = 0.0;
		for (V e : v) z += e.doubleValue();
		if (z == 0.0) return null;
		List<Double> ret = new ArrayList<Double>();
		for (V e : v) ret.add(e.doubleValue() / z);
		return ret;
	}
	
	static public <K, V extends Number> Map<K, Double> vec2Dist(Map<K, V> v) {
		double z = 0.0;
		for (Entry<K, V> e : v.entrySet()) z += e.getValue().doubleValue();
		if (z == 0.0) return null;
		Map<K, Double> ret = new HashMap<K, Double>();
		for (Entry<K, V> e : v.entrySet()) ret.put(e.getKey(), e.getValue().doubleValue() / z);
		return ret;
	}
	
	static public <K, V extends Number> double getL2(Map<K, V> v) {
		double z = 0.0;
		for (Entry<K, V> e : v.entrySet()) z += e.getValue().doubleValue() * e.getValue().doubleValue();
		return Math.sqrt(z);
	}

	static public double getL2(double[] v) {
		double z = 0.0;
		for (int i = 0; i < v.length; ++i) z += v[i] * v[i];
		return Math.sqrt(z);
	}
	
	static public <K, V extends Number> double getL1(Map<K, V> v) {
		double z = 0.0;
		for (Entry<K, V> e : v.entrySet()) z += Math.abs(e.getValue().doubleValue());
		return z;
	}
	
	static public double getL1(double[] v)  {
		double z = 0.0;
		for (int i = 0; i < v.length; ++i) z += Math.abs(v[i]);
		return z;
	}

	static public <K, V extends Number> double innerProd(Map<K, V> v1, Map<K, V> v2) {
		double prod = 0.0;
		for (Entry<K, V> e : v1.entrySet()) {
			prod += v2.containsKey(e.getKey()) ? v2.get(e.getKey()).doubleValue() * e.getValue().doubleValue() : 0.0;
		}
		return prod;
	}
	
	static public double innerProd(double[] v1, double[] v2)  {
		if (v1.length != v2.length) return Double.NaN;
		double z = 0.0;
		for (int i = 0; i < v1.length; ++i) z += v1[i] * v2[i];
		return z;
	}
	
	static public double getDist(double[] v1, double[] v2)  {
		if (v1.length != v2.length) return Double.NaN;
		double z = 0.0;
		for (int i = 0; i < v1.length; ++i) z += (v1[i] - v2[i]) * (v1[i] - v2[i]);
		return z;
	}
	
	static public double[] normalize(double[] v) {
		double z = getL2(v);
		double[] ret = new double[v.length];
		for (int i = 0; i < v.length; ++i) ret[i] = v[i] / z;
		return ret;
	}
	
	static public <V extends Number> double getAvg(List<V> v) {
		if (v.size() == 0) return 0;
		double sum = 0.0;
		for (V x : v) {
			sum += x.doubleValue();
		}
		return sum / v.size();
	}
	
	static public int binSearch(double v, double[] a, int p1, int p2) {
		if (p2 - p1 <= 1) return p1;
		int mid = ((p1 + p2) >> 1);
		if (v < a[mid]) return binSearch(v, a, p1, mid);
		else return binSearch(v, a, mid, p2);
	}
	
	static public int binSearch(double v, double[] a) {
		return binSearch(v, a, 0, a.length);
	}
	
	static public int drawFromCatDist(double[] dist) {
		double[] distSum = new double[dist.length];
		for (int i = 0; i < dist.length - 1; ++i) distSum[i + 1] = distSum[i] + dist[i];
		return binSearch(rand.nextDouble(), distSum);
	}
	
	static public boolean listContain(List<? extends Object> l1, List<? extends Object> l2) {
		int id1 = 0, id2 = 0;
		while (id1 < l1.size() && id2 < l2.size()) {
			if (l1.get(id1).equals(l2.get(id2))) ++id2;
			++id1;
		}
		if (id2 >= l2.size()) return true;
		return false;
	}
	
	static public <T> boolean listContain(T[] l1, T[] l2) {
		int id1 = 0, id2 = 0;
		while (id1 < l1.length && id2 < l2.length) {
			if (l1[id1].equals(l2[id2])) ++id2;
			++id1;
		}
		if (id2 >= l2.length) return true;
		return false;
	}
	
	static public String toString(double[] v) {
		String s = "";
		for (int i = 0; i < v.length - 1; ++i) s += v[i] + ",\t";
		s += v[v.length - 1];
		return s;
	}
	
	static public String toString(List<? extends Object> v) {
		String s = "";
		for (int i = 0; i < v.size() - 1; ++i) s += v.get(i).toString() + ",\t";
		s += v.get(v.size() - 1).toString();
		return s;
	}
	
	static public void main(String[] args) {
		double[] dist = new double[]{0.1, 0.2, 0.3, 0.4};
		int[] cnt = new int[dist.length];
		for (int i =0 ; i < 10000; ++i) {
			int s = VecOp.drawFromCatDist(dist);
			cnt[s]++;
		}
		
		for (int i = 0; i < dist.length; ++i) {
			System.out.println(cnt[i]);
		}
	}
}

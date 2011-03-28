package explo.agent;

import java.util.HashMap;
import java.io.Serializable;

import org.ejml.alg.dense.decomposition.CholeskyDecomposition;
import org.ejml.alg.dense.linsol.LinearSolver;
import org.ejml.alg.dense.linsol.LinearSolverFactory;
import org.ejml.alg.dense.linsol.chol.LinearSolverChol;
import org.ejml.alg.dense.mult.VectorVectorMult;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.SpecializedOps;
import org.ejml.simple.SimpleMatrix;

import explo.model.Arm;
import explo.model.Batch;
import explo.model.Option;


/**
 * ClickPredictors are used to predict which arm in a given batch is most likely to yield a reward of 1. They typically implement a learning algorithm.
 * The class is Serializable so that a ClickPredictor instance can be saved in a file and its learnings can be reused later. 
 * @author Louis Dorard, University College London
 */
public class ClickPredictor implements Serializable {

	private static final long serialVersionUID = 1L;
	/**
	 * This can be used for storing training data. You may change the type of this object depending on your needs.
	 */
	private HashMap<Option,DenseMatrix64F> trainingX;
	private HashMap<Option,DenseMatrix64F> trainingb;
	
	private static int num_options = 6;
	private static int num_features = 100;
	private static double ucb_beta = 5;
	
	/**
	 * Default constructor.
	 */
	public ClickPredictor() {
		trainingX = new HashMap<Option, DenseMatrix64F>();
		trainingb = new HashMap<Option, DenseMatrix64F>();
	}
	
	/**
	 * Copy constructor: copies all fields of the class.
	 * @param cp - original click predictor to copy from
	 */
	public ClickPredictor(ClickPredictor cp) {
		trainingX = new HashMap<Option, DenseMatrix64F>();
		trainingb = new HashMap<Option, DenseMatrix64F>();
		for (Option o : cp.trainingX.keySet()) {
			trainingX.put(o, cp.trainingX.get(o).copy());
			trainingb.put(o, cp.trainingb.get(o).copy());
		}
	}

	private void initializeArm(Arm a) {
		DenseMatrix64F X;
		DenseMatrix64F b;
		X = CommonOps.identity(num_features);
		b = new DenseMatrix64F(num_features, 1);
		CommonOps.set(b,1);
		trainingX.put(a.o, X);
		trainingb.put(a.o, b);
	}
	
	/**
	 * Learns from feedback received from the environment.
	 * @param a - arm that was played
	 * @param reward that was observed for a
	 */
	public void learn(Arm a, Integer reward) {
		// Be careful not to use more than half of the total available memory (explo.control.Run needs to keep a copy of your ClickPredictor instance).
		// You may want to call the garbage collector (System.gc();) for accurate memory usage reporting in the logs, but it might be very expensive if called at each iteration...
		
		DenseMatrix64F X;
		DenseMatrix64F b;
		
		// New option -- initialise data structures
		X = trainingX.get(a.o);
		b = trainingb.get(a.o);

		// Update the features
		DenseMatrix64F xt = getArmFeaturesAsVector(a);
		VectorVectorMult.addOuterProd(1, xt, xt, X);
			
        // Update the reward
        CommonOps.addEquals(b, reward, xt);
                
		trainingX.put(a.o, X);
		trainingb.put(a.o, b);
		
		Runtime runtime = Runtime.getRuntime();
		long free = runtime.freeMemory();
		long total = runtime.totalMemory();
		long used = total - free;
		//trainingX.put(a, reward);
		//if (used > 0.9*total/2) { // making sure we're not using more than 90% of half the total memory: you may want to leave some breathing space for running your computations
		//	trainingX.remove(trainingX.keySet().iterator().next());
		//}
		
	}

	public DenseMatrix64F getArmFeaturesAsVector(Arm a) {
		Number[] feats_con = a.u.getFeaturesCON();
		DenseMatrix64F xt = new DenseMatrix64F(num_features, 1);
		for (int i = 0; i<feats_con.length; i++) {
			if (feats_con[i] == null) continue;
			xt.set(i, feats_con[i].doubleValue());
		}
		return xt;
	}
	
	/**
	 * Returns an upper bound for the reward for each arm in the batch 
	 * @param b - batch of arms
	 * @return UCBs for each arm in batch
	 */
	public double[] predict(Batch batch) {
		double[] ucb = new double[batch.size()];
		for(int i=0; i<batch.size(); i++) {
			Arm a = batch.getArm(i);
			if(!trainingX.containsKey(a.o)) initializeArm(a);
			DenseMatrix64F xt = getArmFeaturesAsVector(a);
			DenseMatrix64F X = trainingX.get(a.o);
			DenseMatrix64F b = trainingb.get(a.o);
			LinearSolver<DenseMatrix64F> solv = LinearSolverFactory.symmPosDef(num_features);
			solv.setA(X);
			DenseMatrix64F theta = new DenseMatrix64F(num_features,1);
			solv.solve(b, theta);
			double mean = VectorVectorMult.innerProd(theta, xt);
			solv.solve(xt, theta);
			double var = 1+VectorVectorMult.innerProd(theta, xt);
			ucb[i] = mean + ucb_beta*Math.sqrt(var); 
		}
		return ucb;
	}
	
	/**
	 * Returns the index of the Arm in the Batch that is most likely to yield a click (indices start at 0). The arm is to be played in the environment.
	 * @param b - batch of arms
	 * @return index of the chosen arm
	 */
	public int choose(Batch b) {
		double ucb[] = predict(b);
		int bestarm = 0;
		double maxucb = 0;
		for (int i=0; i<b.size(); i++) {
			if (ucb[i]<=maxucb) continue;
			maxucb = ucb[i];
			bestarm = i;
		}
		return bestarm;
	}
	
}

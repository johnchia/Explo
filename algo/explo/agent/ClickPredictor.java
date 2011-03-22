package explo.agent;

import java.util.HashMap;
import java.io.Serializable;

import org.ujmp.core.bigdecimalmatrix.BigDecimalMatrix;
import org.ujmp.core.bigdecimalmatrix.BigDecimalMatrix2D;

import explo.model.Arm;
import explo.model.Batch;


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
	private HashMap<Arm,BigDecimalMatrix2D> trainingX;
	private HashMap<Arm,BigDecimalMatrix> trainingb;
	
	/**
	 * Default constructor.
	 */
	public ClickPredictor() {
		trainingX = new HashMap<Arm, BigDecimalMatrix2D>();
		trainingb = new HashMap<Arm, BigDecimalMatrix>();
	}
	
	/**
	 * Copy constructor: copies all fields of the class.
	 * @param cp - original click predictor to copy from
	 */
	public ClickPredictor(ClickPredictor cp) {
		trainingX = cp.trainingX;
	}
	
	/**
	 * Learns from feedback received from the environment.
	 * @param a - arm that was played
	 * @param reward that was observed for a
	 */
	public void learn(Arm a, Integer reward) {
		
		// TODO implement your own code here
		// Be careful not to use more than half of the total available memory (explo.control.Run needs to keep a copy of your ClickPredictor instance).
		// You may want to call the garbage collector (System.gc();) for accurate memory usage reporting in the logs, but it might be very expensive if called at each iteration...
		
		/*
		 * Sample code:
		 * here we don't really "learn" but we just memorise the training examples
		 */
		Runtime runtime = Runtime.getRuntime();
		long free = runtime.freeMemory();
		long total = runtime.totalMemory();
		long used = total - free;
		trainingX.put(a, reward);
		if (used > 0.9*total/2) { // making sure we're not using more than 90% of half the total memory: you may want to leave some breathing space for running your computations
			trainingX.remove(trainingX.keySet().iterator().next());
		}
		
	}
	
	/**
	 * Returns the index of the Arm in the Batch that is most likely to yield a click (indices start at 0). The arm is to be played in the environment.
	 * @param b - batch of arms
	 * @return index of the chosen arm
	 */
	public int choose(Batch b) {
		
		// TODO implement your own code here
		
		/*
		 * Sample code:
		 * here we choose an index randomly
		 */
		return b.getRandomIndex();
		
	}
	
}

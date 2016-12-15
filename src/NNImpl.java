/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
	public ArrayList<Node> inputNodes=null;//list of the output layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public ArrayList<Node> outputNodes=null;// list of the output layer nodes

	public ArrayList<Instance> trainingSet=null;//the training set

	Double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs

	/**
	 * This constructor creates the nodes necessary for the neural network
	 * Also connects the nodes of different layers
	 * After calling the constructor the last node of both inputNodes and  
	 * hiddenNodes will be bias nodes. 
	 */

	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[][] outputWeights)
	{	
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;

		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();

		int outputNodeCount=trainingSet.get(0).classValues.size();
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}

		//bias node from input layer to hidden
		Node biasToHidden=new Node(1);
		inputNodes.add(biasToHidden);

		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i=0;i<hiddenNodeCount;i++)
		{
			Node node=new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j=0;j<inputNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}

		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);
		hiddenNodes.add(biasToOutput);

		//Output node layer
		outputNodes=new ArrayList<Node> ();
		for(int i=0;i<outputNodeCount;i++)
		{
			Node node=new Node(4);
			//Connecting output layer nodes with hidden layer nodes
			for(int j=0;j<hiddenNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}	
	}

	/**
	 * Get the output from the neural network for a single instance
	 * Return the idx with highest output values. For example if the outputs
	 * of the outputNodes are [0.1, 0.5, 0.2], it should return 1. If outputs
	 * of the outputNodes are [0.1, 0.5, 0.5], it should return 2. 
	 * The parameter is a single instance. 
	 */

	public int calculateOutputForInstance(Instance inst)
	{	
		double outputVal = -1.0;
		int index = 0;

		//just run the instance I think
		//bias to hidden at end
		for(int i = 0; i < this.inputNodes.size() - 1; i++)
		{
			Node inputNode = this.inputNodes.get(i);

			inputNode.setInput(inst.attributes.get(i));
			this.inputNodes.set(i, inputNode);
		}

		//override hidden layer
		for(int h = 0; h < this.hiddenNodes.size(); h++)
		{	
			Node hidden = this.hiddenNodes.get(h);
			
			hidden.calculateOutput();

			if(h != this.hiddenNodes.size() - 1)
			{
				for(int i = 0; i < this.inputNodes.size(); i++)
				{
					NodeWeightPair old = hidden.parents.get(i);

					NodeWeightPair newPair = new NodeWeightPair(inputNodes.get(i), old.weight);

					hidden.parents.set(i, newPair);
				}
			}

			this.hiddenNodes.set(h, hidden);
		}

		//override hidden layer
		for(int o = 0; o < this.outputNodes.size(); o++)
		{
			Node output = this.outputNodes.get(o);
			
			output.calculateOutput();

			for(int h = 0; h < this.hiddenNodes.size(); h++)
			{
				NodeWeightPair old = output.parents.get(h);

				NodeWeightPair newPair = new NodeWeightPair(this.hiddenNodes.get(h), old.weight);

				output.parents.set(h, newPair);
			}

			this.outputNodes.set(o, output);
		}

		//for each output node, get output
		for(int i = 0; i < outputNodes.size(); i++)
		{
			Node outputNode = outputNodes.get(i);

			System.out.println("in");
			System.out.println(outputNode.getOutput());
			System.out.println("out");

			if(outputNode.getOutput() >= outputVal)
			{
				outputVal = outputNode.getOutput();
				index = i;
			}
		}
		
		return index;
	}

	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */

	public void train()
	{
		for(int e = 0; e < maxEpoch; e++)
		{
			// TODO: add code here
			//we will do the following
			//train until we run out of training examples
			for(int i = 0; i < this.trainingSet.size(); i++)
			{
				//breakpoint
				//if(i == 1)
					//System.exit(0);
				Instance getInstance = trainingSet.get(i);

				calculateOutputForInstance(getInstance);

				ArrayList <Double> deltaOut = new ArrayList <Double> ();
				ArrayList <Double> deltaHid = new ArrayList <Double>();

				//calculate deltas for outputs
				for(int c = 0; c < getInstance.classValues.size(); c++)
				{
					Node output = this.outputNodes.get(c);
					double actual = (double)(getInstance.classValues.get(c));

					double outputVal = output.getOutput();

					if(outputVal == 0)
						deltaOut.add(0.0);

					else
					{
						//double delta = 1.0;
						double delta = (outputVal - actual);//*outputVal*(1 - outputVal);
						deltaOut.add(delta);
					}
				}

				for(int h = 0; h < this.hiddenNodes.size(); h++)
				{
					Node hidden = this.hiddenNodes.get(h);

					double sum = 0.0;

					//calculate deltas for outputs
					for(int c = 0; c < deltaOut.size(); c++)
					{
						Node output = this.outputNodes.get(c);

						double weight = output.parents.get(h).weight;
						sum += weight * deltaOut.get(c);
					}

					double outputVal = hidden.getOutput();

					if(outputVal == 0)
						deltaHid.add(0.0);

					else
						deltaHid.add(sum);
					//deltaHid.add(1.0);
					//double delta = sum * outputVal * (1 - outputVal);
					//deltaHid.add(delta);
				}

				//new weights
				//for each hidden node, update weights
				for(int h = 0; h < deltaHid.size(); h++)
				{
					Node hidden = this.hiddenNodes.get(h);

					double delta = deltaHid.get(h);

					//bias node
					if(hidden.parents == null)
					{
						continue;
					}

					for(int j = 0; j < hidden.parents.size(); j++)
					{
						NodeWeightPair getPair = hidden.parents.get(j);

						double newWeight = getPair.weight - this.learningRate * delta * getPair.node.getOutput();

						NodeWeightPair newPair = new NodeWeightPair(getPair.node, newWeight);

						hidden.parents.set(j, newPair);
					}

					this.hiddenNodes.set(h, hidden);
				}

				//for each output node, update weights
				//for each hidden node, update weights
				for(int c = 0; c < deltaOut.size(); c++)
				{
					Node output = this.outputNodes.get(c);

					double delta = deltaOut.get(c);

					for(int j = 0; j < output.parents.size(); j++)
					{
						NodeWeightPair getPair = output.parents.get(j);

						double newWeight = getPair.weight - this.learningRate * delta * getPair.node.getOutput();

						NodeWeightPair newPair = new NodeWeightPair(getPair.node, newWeight);

						output.parents.set(j, newPair);
					}

					this.outputNodes.set(c, output);
				}
			}    
		}
	}
}

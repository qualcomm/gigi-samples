/*$(ShaderResources)*/

#include "shared.slang"

static int batchSize = /*$(Variable:mBatchSize)*/;

static int frameNumber = /*$(Variable:frameNumber)*/;


static uint outputHeight = /*$(Variable:frameHeight)*/;
static uint outputWidth  = /*$(Variable:frameWidth)*/;

static float rcpBatchSize = 1.0f / float(/*$(Variable:mBatchSize)*/);

static float adamBeta1 = /*$(Variable:adamBeta1)*/ ;
static float adamBeta2 = /*$(Variable:adamBeta2)*/ ;

static float learningRate = /*$(Variable:mLearningRate)*/;




// .d88888b.           888    d8b               d8b                   
// d88P" "Y88b          888    Y8P               Y8P                   
// 888     888          888                                            
// 888     888 88888b.  888888 888 88888b.d88b.  888 88888888  .d88b.  
// 888     888 888 "88b 888    888 888 "888 "88b 888    d88P  d8P  Y8b 
// 888     888 888  888 888    888 888  888  888 888   d88P   88888888 
// Y88b. .d88P 888 d88P Y88b.  888 888  888  888 888  d88P    Y8b.     
//  "Y88888P"  88888P"   "Y888 888 888  888  888 888 88888888  "Y8888  
//             888                                                     
//             888                                                     
//             888                                                     

#define SGD_OPTIMIZER  0
#define ADAM_OPTIMIZER 1

#define OPTIMIZER ADAM_OPTIMIZER

// Source: "ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION"
// https://arxiv.org/pdf/1412.6980
float AdamOptimizer(const float gradient,
                    inout float mean,
                    inout float variance)
{
    // Update mean and variance for this training step
    mean     = lerp( gradient,             mean,     adamBeta1);
    variance = lerp((gradient * gradient), variance, adamBeta2);
    
    const float adamEpsilon = 0.00000001f;

    // Calculate weight adjustment
    const float correctedMean     =      mean / (1.0f - nnData[0].adamBeta1T);
    const float correctedVariance =  variance / (1.0f - nnData[0].adamBeta2T);
    const float weightAdjustment  = -learningRate * (correctedMean / (sqrt(correctedVariance) + adamEpsilon));

    return weightAdjustment;
}

float WeightOptimizer(const float gradient, 
                      const uint dataIndex)
{
  float weightAdjustment = 0.0f;

#if OPTIMIZER == SGD_OPTIMIZER
    weightAdjustment = -gradient * learningRate;
#elif OPTIMIZER == ADAM_OPTIMIZER

    // Load mean and variance
    float mean     = adamWeightsMeans    [dataIndex];
    float variance = adamWeightsVariances[dataIndex];

    weightAdjustment = AdamOptimizer(gradient, mean, variance);

    adamWeightsMeans    [dataIndex] = mean;
    adamWeightsVariances[dataIndex] = variance;

#endif

    return weightAdjustment;
}

float BiasOptimizer(const float gradient, 
                    const uint dataIndex)
{
    float weightAdjustment = 0.0f;

#if OPTIMIZER == SGD_OPTIMIZER
    weightAdjustment = -gradient * learningRate;
#elif OPTIMIZER == ADAM_OPTIMIZER    
    // Load mean and variance
    float mean     = adamBiasesMeans    [dataIndex];
    float variance = adamBiasesVariances[dataIndex];

    weightAdjustment = AdamOptimizer(gradient, mean, variance);

    adamBiasesMeans    [dataIndex] = mean;
    adamBiasesVariances[dataIndex] = variance;

#endif

    return weightAdjustment;
}

/*$(_compute:Optimize)*/ (uint3 DTid: SV_DispatchThreadID)
{
    const uint layer  = DTid.x + 1;
    const uint neuron = DTid.y;

    if (layer >= LAYER_COUNT) return;
    
    const uint neuronCountCurrentLayer  = neuronsPerLayer[layer];
    const uint neuronCountPreviousLayer = neuronsPerLayer[layer - 1];
    if (neuron >= neuronCountCurrentLayer) return;
    
    // Update bias value
    const uint  neuronDataIndex = getNeuronDataIndex(layer, neuron);
    const float gradBias        = unpackFloat(gradientBiases[neuronDataIndex]) * rcpBatchSize;
    
    nnBiases[neuronDataIndex] += BiasOptimizer(gradBias, neuronDataIndex);

    // Update weights leading to this neuron
    for (uint previousNeuron = 0; previousNeuron < neuronCountPreviousLayer; previousNeuron++)
    {
        const uint weightIndex = getConnectionDataIndex(layer, previousNeuron, neuron);
        const float gradWeight = unpackFloat(gradientWeights[weightIndex]) * rcpBatchSize;
    
        nnWeights[weightIndex] += BiasOptimizer(gradWeight, weightIndex);
    }
}
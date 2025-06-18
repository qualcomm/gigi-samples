/*$(ShaderResources)*/

#include "shared.slang"


static int batchSize   = /*$(Variable:mBatchSize)*/;
static int frameNumber = /*$(Variable:frameNumber)*/;


// 888888b.                     888      8888888b.                           
// 888  "88b                    888      888   Y88b                          
// 888  .88P                    888      888    888                          
// 8888888K.   8888b.   .d8888b 888  888 888   d88P 888d888 .d88b.  88888b.  
// 888  "Y88b     "88b d88P"    888 .88P 8888888P"  888P"  d88""88b 888 "88b 
// 888    888 .d888888 888      888888K  888        888    888  888 888  888 
// 888   d88P 888  888 Y88b.    888 "88b 888        888    Y88..88P 888 d88P 
// 8888888P"  "Y888888  "Y8888P 888  888 888        888     "Y88P"  88888P"  
//                                                                  888      
//                                                                  888      
//                                                                  888      


void backpropagation(float3 target, 
                     float  activations[LAYER_COUNT * MAX_NEURONS_PER_LAYER])
{
    float errors[LAYER_COUNT * MAX_NEURONS_PER_LAYER];

    // Output layer derivatives
    {
        const uint neuronCountCurrentLayer  = neuronsPerLayer[LAYER_COUNT - 1];
        const uint neuronCountPreviousLayer = neuronsPerLayer[LAYER_COUNT - 2];
   
        for (uint neuron = 0; neuron < neuronCountCurrentLayer; neuron++)
        {
            const uint  neuronDataIndex  = getNeuronDataIndex(LAYER_COUNT - 1, neuron);
            const float neuronActivation = activations[neuronDataIndex];
            const float dCost_O = 2.0f * (neuronActivation - target[neuron]);
            const float dO_Z = ACTIVATION_FUNCTION_DERIV(neuronActivation);
            const float dCost_Z = dCost_O * dO_Z;
            errors[neuronDataIndex] = dCost_Z;
            InterlockedAdd(gradientBiases[NonUniformResourceIndex(neuronDataIndex)], packFloat(dCost_Z));
            
            // Update weights
            for (uint previousNeuron = 0; previousNeuron < neuronCountPreviousLayer; previousNeuron++)
            {
                const uint previousNeuronDataIndex = getNeuronDataIndex(LAYER_COUNT - 2, previousNeuron);
                const float dCost_weight = dCost_Z * activations[previousNeuronDataIndex];
                const uint weightIndex = getConnectionDataIndex(LAYER_COUNT - 1, previousNeuron, neuron);
                InterlockedAdd(gradientWeights[NonUniformResourceIndex(weightIndex)], packFloat(dCost_weight));
            }
        }
    }
    
    // Hidden layer derivatives
    {
        [unroll(LAYER_COUNT - 2)]
        for (uint layer = LAYER_COUNT - 2; layer > 0; layer--)
        {
            const uint neuronCountCurrentLayer  = neuronsPerLayer[layer];
            const uint neuronCountPreviousLayer = neuronsPerLayer[layer - 1];
            const uint neuronCountNextLayer     = neuronsPerLayer[layer + 1];
   
            for (uint neuron = 0; neuron < neuronCountCurrentLayer; neuron++)
            {
                float dCost_O = 0.0f;
                for (uint nextNeuron = 0; nextNeuron < neuronCountNextLayer; nextNeuron++)
                {
                    const uint weightIndex         = getConnectionDataIndex(layer + 1, neuron, nextNeuron);
                    const uint nextNeuronDataIndex = getNeuronDataIndex(layer + 1, nextNeuron);
                    dCost_O += (errors[nextNeuronDataIndex] * nnWeights[NonUniformResourceIndex(weightIndex)]);
                }
                
                const uint  neuronDataIndex  = getNeuronDataIndex(layer, neuron);
                const float neuronActivation = activations[neuronDataIndex];
                const float dO_Z    = ACTIVATION_FUNCTION_DERIV(neuronActivation);
                const float dCost_Z = dCost_O * dO_Z;
                errors[neuronDataIndex] = dCost_Z;
                InterlockedAdd(gradientBiases[NonUniformResourceIndex(neuronDataIndex)], packFloat(dCost_Z));

                // Update weights
                for (uint previousNeuron = 0; previousNeuron < neuronCountPreviousLayer; previousNeuron++)
                {
                    const uint  previousNeuronDataIndex = getNeuronDataIndex(layer - 1, previousNeuron);
                    const float dCost_weight = dCost_Z * activations[previousNeuronDataIndex];
                    const uint  weightIndex  = getConnectionDataIndex(layer, previousNeuron, neuron);
                    InterlockedAdd(gradientWeights[NonUniformResourceIndex(weightIndex)], packFloat(dCost_weight));
                }
            }
        }
    }
}






// 88888888888              d8b          d8b                   
//     888                  Y8P          Y8P                   
//     888                                                     
//     888  888d888 8888b.  888 88888b.  888 88888b.   .d88b.  
//     888  888P"      "88b 888 888 "88b 888 888 "88b d88P"88b 
//     888  888    .d888888 888 888  888 888 888  888 888  888 
//     888  888    888  888 888 888  888 888 888  888 Y88b 888 
//     888  888    "Y888888 888 888  888 888 888  888  "Y88888 
//                                                         888 
//                                                    Y8b d88P 
//                                                     "Y88P"  


/*$(_compute:Training)*/ (uint3 DTid: SV_DispatchThreadID)
{
    if (DTid.x >= batchSize || DTid.y > 0)
        return;

    // Initialize random numbers generator
    uint rng = initRNG(DTid.xy, uint2(1, 1), frameNumber);
    
    // Generate a random input (UV coordinates in the image)
    const float2 UVs = float2(rand(rng), rand(rng));

    uint2 texIdx = uint2(uint(UVs.x * /*$(Variable:frameWidth)*/ - 1), uint(UVs.y * /*$(Variable:frameHeight)*/ - 1));

    // Load target value to learn for this input from reference image
    const float3 target = targetTexture[texIdx].rgb;
    
    // First run forward pass to evaluate network activations for given input
    float activations[LAYER_COUNT * MAX_NEURONS_PER_LAYER];
    forwardPass(UVs, nnWeights, nnBiases, activations);
    
    // Run backpropagation on current network state
    backpropagation(target, activations);
}
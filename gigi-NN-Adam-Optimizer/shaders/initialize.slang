/*$(ShaderResources)*/

#include "shared.slang"

static const int frameNumber = 0;///*$(Variable:frameNumber)*/;



// 8888888          d8b 888    d8b          888 d8b                   
//   888            Y8P 888    Y8P          888 Y8P                   
//   888                888                 888                       
//   888   88888b.  888 888888 888  8888b.  888 888 88888888  .d88b.  
//   888   888 "88b 888 888    888     "88b 888 888    d88P  d8P  Y8b 
//   888   888  888 888 888    888 .d888888 888 888   d88P   88888888 
//   888   888  888 888 Y88b.  888 888  888 888 888  d88P    Y8b.     
// 8888888 888  888 888  "Y888 888 "Y888888 888 888 88888888  "Y8888  


/*$(_compute:Initialize)*/ (uint3 DTid: SV_DispatchThreadID)
{

    const uint layer  = DTid.x + 1;
    const uint neuron = DTid.y;

    if (layer >= LAYER_COUNT)
        return;

    const uint neuronCountCurrentLayer  = neuronsPerLayer[layer + 0];
    const uint neuronCountPreviousLayer = neuronsPerLayer[layer - 1];
    if (neuron >= neuronCountCurrentLayer)
        return;

    // Initialize bias value to zero
    const uint neuronDataIndex = getNeuronDataIndex(layer, neuron);
    nnBiases[neuronDataIndex] = 0.0f;

    // Initialize Adam optimizer biases data (mean & variance)
    adamBiasesMeans    [neuronDataIndex] = 0.0f;
    adamBiasesVariances[neuronDataIndex] = 0.0f;

    uint rng = initRNG(DTid.xy, uint2(LAYER_COUNT, neuronCountCurrentLayer), frameNumber);
    const float xavierWeightScale = xavierUniformScale(rng, neuronCountPreviousLayer, neuronCountCurrentLayer);

    // Initialize weights leading to this neuron
    for (uint previousNeuron = 0; previousNeuron < neuronCountPreviousLayer; previousNeuron++)
    {
        const float initialWeight = randInRange(rng, xavierWeightScale);
        const uint weightIndex = getConnectionDataIndex(layer, previousNeuron, neuron);
        nnWeights[weightIndex] = initialWeight;

        // Initialize Adam optimizer weights data (mean & variance)
        adamWeightsMeans    [weightIndex] = 0.0f;
        adamWeightsVariances[weightIndex] = 0.0f;
    }

}


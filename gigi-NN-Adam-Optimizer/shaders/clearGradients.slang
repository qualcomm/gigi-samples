/*$(ShaderResources)*/

#include "shared.slang"


// This kernel clears gradient buffers to zeroes
/*$(_compute:ClearGradients)*/ (uint3 DTid: SV_DispatchThreadID)
{
    const uint layer  = DTid.x + 1;
    const uint neuron = DTid.y;
    if (layer >= LAYER_COUNT)
        return;

    const uint neuronCountCurrentLayer  = neuronsPerLayer[layer + 0];
    const uint neuronCountPreviousLayer = neuronsPerLayer[layer - 1];
    if (neuron >= neuronCountCurrentLayer)
        return;

    // Clear bias value
    const uint neuronDataIndex = getNeuronDataIndex(layer, neuron);
    gradientBiases[neuronDataIndex] = 0;

    // Clear weights leading to this neuron
    for (uint previousNeuron = 0; previousNeuron < neuronCountPreviousLayer; previousNeuron++)
    {
        const uint weightIndex = getConnectionDataIndex(layer, previousNeuron, neuron);
        gradientWeights[weightIndex] = 0;
    }
}
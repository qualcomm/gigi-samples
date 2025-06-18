/*$(ShaderResources)*/


#include "shared.slang"


static uint outputHeight = /*$(Variable:frameHeight)*/;
static uint outputWidth  = /*$(Variable:frameWidth)*/;


// 8888888           .d888                                                   
//   888            d88P"                                                    
//   888            888                                                      
//   888   88888b.  888888 .d88b.  888d888 .d88b.  88888b.   .d8888b .d88b.  
//   888   888 "88b 888   d8P  Y8b 888P"  d8P  Y8b 888 "88b d88P"   d8P  Y8b 
//   888   888  888 888   88888888 888    88888888 888  888 888     88888888 
//   888   888  888 888   Y8b.     888    Y8b.     888  888 Y88b.   Y8b.     
// 8888888 888  888 888    "Y8888  888     "Y8888  888  888  "Y8888P "Y8888  


/*$(_compute:Inference)*/ (uint3 DTid: SV_DispatchThreadID)
{
    if (DTid.x >= outputWidth || DTid.y >= outputHeight)
        return;

    // Figure out UV coordinates for this pixel
    const float2 UVcoord = float2(DTid.xy) / float2(outputWidth - 1, outputHeight - 1);

    // Do forward pass through the neural network with this pixel's UV coordinates as input
    float activations[LAYER_COUNT * MAX_NEURONS_PER_LAYER];
    forwardPass(UVcoord, nnWeights, nnBiases, activations);

    // Map NN output to RGB result
    const uint outputLayerActivationIndex = getNeuronDataIndex(LAYER_COUNT - 1, 0);
    const float3 result = float3(activations[outputLayerActivationIndex + 0],
                                 activations[outputLayerActivationIndex + 1],
                                 activations[outputLayerActivationIndex + 2]);

    // Store output
    Output[DTid.xy] = float4(result, 1);
}
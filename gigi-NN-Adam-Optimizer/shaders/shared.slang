
// 8888888b.            .d888 d8b                            
// 888  "Y88b          d88P"  Y8P                            
// 888    888          888                                   
// 888    888  .d88b.  888888 888 88888b.   .d88b.  .d8888b  
// 888    888 d8P  Y8b 888    888 888 "88b d8P  Y8b 88K      
// 888    888 88888888 888    888 888  888 88888888 "Y8888b. 
// 888  .d88P Y8b.     888    888 888  888 Y8b.          X88 
// 8888888P"   "Y8888  888    888 888  888  "Y8888   88888P' 

#define PI 3.141593

#define NUM_FREQUENCIES 8

#define LAYER_COUNT 4

#define MAX_LAYERS 5
#define NEURONS_PER_LAYER_0 /*$(Variable:NEURONS_PER_LAYER_0)*/
#define NEURONS_PER_LAYER_1 /*$(Variable:NEURONS_PER_LAYER_1)*/
#define NEURONS_PER_LAYER_2 /*$(Variable:NEURONS_PER_LAYER_2)*/
#define NEURONS_PER_LAYER_3 /*$(Variable:NEURONS_PER_LAYER_3)*/
#define NEURONS_PER_LAYER_4 /*$(Variable:NEURONS_PER_LAYER_4)*/

static const uint neuronsPerLayer[MAX_LAYERS] = {
    NEURONS_PER_LAYER_0, 
    NEURONS_PER_LAYER_1,
    NEURONS_PER_LAYER_2,
    NEURONS_PER_LAYER_3,
    NEURONS_PER_LAYER_4
};

#define CONNECTION_OFFSET_0 0
#define CONNECTION_OFFSET_1 0
#define CONNECTION_OFFSET_2 NEURONS_PER_LAYER_0 * NEURONS_PER_LAYER_1 + CONNECTION_OFFSET_1
#define CONNECTION_OFFSET_3 NEURONS_PER_LAYER_1 * NEURONS_PER_LAYER_2 + CONNECTION_OFFSET_2
#define CONNECTION_OFFSET_4 NEURONS_PER_LAYER_2 * NEURONS_PER_LAYER_3 + CONNECTION_OFFSET_3

#define NEURON_OFFSET_0 0
#define NEURON_OFFSET_1 NEURON_OFFSET_0 + /*$(Variable:NEURONS_PER_LAYER_0)*/
#define NEURON_OFFSET_2 NEURON_OFFSET_1 + /*$(Variable:NEURONS_PER_LAYER_1)*/
#define NEURON_OFFSET_3 NEURON_OFFSET_2 + /*$(Variable:NEURONS_PER_LAYER_2)*/
#define NEURON_OFFSET_4 NEURON_OFFSET_3 + /*$(Variable:NEURONS_PER_LAYER_3)*/


// Base offsets of connection data per layer
static const uint connectionDataBaseOffsets[MAX_LAYERS] = { 
    CONNECTION_OFFSET_0,
    CONNECTION_OFFSET_1,
    CONNECTION_OFFSET_2,
    CONNECTION_OFFSET_3,
    CONNECTION_OFFSET_4
};

// Base offsets of neuron data per layer
static const uint neuronDataBaseOffsets[MAX_LAYERS] = { 
    NEURON_OFFSET_0,
    NEURON_OFFSET_1,
    NEURON_OFFSET_2,
    NEURON_OFFSET_3,
    NEURON_OFFSET_4
};

#define MAX_NEURONS_PER_LAYER 64


// 888    888          888                                    
// 888    888          888                                    
// 888    888          888                                    
// 8888888888  .d88b.  888 88888b.   .d88b.  888d888 .d8888b  
// 888    888 d8P  Y8b 888 888 "88b d8P  Y8b 888P"   88K      
// 888    888 88888888 888 888  888 88888888 888     "Y8888b. 
// 888    888 Y8b.     888 888 d88P Y8b.     888          X88 
// 888    888  "Y8888  888 88888P"   "Y8888  888      88888P' 
//                         888                                
//                         888                                
//                         888                                



#define FLOAT_PACKING_CONSTANT 1000000.0f
int packFloat(float x)
{
    return int(x * FLOAT_PACKING_CONSTANT);
}

float unpackFloat(int x)
{
    return float(x) / FLOAT_PACKING_CONSTANT;
}

uint getNeuronDataIndex(uint layer, uint neuron)
{
  return neuronDataBaseOffsets[layer] + neuron;
}

uint getConnectionDataIndex(uint layer, uint neuronFrom, uint neuronTo)
{
  return connectionDataBaseOffsets[layer] + (neuronTo * neuronsPerLayer[layer - 1]) + neuronFrom;
}



// 8888888b.  888b    888  .d8888b.  
// 888   Y88b 8888b   888 d88P  Y88b 
// 888    888 88888b  888 888    888 
// 888   d88P 888Y88b 888 888        
// 8888888P"  888 Y88b888 888  88888 
// 888 T88b   888  Y88888 888    888 
// 888  T88b  888   Y8888 Y88b  d88P 
// 888   T88b 888    Y888  "Y8888P88 
                                  

// 32-bit Xorshift random number generator
uint xorshift(inout uint rngState)
{
  rngState ^= rngState << 13;
  rngState ^= rngState >> 17;
  rngState ^= rngState << 5;
  return rngState;
}

// Jenkins's "one at a time" hash function
uint jenkinsHash(uint x)
{
  x += x << 10;
  x ^= x >> 6;
  x += x << 3;
  x ^= x >> 11;
  x += x << 15;
  return x;
}

// Converts unsigned integer into float int range <0; 1) by using 23 most significant bits for mantissa
float uintToFloat(uint x)
{
    return asfloat(0x3f800000 | (x >> 9)) - 1.0f;
}

// Initialize RNG for given pixel, and frame number (Xorshift-based version)
uint initRNG(uint2 pixelCoords, uint2 resolution, uint frameNumber)
{
  uint seed = dot(pixelCoords, uint2(1, resolution.x)) ^ jenkinsHash(frameNumber);
  return jenkinsHash(seed);
}

// Return random float in <0; 1) range (Xorshift-based version)
float rand(inout uint rngState)
{
    return uintToFloat(xorshift(rngState));
}

float randInRange(inout uint rng, float range)
{
  return (rand(rng) * 2.0f - 1.0f) * range;
}



//        d8888          888    d8b                   888    d8b                            
//       d88888          888    Y8P                   888    Y8P                            
//      d88P888          888                          888                                   
//     d88P 888  .d8888b 888888 888 888  888  8888b.  888888 888  .d88b.  88888b.  .d8888b  
//    d88P  888 d88P"    888    888 888  888     "88b 888    888 d88""88b 888 "88b 88K      
//   d88P   888 888      888    888 Y88  88P .d888888 888    888 888  888 888  888 "Y8888b. 
//  d8888888888 Y88b.    Y88b.  888  Y8bd8P  888  888 Y88b.  888 Y88..88P 888  888      X88 
// d88P     888  "Y8888P  "Y888 888   Y88P   "Y888888  "Y888 888  "Y88P"  888  888  88888P' 


#ifndef ACTIVATION_FUNCTION
    #define ACTIVATION_FUNCTION leakyRelu
#endif

#ifndef ACTIVATION_FUNCTION_DERIV
    #define ACTIVATION_FUNCTION_DERIV leakyReluDeriv
#endif

#define LEAKY_RELU_SLOPE 0.01f

float relu(float x)
{
    return max(0.0f, x);
}

float reluDeriv(float x)
{
    return (x <= 0.0f) ? (0.0f) : (1.0f);
}

float leakyRelu(float x)
{
    return (x >= 0.0f) ? x : (x * LEAKY_RELU_SLOPE);
}

float leakyReluDeriv(float x)
{
    return (x <= 0.0f) ? (LEAKY_RELU_SLOPE) : (1.0f);
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

float sigmoidDeriv(float x)
{
    return x * (1.0f - x);
}

// Source: "Normalized Initialization" from "Understanding the difficulty of training deep feedforward neural networks"
// https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
float xavierUniformScale(inout uint rng, const uint nInputs, const uint nOutputs)
{
    return sqrt(6.0f / (nInputs + nOutputs));
}

// Source: "Positional Encoding" from "NeRF: Representing Scenes as Neural RadianceFields for View Synthesis"
// https://arxiv.org/pdf/2003.08934
void frequencyEncoding( const float2 input, inout float activations[LAYER_COUNT * MAX_NEURONS_PER_LAYER])
{
    const int inputCount = 2;

    int index = 0;
    [unroll]
    for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
    {
        const float p = PI * input[inputIndex];
        int modifier = 1;

        [unroll]
        for (int f = 0; f < NUM_FREQUENCIES; f++)
        {
            const float x = modifier * p;
            activations[index++] = sin(x);
            activations[index++] = cos(x);
            modifier *= 2;
        }
    }
}


#define IDENTITY_ENCODING  0
#define FREQUENCY_ENCODING 1

#define ENCODING_TYPE FREQUENCY_ENCODING

void encodeInput(const float2 input, 
                //  const int encodingType,
                 inout float  activations[LAYER_COUNT * MAX_NEURONS_PER_LAYER])
{
#if ENCODING_TYPE == IDENTITY_ENCODING
    activations[0] = input.x;
    activations[1] = input.y;
#elif ENCODING_TYPE == FREQUENCY_ENCODING
    frequencyEncoding(input, activations);
#endif
}


// 8888888888                                                  888 8888888b.                            
// 888                                                         888 888   Y88b                           
// 888                                                         888 888    888                           
// 8888888  .d88b.  888d888 888  888  888  8888b.  888d888 .d88888 888   d88P 8888b.  .d8888b  .d8888b  
// 888     d88""88b 888P"   888  888  888     "88b 888P"  d88" 888 8888888P"     "88b 88K      88K      
// 888     888  888 888     888  888  888 .d888888 888    888  888 888       .d888888 "Y8888b. "Y8888b. 
// 888     Y88..88P 888     Y88b 888 d88P 888  888 888    Y88b 888 888       888  888      X88      X88 
// 888      "Y88P"  888      "Y8888888P"  "Y888888 888     "Y88888 888       "Y888888  88888P'  88888P' 


void forwardPass(         float2 input,
                 RWBuffer<float> nnWeights,
                 RWBuffer<float> nnBiases,
                 inout    float  activations[LAYER_COUNT * MAX_NEURONS_PER_LAYER])
{
    // Encode input into first layer activations
    encodeInput(input, activations);
    
    // Calculate activations for every layer, going forward through the MLP network
    [unroll]
    for (uint layer = 1; layer < LAYER_COUNT; layer++)
    {
        const uint neuronCountCurrentLayer  = neuronsPerLayer[layer];
        const uint neuronCountPreviousLayer = neuronsPerLayer[layer - 1];
   
        [unroll(MAX_NEURONS_PER_LAYER)]
        for (uint neuron = 0; neuron < neuronCountCurrentLayer; neuron++)
        {
            const uint neuronDataIndex = getNeuronDataIndex(layer, neuron);
            
            // Evaluate neuron activation
            float neuronValue = nnBiases[neuronDataIndex];
            
            // Accumulate weighted contribution from all neurons connected to this neuron in previous layer
            for (uint previousNeuron = 0; previousNeuron < neuronCountPreviousLayer; previousNeuron++)
            {
                const uint weightDataIndex         = getConnectionDataIndex(layer, previousNeuron, neuron);
                const uint previousNeuronDataIndex = getNeuronDataIndex(layer - 1, previousNeuron);
                
                neuronValue += nnWeights[weightDataIndex] * activations[previousNeuronDataIndex];
            }
            
            activations[neuronDataIndex] = ACTIVATION_FUNCTION(neuronValue);
        }
    }
}


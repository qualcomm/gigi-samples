// Unnamed technique, shader gradientDescendVertices
/*$(ShaderResources)*/

// World origin
static float2 cameraPos = /*$(Variable:cameraPos)*/;

// World scale
static float2 scale = /*$(Variable:scale)*/;

// Frame dimensions (i.e. image resolution)
static float2 frameDim = /*$(Variable:frameDim)*/;

// Learning Rate
static float learningRate = /*$(Variable:learningRate)*/;

// Target Color
static float3 targetColor = /*$(Variable:targetColor)*/;

// Convert from
// screen coordinates [(0, 0), (W, H)] to
// world coordinates [(o.x - scale.x, o.y - scale.y), (o.x + scale.x, o.y + scale.y)]
[Differentiable]
float2 screenToWorld(float2 uv)
{
    float2 xy = uv / frameDim;
    float2 ndc = xy * 2.0f - 1.0f;
    return ndc * scale + cameraPos;
}

[PreferRecompute]
int _sign(float x)
{
    return sign(x);
}

[Differentiable]
[PreferRecompute]
float distanceToEdge(float2 u, float2 v, float2 pt)
{
    float2 e = v - u;
    float2 n = float2(-e.y, e.x);
    float2 d = pt - u;
    float n_dot_d = dot(n, d);

    // Compute the signed distance from a point to an edge
    if (dot(pt - u, v - u) < 0.f)
        return distanceToVertex(pt, u) * (_sign(n_dot_d)); // u is the closest vertex
    else if (dot(pt - v, u - v) < 0.f)
        return distanceToVertex(pt, v) * (_sign(n_dot_d)); // v is the closest vertex
    else
    {
        return n_dot_d / length(n); // closest edge
    }
}

[Differentiable]
[PreferRecompute]
float distanceToVertex(float2 xy2, float2 v)
{
    // Compute the distance from a point to a vertex
    return length(xy2 - v);
}

[Differentiable]
[PreferRecompute]
float distanceToTriangle(float2 xy, float2 v1, float2 v2, float2 v3)
{
    // Minimum distance to the edge of the triangle
    float d1 = distanceToEdge(v2, v1, xy);
    float d2 = distanceToEdge(v3, v2, xy);
    float d3 = distanceToEdge(v1, v3, xy);

    int sd1 = _sign(d1);
    int sd2 = _sign(d2);
    int sd3 = _sign(d3);

    if (sd1 > 0.f && sd2 > 0.f)
        return distanceToVertex(xy, v2); // v2 is the closest vertex
    else if (sd1 > 0.f && sd3 > 0.f)
        return distanceToVertex(xy, v1); // v1 is the closest vertex
    else if (sd2 > 0.f && sd3 > 0.f)
        return distanceToVertex(xy, v3); // v3 is the closest vertex
    else
        return max(max(d1, d2), d3);
}

[Differentiable]
[PreferRecompute]
float sigmoid(float x, float sigma)
{
    // Sigmoid function
    return 1.0 / (1.0 + exp(-x / sigma));
}

[Differentiable]
[PreferRecompute]
float softTriangle(float2 xy, float2 v1, float2 v2, float2 v3, float sigma)
{
    float d = distanceToTriangle(xy, v1, v2, v3);
    return sigmoid(-d, sigma);
}


[Differentiable]
float loss(float3 inputColor, float3 targetColor)
{
    float3 d = inputColor - targetColor;
    return dot(d, d);
}


[Differentiable]
float computeImageLoss(no_diff uint2 loc,
                       float v0x, float v0y,      // differentiable
                       float v1x, float v1y,      // differentiable
                       float v2x, float v2y,      // differentiable
                       float r, float g, float b, // differentiable
                       no_diff float2 tV0,
                       no_diff float2 tV1,
                       no_diff float2 tV2)
{
    float2 pixel_coord = loc;
    float2 screen_sample = pixel_coord + 0.5;

    // Convert screen space pixel coordinates to world space.
    float2 world_sample = screenToWorld(screen_sample);

    float3 c = float3(r, g, b);
    float3 background_color = float3(1.0, 1.0, 1.0);

    // Sigma for the sigmoid function
    float sigma = /*$(Variable:sigma)*/;

    float2 v0 = float2(v0x, v0y);
    float2 v1 = float2(v1x, v1y);
    float2 v2 = float2(v2x, v2y);

    float  in_hit   = softTriangle(world_sample, v0, v1, v2, sigma);
    float3 in_color = in_hit * c + (1 - in_hit) * background_color;

    float  target_hit   = softTriangle(world_sample, tV0, tV1, tV2, sigma);
    float3 target_color = target_hit * targetColor + (1 - target_hit) * background_color;

    return loss(in_color, target_color);
}



/*$(_compute:gradientDescent)*/ (uint3 DTid: SV_DispatchThreadID)
{
    if (DTid.x > frameDim.x || DTid.y > frameDim.y)
        return;


    float2 in_v0 = inputVertices[0];
    float2 in_v1 = inputVertices[1];
    float2 in_v2 = inputVertices[2];

    float2 target_v0 = targetVertices[0];
    float2 target_v1 = targetVertices[1];
    float2 target_v2 = targetVertices[2];

    float3 in_color = inputColor[0].rgb;
    float3 background_color = float3(1.0, 1.0, 1.0);

    float loss = computeImageLoss(DTid.xy, 
                                  in_v0.x, in_v0.y, 
                                  in_v1.x, in_v1.y, 
                                  in_v2.x, in_v2.y,
                                  in_color.r, in_color.g, in_color.b,
                                  target_v0, 
                                  target_v1, 
                                  target_v2);


    // Construct Differential Pairs for all elements {x,y} of input vertices                                  
    var in_v0x = diffPair(in_v0.x);
    var in_v0y = diffPair(in_v0.y);

    var in_v1x = diffPair(in_v1.x);
    var in_v1y = diffPair(in_v1.y);

    var in_v2x = diffPair(in_v2.x);
    var in_v2y = diffPair(in_v2.y);

    // Construct Differential Pairs for all input color components {r,g,b}
    var in_r = diffPair(in_color.r);
    var in_g = diffPair(in_color.g);
    var in_b = diffPair(in_color.b);

    // Use the bwd_diff function on computeImageLoss to update the Differential Pairs
    bwd_diff(computeImageLoss)(DTid.xy, 
                               in_v0x, in_v0y, 
                               in_v1x, in_v1y, 
                               in_v2x, in_v2y,
                               in_r, in_g, in_b,
                               target_v0, 
                               target_v1, 
                               target_v2,
                               loss);

    if (bool(/*$(Variable:descendVertices)*/))
    {
        // Update the input vertices using the differential (from the diffPair) and the learning rate
        in_v0 -= float2(in_v0x.d, in_v0y.d) * learningRate;
        in_v1 -= float2(in_v1x.d, in_v1y.d) * learningRate;
        in_v2 -= float2(in_v2x.d, in_v2y.d) * learningRate;

        // Store the updated vertices
        inputVertices[0] = in_v0;
        inputVertices[1] = in_v1;
        inputVertices[2] = in_v2;
    }

    if(bool(/*$(Variable:descendColor)*/))
    {
        // Update the input color using the differential (from the diffPair) and the learning rate
        in_color -= float3(in_r.d, in_g.d, in_b.d) * learningRate;

        // Store the updated color
        inputColor[0] = in_color;
    }
}

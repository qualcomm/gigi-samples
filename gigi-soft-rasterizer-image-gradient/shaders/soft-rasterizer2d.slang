// Unnamed technique, shader rasterizer2d
/*$(ShaderResources)*/

// World origin
static float2 cameraPos = /*$(Variable:cameraPos)*/;

// World scale
static float2 scale = /*$(Variable:scale)*/;

// Frame dimensions (i.e. image resolution)
static float2 frameDim = /*$(Variable:frameDim)*/;

// Learning Rate
static float learningRate = /*$(Variable:learningRate)*/;

// Convert from
// screen coordinates [(0, 0), (W, H)] to
// world coordinates [(o.x - scale.x, o.y - scale.y), (o.x + scale.x, o.y + scale.y)]
//
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


bool isNaN(float x)
{
    return (asuint(x) & 0x7fffffff) > 0x7f800000;
}

/*$(_compute:rasterize)*/ (uint3 DTid: SV_DispatchThreadID)
{
    if (DTid.x > frameDim.x || DTid.y > frameDim.y)
        return;

    uint index = DTid.x;

    float2 in_v0 = inputVertices[0];
    float2 in_v1 = inputVertices[1];
    float2 in_v2 = inputVertices[2];

    float2 target_v0 = targetVertices[0];
    float2 target_v1 = targetVertices[1];
    float2 target_v2 = targetVertices[2];

    float2 pixel_coord = DTid.xy;
    float2 screen_sample = pixel_coord + 0.5;
    
    // Convert screen space pixel coordinates to world space.
    float2 world_sample = screenToWorld(screen_sample);
    
    // Sigma for the sigmoid function
    float sigma = /*$(Variable:sigma)*/;

    float3 background_color = float3(1.0, 1.0, 1.0);

    float3 in_c     = inputColor[0];
    float  in_hit   = softTriangle(world_sample, in_v0, in_v1, in_v2, sigma);
    float3 in_color = in_hit * in_c + (1 - in_hit) * background_color;

    float3 target_c     = /*$(Variable:targetColor)*/;
    float  target_hit   = softTriangle(world_sample, target_v0, target_v1, target_v2, sigma);
    float3 target_color = target_hit * target_c + (1 - target_hit) * background_color;

    // Detect invalid interestion with input triangle if it has not been set yet
    if (isNaN(in_hit))
    {
        output[DTid.xy] = float4(target_color, 1);
    }
    else
    {
        output[DTid.xy] = float4(lerp(in_color, target_color, 1-in_hit), 1);
    }
}
